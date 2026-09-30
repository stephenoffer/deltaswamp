//! Refreshable vended credentials for the object store.
//!
//! A store was built from static options, so it held the credential it was
//! built with for as long as it lived: a snapshot cached for a whole job, or a
//! single long read or write, outlived the vended token (about an hour) and
//! failed partway through with a storage 403.
//!
//! A *slot* is a process-wide cell holding the latest vended options for one
//! credential source (one provider and operation, on the Python side). Python
//! pushes fresh options into it ahead of expiry, from its own refresher
//! thread; a store built with the slot's key in its options asks the slot for
//! its credential on every request. The store never calls into Python, so no
//! request waits on the GIL, and a thread holding the GIL while it waits on a
//! store cannot deadlock against it.
//!
//! A credential at or near its stated expiry is served only after waiting
//! (briefly) for a fresher one: a refresh that lands a moment late then
//! rescues the request instead of letting it fail.

use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, LazyLock, Mutex, RwLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use delta_kernel::object_store::aws::AwsCredential;
use delta_kernel::object_store::azure::AzureCredential;
use delta_kernel::object_store::gcp::GcpCredential;
use delta_kernel::object_store::CredentialProvider;
use percent_encoding::percent_decode_str;

/// The storage option naming a slot. Never forwarded to object_store.
pub const SLOT_KEY: &str = "deltaswamp_credential_slot";

/// How close to its expiry a credential must be for a request to wait for a
/// fresher one.
const NEAR_EXPIRY: Duration = Duration::from_secs(10);

/// How long a request waits for that fresher credential before using the
/// one it has (and, if it is dead, failing as it would have anyway).
const MAX_WAIT: Duration = Duration::from_secs(30);

#[derive(Default)]
struct SlotState {
    options: HashMap<String, String>,
    expires_at: Option<f64>,
    generation: u64,
}

#[derive(Default)]
struct Slot {
    state: RwLock<SlotState>,
    changed: tokio::sync::Notify,
}

static SLOTS: LazyLock<Mutex<HashMap<String, Arc<Slot>>>> = LazyLock::new(Default::default);

fn lookup(id: &str) -> Option<Arc<Slot>> {
    SLOTS
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .get(id)
        .cloned()
}

/// Publish `options` (a vended credential, as storage options) in slot `id`,
/// creating it on first use.
#[cfg(test)]
pub fn set(id: &str, options: HashMap<String, String>, expires_at: Option<f64>) {
    publish(id, options, expires_at, false);
}

/// [`set`]; `force`: a new generation even when unchanged, the answer to a
/// store's [`request_refresh`] that vending gave the same credential again,
/// so the store retries at once instead of waiting.
pub fn publish(id: &str, options: HashMap<String, String>, expires_at: Option<f64>, force: bool) {
    let slot = {
        let mut slots = SLOTS.lock().unwrap_or_else(|p| p.into_inner());
        slots.entry(id.to_string()).or_default().clone()
    };
    {
        let mut state = slot.state.write().unwrap_or_else(|p| p.into_inner());
        if !force && state.options == options && state.expires_at == expires_at {
            return;
        }
        state.options = options;
        state.expires_at = expires_at;
        state.generation += 1;
    }
    slot.changed.notify_waiters();
}

/// The generation of slot `id`'s credential, if the slot exists.
pub fn generation(id: &str) -> Option<u64> {
    lookup(id).map(|slot| {
        slot.state
            .read()
            .unwrap_or_else(|p| p.into_inner())
            .generation
    })
}

/// Slots whose stores were refused by storage (a 401 or 403) and want a
/// fresh credential now, and the condition the refresher waits on.
static REQUESTS: LazyLock<(
    Mutex<std::collections::BTreeSet<String>>,
    std::sync::Condvar,
)> = LazyLock::new(Default::default);

/// Ask whoever publishes slot `id` for a fresh credential now.
pub fn request_refresh(id: &str) {
    let (requests, ready) = &*REQUESTS;
    requests
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .insert(id.to_string());
    ready.notify_all();
}

/// The slots asked for a refresh, waiting up to `timeout` for one. Blocks:
/// the publisher's own thread calls it, without the GIL.
pub fn wait_requests(timeout: Duration) -> Vec<String> {
    let (requests, ready) = &*REQUESTS;
    let mut pending = requests.lock().unwrap_or_else(|p| p.into_inner());
    if pending.is_empty() {
        pending = ready
            .wait_timeout(pending, timeout)
            .map(|(guard, _)| guard)
            .unwrap_or_else(|p| p.into_inner().0);
    }
    std::mem::take(&mut *pending).into_iter().collect()
}

/// Whether slot `id` publishes a generation after `since` within `timeout`.
pub async fn wait_newer(id: &str, since: u64, timeout: Duration) -> bool {
    let Some(slot) = lookup(id) else {
        return false;
    };
    let deadline = Instant::now() + timeout;
    loop {
        let notified = slot.changed.notified();
        let now = slot
            .state
            .read()
            .unwrap_or_else(|p| p.into_inner())
            .generation;
        if now > since {
            return true;
        }
        let left = deadline.saturating_duration_since(Instant::now());
        if left.is_zero() {
            return false;
        }
        let _ = tokio::time::timeout(left.min(Duration::from_secs(1)), notified).await;
    }
}

/// Forget slot `id`. Stores built from it keep the credential they last saw.
pub fn remove(id: &str) {
    SLOTS.lock().unwrap_or_else(|p| p.into_inner()).remove(id);
}

fn now_seconds() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

fn near_expiry(expires_at: Option<f64>) -> bool {
    expires_at.is_some_and(|at| at - now_seconds() <= NEAR_EXPIRY.as_secs_f64())
}

fn value<'a>(options: &'a HashMap<String, String>, keys: &[&str]) -> Option<&'a str> {
    keys.iter().find_map(|key| {
        options
            .iter()
            .find(|(k, v)| k.eq_ignore_ascii_case(key) && !v.trim().is_empty())
            .map(|(_, v)| v.trim())
    })
}

pub const AWS_KEYS: &[&str] = &[
    "aws_access_key_id",
    "access_key_id",
    "aws_secret_access_key",
    "secret_access_key",
    "aws_session_token",
    "aws_token",
    "session_token",
    "token",
];

pub const AZURE_KEYS: &[&str] = &[
    "azure_storage_sas_key",
    "azure_storage_sas_token",
    "sas_key",
    "sas_token",
    "azure_storage_token",
    "bearer_token",
    "token",
];

/// The AWS credential in `options`, if they carry keys.
pub fn aws(options: &HashMap<String, String>) -> Option<AwsCredential> {
    Some(AwsCredential {
        key_id: value(options, &["aws_access_key_id", "access_key_id"])?.to_string(),
        secret_key: value(options, &["aws_secret_access_key", "secret_access_key"])?.to_string(),
        token: value(
            options,
            &["aws_session_token", "aws_token", "session_token", "token"],
        )
        .map(str::to_string),
    })
}

/// The Azure credential in `options`: a SAS, else a bearer token.
pub fn azure(options: &HashMap<String, String>) -> Option<AzureCredential> {
    if let Some(sas) = value(
        options,
        &[
            "azure_storage_sas_key",
            "azure_storage_sas_token",
            "sas_key",
            "sas_token",
        ],
    ) {
        return split_sas(sas).map(AzureCredential::SASToken);
    }
    value(options, &["azure_storage_token", "bearer_token", "token"])
        .map(|t| AzureCredential::BearerToken(t.to_string()))
}

/// The GCS bearer token in `options`.
pub fn gcp(options: &HashMap<String, String>) -> Option<GcpCredential> {
    value(options, crate::store::GCS_BEARER_KEYS).map(|t| GcpCredential {
        bearer: t.to_string(),
    })
}

/// A SAS query string as object_store's own builder splits it.
fn split_sas(sas: &str) -> Option<Vec<(String, String)>> {
    let sas = percent_decode_str(sas).decode_utf8().ok()?;
    let mut pairs = Vec::new();
    for pair in sas
        .trim_start_matches('?')
        .split('&')
        .filter(|s| !s.chars().all(char::is_whitespace))
    {
        let (k, v) = pair.trim().split_once('=')?;
        pairs.push((k.to_string(), v.to_string()));
    }
    Some(pairs)
}

/// A credential provider reading slot `id`, converting its options with
/// `convert`. `initial` serves when the slot is gone (another process after
/// a fork, or a slot Python has dropped).
pub struct SlotProvider<C> {
    id: String,
    initial: Arc<C>,
    convert: fn(&HashMap<String, String>) -> Option<C>,
    cached: Mutex<Option<(u64, Arc<C>)>>,
}

impl<C> SlotProvider<C> {
    pub fn new(id: &str, initial: C, convert: fn(&HashMap<String, String>) -> Option<C>) -> Self {
        Self {
            id: id.to_string(),
            initial: Arc::new(initial),
            convert,
            cached: Mutex::new(None),
        }
    }
}

impl<C> fmt::Debug for SlotProvider<C> {
    // Never the credential itself.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SlotProvider")
            .field("id", &self.id)
            .finish()
    }
}

#[async_trait]
impl<C: Send + Sync + 'static> CredentialProvider for SlotProvider<C> {
    type Credential = C;

    async fn get_credential(&self) -> delta_kernel::object_store::Result<Arc<C>> {
        let Some(slot) = lookup(&self.id) else {
            return Ok(self.initial.clone());
        };
        let deadline = Instant::now() + MAX_WAIT;
        loop {
            let notified = slot.changed.notified();
            let expires_at = slot
                .state
                .read()
                .unwrap_or_else(|p| p.into_inner())
                .expires_at;
            let left = deadline.saturating_duration_since(Instant::now());
            if !near_expiry(expires_at) || left.is_zero() {
                break;
            }
            // Woken by a publish, or re-checked each second.
            let _ = tokio::time::timeout(left.min(Duration::from_secs(1)), notified).await;
        }
        let state = slot.state.read().unwrap_or_else(|p| p.into_inner());
        let mut cached = self.cached.lock().unwrap_or_else(|p| p.into_inner());
        if let Some((generation, credential)) = cached.as_ref() {
            if *generation == state.generation {
                return Ok(credential.clone());
            }
        }
        let Some(credential) = (self.convert)(&state.options) else {
            // A publish without a usable credential (a cloud the store is not
            // for) keeps the last good one.
            return Ok(cached
                .as_ref()
                .map_or_else(|| self.initial.clone(), |(_, c)| c.clone()));
        };
        let credential = Arc::new(credential);
        *cached = Some((state.generation, credential.clone()));
        Ok(credential)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn opts(pairs: &[(&str, &str)]) -> HashMap<String, String> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    #[test]
    fn a_publish_reaches_a_provider_built_before_it() {
        let id = format!("t-{}", uuid::Uuid::new_v4());
        let first = opts(&[("aws_access_key_id", "k1"), ("aws_secret_access_key", "s1")]);
        set(&id, first.clone(), Some(now_seconds() + 3600.0));
        let provider = SlotProvider::new(&id, aws(&first).unwrap(), aws);
        let got = crate::runtime::block_on(provider.get_credential()).unwrap();
        assert_eq!(got.key_id, "k1");
        set(
            &id,
            opts(&[
                ("aws_access_key_id", "k2"),
                ("aws_secret_access_key", "s2"),
                ("aws_session_token", "t2"),
            ]),
            Some(now_seconds() + 3600.0),
        );
        let got = crate::runtime::block_on(provider.get_credential()).unwrap();
        assert_eq!(got.key_id, "k2");
        assert_eq!(got.token.as_deref(), Some("t2"));
        remove(&id);
        // Gone: the credential it was built with.
        let got = crate::runtime::block_on(provider.get_credential()).unwrap();
        assert_eq!(got.key_id, "k1");
    }

    #[test]
    fn a_request_near_expiry_waits_for_the_refresh() {
        let id = format!("t-{}", uuid::Uuid::new_v4());
        let old = opts(&[("google_bearer_token", "old")]);
        set(&id, old.clone(), Some(now_seconds() + 1.0));
        let provider = SlotProvider::new(&id, gcp(&old).unwrap(), gcp);
        let publisher = {
            let id = id.clone();
            std::thread::spawn(move || {
                std::thread::sleep(Duration::from_millis(300));
                set(
                    &id,
                    opts(&[("google_bearer_token", "new")]),
                    Some(now_seconds() + 3600.0),
                );
            })
        };
        let got = crate::runtime::block_on(provider.get_credential()).unwrap();
        publisher.join().unwrap();
        assert_eq!(got.bearer, "new");
        remove(&id);
    }

    #[test]
    fn sas_and_bearer_tokens_convert() {
        let sas = azure(&opts(&[("azure_storage_sas_key", "?sv=1&sig=a%2Bb")])).unwrap();
        match sas {
            AzureCredential::SASToken(pairs) => {
                assert_eq!(
                    pairs,
                    vec![("sv".into(), "1".into()), ("sig".into(), "a+b".into())]
                )
            }
            other => panic!("{other:?}"),
        }
        assert!(matches!(
            azure(&opts(&[("bearer_token", "ey")])),
            Some(AzureCredential::BearerToken(t)) if t == "ey"
        ));
        assert!(aws(&opts(&[("aws_access_key_id", "k")])).is_none());
    }
}
