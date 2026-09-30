//! Object-store construction from a table URL plus vended credential options.
//!
//! Two deviations from what `parse_url_opts` alone would give you, both of them
//! bugs in the wider ecosystem that we decline to inherit.
//!
//! **GCS bearer tokens.** Unity Catalog vends GCS credentials as a raw OAuth2
//! access token (`gcp_oauth_token.oauth_token`). delta-rs maps that onto
//! `google_application_credentials`, which `object_store` interprets as a
//! *filesystem path to an ADC JSON file* -- so the token is treated as a
//! filename and the read fails. `GoogleConfigKey::BearerToken` would be the
//! clean fix, but it only exists in object_store 0.14+, and delta_kernel 0.28
//! pins 0.13.2. So we build the GCS store directly and hand the token over as a
//! credential provider, which is what DuckDB had to do for the same reason.
//!
//! **Azure endpoints.** We never rely on account-name inference. It happens to
//! work for `*.blob.core.windows.net` and silently breaks Azurite, private-link
//! DNS, and the sovereign clouds. A fully qualified
//! `abfss://container@account.dfs.<suffix>/` URL names its host, so the
//! endpoint is taken from it (`https://account.blob.<suffix>`); only a host-less
//! `az://container/` URL needs one passed. The endpoint is taken only from a
//! host under an Azure Storage suffix, and only for the account the options
//! name: the connection's secret otherwise went to whatever host a catalog
//! entry or a log named.
//!
//! **Sovereign-cloud URLs.** object_store's URL parser knows only the
//! `core.windows.net` and Fabric hosts; `abfss://c@a.dfs.core.chinacloudapi.cn/`
//! (or `usgovcloudapi.net`) failed with "URL did not match any known pattern".
//! Such a URL is handed to object_store as `az://c/<path>` with the account
//! named and the endpoint set, which addresses the same objects. The Python
//! delta-rs engine does the same rewrite (`_storage.azure_store_location`).

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use delta_kernel::object_store::aws::AwsCredentialProvider;
use delta_kernel::object_store::azure::AzureCredentialProvider;
use delta_kernel::object_store::client::StaticCredentialProvider;
use delta_kernel::object_store::gcp::{
    GcpCredential, GcpCredentialProvider, GoogleCloudStorageBuilder,
};
use delta_kernel::object_store::{parse_url_opts, BackoffConfig, DynObjectStore, RetryConfig};
use url::Url;

use crate::credential_slot::{self, SlotProvider, SLOT_KEY};
use crate::error::{NativeError, Result};

/// Option keys accepted for a GCS OAuth bearer token, most specific first.
///
/// `gcp_oauth_token` is the name Unity Catalog uses in its credential response,
/// so accepting it verbatim means callers can pass the vended block through
/// without renaming fields.
pub const GCS_BEARER_KEYS: &[&str] = &["google_bearer_token", "gcp_oauth_token", "bearer_token"];

/// Keys that must never be forwarded to `parse_url_opts`, because object_store
/// would misinterpret them.
const GCS_INTERNAL_KEYS: &[&str] = GCS_BEARER_KEYS;

fn is_gcs(url: &Url) -> bool {
    matches!(url.scheme(), "gs" | "gcs")
}

fn is_azure(url: &Url) -> bool {
    matches!(
        url.scheme(),
        "abfs" | "abfss" | "az" | "adl" | "wasb" | "wasbs"
    )
}

fn gcs_bearer_token(options: &HashMap<String, String>) -> Option<&str> {
    // Keys are matched case-insensitively, as object_store does, and an empty
    // value under one alias must not hide a real token under the next.
    GCS_BEARER_KEYS.iter().find_map(|k| {
        options
            .iter()
            .find(|(key, value)| key.eq_ignore_ascii_case(k) && !value.is_empty())
            .map(|(_, value)| value.as_str())
    })
}

/// True if the options name an Azure endpoint, or ask for the emulator
/// (which object_store points at Azurite itself). Keys are case-insensitive,
/// as `parse_url_opts` lowercases them; every alias object_store accepts for
/// the endpoint counts.
fn has_azure_endpoint(options: &HashMap<String, String>) -> bool {
    options.iter().any(|(k, v)| {
        let k = k.to_ascii_lowercase();
        let endpoint = matches!(
            k.as_str(),
            "azure_storage_endpoint" | "azure_endpoint" | "endpoint"
        ) && !v.trim().is_empty();
        let emulator = matches!(k.as_str(), "azure_storage_use_emulator" | "use_emulator")
            && matches!(
                v.trim().to_ascii_lowercase().as_str(),
                "true" | "1" | "yes" | "on"
            );
        endpoint || emulator
    })
}

/// Host suffixes object_store's Azure URL parser recognises after
/// `container@account.`.
const PARSEABLE_AZURE_SUFFIXES: &[&str] = &[
    "dfs.core.windows.net",
    "blob.core.windows.net",
    "dfs.fabric.microsoft.com",
    "blob.fabric.microsoft.com",
];

/// DNS suffixes Azure Storage serves accounts under: the public cloud (its
/// private-link aliases included), the sovereign clouds and Fabric OneLake.
/// Mirrors `credentials.databricks.AZURE_STORAGE_SUFFIXES`.
const AZURE_STORAGE_SUFFIXES: &[&str] = &[
    "core.windows.net",
    "core.chinacloudapi.cn",
    "core.usgovcloudapi.net",
    "core.cloudapi.de",
    "fabric.microsoft.com",
];

fn is_azure_storage_host(host: &str) -> bool {
    let host = host.trim_end_matches('.');
    AZURE_STORAGE_SUFFIXES
        .iter()
        .any(|s| host.ends_with(&format!(".{s}")))
}

fn option_value<'a>(options: &'a HashMap<String, String>, keys: &[&str]) -> Option<&'a str> {
    options
        .iter()
        .find(|(k, v)| keys.iter().any(|key| k.eq_ignore_ascii_case(key)) && !v.trim().is_empty())
        .map(|(_, v)| v.trim())
}

fn has_option(options: &HashMap<String, String>, keys: &[&str]) -> bool {
    options
        .iter()
        .any(|(k, v)| keys.iter().any(|key| k.eq_ignore_ascii_case(key)) && !v.trim().is_empty())
}

/// The URL and options object_store is given for an Azure `url`.
///
/// See the module docs: the endpoint comes from a fully qualified host, and a
/// host object_store cannot parse is rewritten to `az://container/path`.
fn azure_target(
    url: &Url,
    options: &HashMap<String, String>,
) -> Result<(Url, HashMap<String, String>)> {
    if matches!(url.scheme(), "wasb" | "wasbs") {
        return Err(NativeError::Invalid(format!(
            "{url} uses the legacy WASB driver scheme, which object_store does not \
             support; address the same data as abfss://<container>@<account>.dfs.<suffix>/<path>"
        )));
    }
    let mut out = options.clone();
    let host = url.host_str().unwrap_or_default().to_ascii_lowercase();
    let qualified = matches!(url.scheme(), "abfs" | "abfss" | "az")
        && !url.username().is_empty()
        && host.contains('.');
    if !qualified {
        if !has_azure_endpoint(options) {
            // Refusing here rather than letting a confusing 403 surface later.
            return Err(NativeError::Invalid(format!(
                "Azure URL {url} has no explicit endpoint: it names no account host, and \
                 relying on account-name inference breaks Azurite, private-link DNS and \
                 sovereign clouds; pass azure_endpoint, or use \
                 abfss://<container>@<account>.dfs.<suffix>/<path>."
            )));
        }
        return Ok((url.clone(), out));
    }
    let (account, rest) = host.split_once('.').unwrap_or((host.as_str(), ""));
    let fabric = rest.ends_with("fabric.microsoft.com");
    if !has_azure_endpoint(options) {
        // The connection's Azure secret goes to the host the URL names, and
        // the URL comes from a catalog entry or a log others may write: a
        // host outside Azure Storage's domains, or another account than the
        // options name, is used only when the caller names the endpoint.
        if !is_azure_storage_host(&host) {
            return Err(NativeError::Invalid(format!(
                "{url} names the host {host}, which is not an Azure Storage domain \
                 (*.core.windows.net, the sovereign clouds, Fabric); storage credentials are \
                 sent there only when you pass it as azure_storage_endpoint"
            )));
        }
        if let Some(named) = option_value(options, &["azure_storage_account_name", "account_name"])
        {
            if !fabric && !named.eq_ignore_ascii_case(account) {
                return Err(NativeError::Invalid(format!(
                    "{url} is in the storage account {account}, but the connection's \
                     credentials are for {named}; refused rather than send them to another \
                     account"
                )));
            }
        }
    }
    if !has_azure_endpoint(options) && !fabric {
        // object_store speaks only the Blob API, so a dfs host maps to its
        // blob sibling; the suffix is kept, which is what makes this right on
        // sovereign clouds.
        let service = rest
            .strip_prefix("dfs.")
            .map_or(rest.to_string(), |s| format!("blob.{s}"));
        out.insert(
            "azure_storage_endpoint".to_string(),
            format!("https://{account}.{service}"),
        );
    }
    if PARSEABLE_AZURE_SUFFIXES.iter().any(|s| host.ends_with(s)) {
        return Ok((url.clone(), out));
    }
    if !has_option(options, &["azure_storage_account_name", "account_name"]) {
        out.insert(
            "azure_storage_account_name".to_string(),
            account.to_string(),
        );
    }
    let rewritten = Url::parse(&format!("az://{}{}", url.username(), url.path()))
        .map_err(|e| NativeError::Invalid(format!("cannot address {url} as az://: {e}")))?;
    Ok((rewritten, out))
}

/// Storage-option keys for the client's retry policy, as delta-rs reads them.
/// object_store has no configuration key for any of these, so without this
/// the kernel retried 10 times over up to 180 s whatever the caller asked.
const RETRY_KEYS: [&str; 5] = [
    "max_retries",
    "retry_timeout",
    "backoff_config.init_backoff",
    "backoff_config.max_backoff",
    "backoff_config.base",
];

/// The longest retry duration taken. object_store's backoff turns durations
/// into f64 seconds and back (`Duration::from_secs_f64`, which panics past
/// u64 seconds), so a value near that bound -- "20000000000000000000s" --
/// panicked inside a read instead of being refused. A century is far more
/// than any retry policy means.
const MAX_RETRY_DURATION: Duration = Duration::from_secs(100 * 365 * 24 * 3600);

/// A duration as delta-rs reads one: humantime's form (`30s`, `30 s`,
/// `2 minutes`, `1h 30m`, `1.5s`, `1d`), which object_store's own config
/// parsing uses. Bare seconds ("2") are not one: delta-rs refuses them, and
/// taking them here let the kernel read with options delta-rs then failed on
/// after can() had named it.
fn parse_duration(key: &str, text: &str) -> Result<Duration> {
    let parsed = humantime::parse_duration(text.trim()).map_err(|e| {
        NativeError::Invalid(format!(
            "{key}: {text:?} is not a duration such as 30s, 500ms or 2 minutes ({e})"
        ))
    })?;
    if parsed > MAX_RETRY_DURATION {
        return Err(NativeError::Invalid(format!(
            "{key}: {text:?} is longer than a retry policy can use (at most 100 years)"
        )));
    }
    Ok(parsed)
}

/// The retry policy the options ask for, or None to keep object_store's.
///
/// Read as delta-rs reads the same keys, so that both engines accept the same
/// options: the key names exactly (delta-rs ignores `RETRY_TIMEOUT`), an
/// unsigned integer for `max_retries` and a float for the base, untrimmed.
/// Beyond delta-rs, values object_store's backoff would panic on are refused
/// (`validate_retry_options` applies the same rules for every engine): a
/// base that is not a finite number above 1 (the next backoff is drawn from
/// `init..prev * base`, empty otherwise) and a zero initial backoff.
pub fn retry_config(options: &HashMap<String, String>) -> Result<Option<RetryConfig>> {
    let get = |key: &str| options.get(key).map(String::as_str);
    if RETRY_KEYS.iter().all(|k| get(k).is_none()) {
        return Ok(None);
    }
    let mut config = RetryConfig::default();
    let mut backoff = BackoffConfig::default();
    if let Some(v) = get("max_retries") {
        config.max_retries = v.parse().map_err(|_| {
            NativeError::Invalid(format!(
                "max_retries must be a non-negative integer, got {v:?}"
            ))
        })?;
    }
    if let Some(v) = get("retry_timeout") {
        config.retry_timeout = parse_duration("retry_timeout", v)?;
    }
    if let Some(v) = get("backoff_config.init_backoff") {
        backoff.init_backoff = parse_duration("backoff_config.init_backoff", v)?;
        if backoff.init_backoff.is_zero() {
            return Err(NativeError::Invalid(format!(
                "backoff_config.init_backoff must be longer than zero, got {v:?}"
            )));
        }
    }
    if let Some(v) = get("backoff_config.max_backoff") {
        backoff.max_backoff = parse_duration("backoff_config.max_backoff", v)?;
    }
    if let Some(v) = get("backoff_config.base") {
        let base: f64 = v.parse().map_err(|_| {
            NativeError::Invalid(format!("backoff_config.base must be a number, got {v:?}"))
        })?;
        if !(base.is_finite() && base > 1.0) {
            return Err(NativeError::Invalid(format!(
                "backoff_config.base must be a finite number above 1, got {v:?}"
            )));
        }
        backoff.base = base;
    }
    config.backoff = backoff;
    Ok(Some(config))
}

/// A refreshing credential provider for the store's cloud (`credential_slot`).
enum Refreshing {
    Aws(AwsCredentialProvider),
    Azure(AzureCredentialProvider),
}

/// `options` without the keys (case-insensitively) in `keys`.
fn without(options: &HashMap<String, String>, keys: &[&str]) -> HashMap<String, String> {
    options
        .iter()
        .filter(|(k, _)| !keys.iter().any(|key| k.eq_ignore_ascii_case(key)))
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect()
}

/// `parse_url_opts`, with a retry policy and a refreshing credential:
/// object_store's URL parser takes neither, so the builder for the URL's
/// cloud is built here, with the same key handling (unknown keys are
/// ignored, as there).
fn build_with(
    url: &Url,
    options: &HashMap<String, String>,
    retry: Option<RetryConfig>,
    refreshing: Option<Refreshing>,
) -> Result<Arc<DynObjectStore>> {
    use delta_kernel::object_store::aws::AmazonS3Builder;
    use delta_kernel::object_store::azure::MicrosoftAzureBuilder;

    macro_rules! build {
        ($builder:ty, $credentials:expr) => {{
            let mut builder = options.iter().fold(
                <$builder>::new().with_url(url.to_string()),
                |builder, (key, value)| match key.to_ascii_lowercase().parse() {
                    Ok(k) => builder.with_config(k, value),
                    Err(_) => builder,
                },
            );
            if let Some(retry) = retry {
                builder = builder.with_retry(retry);
            }
            if let Some(credentials) = $credentials {
                builder = builder.with_credentials(credentials);
            }
            Ok(Arc::new(builder.build()?) as Arc<DynObjectStore>)
        }};
    }
    let aws = match &refreshing {
        Some(Refreshing::Aws(p)) => Some(p.clone()),
        _ => None,
    };
    let azure = match &refreshing {
        Some(Refreshing::Azure(p)) => Some(p.clone()),
        _ => None,
    };
    match url.scheme() {
        "s3" | "s3a" => build!(AmazonS3Builder, aws),
        "gs" | "gcs" => build!(GoogleCloudStorageBuilder, None::<GcpCredentialProvider>),
        "abfs" | "abfss" | "az" | "adl" | "azure" => build!(MicrosoftAzureBuilder, azure),
        _ => {
            // Local files and memory make no network requests to retry, and
            // take no credential.
            let pairs = options.iter().map(|(k, v)| (k.as_str(), v.as_str()));
            let (store, _path) = parse_url_opts(url, pairs)?;
            Ok(Arc::from(store))
        }
    }
}

fn is_s3(url: &Url) -> bool {
    matches!(url.scheme(), "s3" | "s3a")
}

/// Build an object store for `url`, honoring vended credentials.
///
/// With a credential slot named in the options (`credential_slot::SLOT_KEY`)
/// and a vended credential for the URL's cloud, the store reads its
/// credential from the slot on every request, so it outlives the one it was
/// built with.
pub fn build_store(url: &Url, options: &HashMap<String, String>) -> Result<Arc<DynObjectStore>> {
    let slot = option_value(options, &[SLOT_KEY]).map(str::to_string);
    let stripped;
    let options = if slot.is_some() {
        stripped = without(options, &[SLOT_KEY]);
        &stripped
    } else {
        options
    };
    let retry = retry_config(options)?;
    if is_azure(url) {
        let (target, options) = azure_target(url, options)?;
        if let Some(slot) = &slot {
            if let Some(credential) = credential_slot::azure(&options) {
                let provider: AzureCredentialProvider =
                    Arc::new(SlotProvider::new(slot, credential, credential_slot::azure));
                let options = without(&options, credential_slot::AZURE_KEYS);
                let store =
                    build_with(&target, &options, retry, Some(Refreshing::Azure(provider)))?;
                return Ok(retrying(store, slot));
            }
        }
        if retry.is_some() {
            return build_with(&target, &options, retry, None);
        }
        let pairs = options.iter().map(|(k, v)| (k.as_str(), v.as_str()));
        let (store, _path) = parse_url_opts(&target, pairs)?;
        return Ok(Arc::from(store));
    }

    if is_gcs(url) {
        if let Some(token) = gcs_bearer_token(options) {
            return build_gcs_with_bearer(url, options, token, retry, slot.as_deref());
        }
    }

    if is_s3(url) {
        if let Some(slot) = &slot {
            if let Some(credential) = credential_slot::aws(options) {
                let provider: AwsCredentialProvider =
                    Arc::new(SlotProvider::new(slot, credential, credential_slot::aws));
                let options = without(options, credential_slot::AWS_KEYS);
                let store = build_with(url, &options, retry, Some(Refreshing::Aws(provider)))?;
                return Ok(retrying(store, slot));
            }
        }
    }

    if retry.is_some() {
        return build_with(url, options, retry, None);
    }
    let pairs = options.iter().map(|(k, v)| (k.as_str(), v.as_str()));
    let (store, _path) = parse_url_opts(url, pairs)?;
    Ok(Arc::from(store))
}

/// Whether the store at `url` honours put-if-absent, found by trying it.
///
/// Two `PutMode::Create` puts of one sentinel under `_delta_log/`: a store
/// that accepts the second ignores the condition every commit relies on
/// (some S3-compatible stores drop `If-None-Match`). The sentinel is deleted
/// afterwards; its name matches no log file, so a reader listing the log in
/// between skips it.
pub fn probe_put_if_absent(url: &Url, options: &HashMap<String, String>) -> Result<bool> {
    use delta_kernel::object_store::path::Path;
    use delta_kernel::object_store::{Error, ObjectStoreExt, PutMode, PutOptions, PutPayload};

    let store = build_store(url, options)?;
    let root = Path::from_url_path(url.path()).map_err(Error::from)?;
    let sentinel = root.join("_delta_log").join(format!(
        ".deltaswamp-put-if-absent-probe-{}",
        uuid::Uuid::new_v4()
    ));
    crate::runtime::block_on(async {
        let put = || {
            store.put_opts(
                &sentinel,
                PutPayload::from_static(b"probe"),
                PutOptions::from(PutMode::Create),
            )
        };
        match put().await {
            Ok(_) => {}
            // aws_conditional_put=disabled: no put-if-absent at all.
            Err(Error::NotImplemented { .. }) => return Ok(false),
            Err(e) => return Err(e.into()),
        }
        let second = put().await;
        let _ = store.delete(&sentinel).await;
        match second {
            Err(Error::AlreadyExists { .. }) | Err(Error::Precondition { .. }) => Ok(true),
            Ok(_) | Err(Error::NotImplemented { .. }) => Ok(false),
            Err(e) => Err(e.into()),
        }
    })
}

/// The entries directly under `url`: each child's name and whether it is a
/// directory (a common prefix), through the store the kernel writes with.
///
/// For checks made before a table exists, such as refusing to create one in a
/// directory that already holds other files. Listing through delta-rs there
/// could not use a GCS bearer token or endpoint: it fell back to ambient
/// credentials (the GCE metadata server, which timed out off GCP) and so
/// listed nothing, or listed as a different principal.
pub fn list_directory(url: &Url, options: &HashMap<String, String>) -> Result<Vec<(String, bool)>> {
    use delta_kernel::object_store::path::Path;

    let store = build_store(url, options)?;
    let root = Path::from_url_path(url.path()).map_err(delta_kernel::object_store::Error::from)?;
    let listing = crate::runtime::block_on(async { store.list_with_delimiter(Some(&root)).await })?;
    let name = |p: &Path| p.filename().unwrap_or_default().to_string();
    let mut entries: Vec<(String, bool)> = listing
        .common_prefixes
        .iter()
        .map(|p| (name(p), true))
        .chain(listing.objects.iter().map(|o| (name(&o.location), false)))
        .filter(|(n, _)| !n.is_empty())
        .collect();
    entries.sort();
    Ok(entries)
}

/// Build a GCS store authenticated with a raw OAuth2 bearer token.
fn build_gcs_with_bearer(
    url: &Url,
    options: &HashMap<String, String>,
    token: &str,
    retry: Option<RetryConfig>,
    slot: Option<&str>,
) -> Result<Arc<DynObjectStore>> {
    let mut builder = GoogleCloudStorageBuilder::new().with_url(url.as_str());
    if let Some(retry) = retry {
        builder = builder.with_retry(retry);
    }

    // Forward everything except our own token keys; object_store would reject
    // them as unknown configuration.
    for (k, v) in options {
        let k = k.to_ascii_lowercase();
        if GCS_INTERNAL_KEYS.contains(&k.as_str()) {
            continue;
        }
        if let Ok(key) = k.parse() {
            builder = builder.with_config(key, v);
        }
    }

    let credential = GcpCredential {
        bearer: token.to_string(),
    };
    let provider: GcpCredentialProvider = match slot {
        Some(slot) => Arc::new(SlotProvider::new(slot, credential, credential_slot::gcp)),
        None => Arc::new(StaticCredentialProvider::new(credential)),
    };
    let store: Arc<DynObjectStore> = Arc::new(builder.with_credentials(provider).build()?);
    Ok(match slot {
        Some(slot) => retrying(store, slot),
        None => store,
    })
}

/// `store`, asking slot `slot` for a fresh credential once when storage
/// refuses the one it has (see `crate::auth_retry`).
fn retrying(store: Arc<DynObjectStore>, slot: &str) -> Arc<DynObjectStore> {
    Arc::new(crate::auth_retry::AuthRetryStore::new(store, slot))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn durations_parse_as_delta_rs_reads_them() {
        let d = |t: &str| parse_duration("k", t);
        assert_eq!(d("30s").unwrap(), Duration::from_secs(30));
        assert_eq!(d("30 s").unwrap(), Duration::from_secs(30));
        assert_eq!(d(" 30s ").unwrap(), Duration::from_secs(30));
        assert_eq!(d("2 minutes").unwrap(), Duration::from_secs(120));
        assert_eq!(d("1d").unwrap(), Duration::from_secs(86400));
        assert_eq!(d("500ms").unwrap(), Duration::from_millis(500));
        assert_eq!(d("1h 30m").unwrap(), Duration::from_secs(5400));
        assert_eq!(d("1.5s").unwrap(), Duration::from_millis(1500));
        // delta-rs refuses all of these.
        for bad in ["2", "1e19", "-1s", "30S", "nan", "", "soon", "+1s"] {
            assert!(d(bad).is_err(), "{bad}");
        }
        // Accepted by delta-rs, but a panic in object_store's backoff.
        assert!(d("20000000000000000000s").is_err());
        assert!(d("1e30").is_err());
    }

    #[test]
    fn retry_options_become_a_retry_config() {
        assert!(retry_config(&opts(&[("aws_region", "us-east-1")]))
            .unwrap()
            .is_none());
        let config = retry_config(&opts(&[("max_retries", "0"), ("retry_timeout", "1s")]))
            .unwrap()
            .unwrap();
        assert_eq!(config.max_retries, 0);
        assert_eq!(config.retry_timeout, Duration::from_secs(1));
        // delta-rs reads the keys as spelled, and ignores this one.
        assert!(retry_config(&opts(&[("RETRY_TIMEOUT", "1s")]))
            .unwrap()
            .is_none());
        for (key, bad) in [
            ("max_retries", "many"),
            ("max_retries", " 3 "),
            ("max_retries", "-1"),
            ("backoff_config.base", "nan"),
            ("backoff_config.base", "-5"),
            ("backoff_config.base", "1"),
            ("backoff_config.base", "inf"),
            ("backoff_config.init_backoff", "0s"),
        ] {
            assert!(retry_config(&opts(&[(key, bad)])).is_err(), "{key}={bad}");
        }
    }

    #[test]
    fn a_store_with_a_retry_policy_builds_on_every_cloud() {
        let retry = opts(&[("max_retries", "1"), ("retry_timeout", "2s")]);
        for url in ["s3://b/t", "gs://b/t", "file:///tmp/t"] {
            build_store(&Url::parse(url).unwrap(), &retry).unwrap();
        }
        let mut azure = retry.clone();
        azure.insert("azure_storage_account_name".into(), "acct".into());
        azure.insert("azure_storage_account_key".into(), "a2V5".into());
        build_store(
            &Url::parse("abfss://c@acct.dfs.core.windows.net/t").unwrap(),
            &azure,
        )
        .unwrap();
    }

    fn opts(pairs: &[(&str, &str)]) -> HashMap<String, String> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    #[test]
    fn a_directory_lists_its_files_and_subdirectories() {
        let dir = std::env::temp_dir().join(format!("ds-list-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(dir.join("_delta_log")).unwrap();
        std::fs::create_dir_all(dir.join("p=1")).unwrap();
        std::fs::write(dir.join("p=1").join("a.parquet"), b"x").unwrap();
        std::fs::write(dir.join("notes.txt"), b"x").unwrap();
        std::fs::write(dir.join("_delta_log").join("0.json"), b"x").unwrap();
        let url = Url::from_directory_path(&dir).unwrap();
        let listed = list_directory(&url, &HashMap::new()).unwrap();
        std::fs::remove_dir_all(&dir).unwrap();
        assert_eq!(
            listed,
            vec![
                ("_delta_log".to_string(), true),
                ("notes.txt".to_string(), false),
                ("p=1".to_string(), true),
            ]
        );
    }

    #[test]
    fn local_file_store_builds() {
        let url = Url::parse("file:///tmp/some-table/").unwrap();
        assert!(build_store(&url, &HashMap::new()).is_ok());
    }

    #[test]
    fn host_less_azure_without_explicit_endpoint_is_refused() {
        // The failure mode this prevents is a 403 several layers away, on
        // Azurite / private link / sovereign clouds.
        let url = Url::parse("az://container/t/").unwrap();
        let err = build_store(&url, &opts(&[("azure_storage_sas_key", "sig")])).unwrap_err();
        assert!(err.to_string().contains("no explicit endpoint"), "{err}");
    }

    #[test]
    fn a_fully_qualified_azure_host_supplies_the_endpoint() {
        // The host is in the URL, so nothing is inferred; the kernel refused
        // this while delta-rs served it with the same options.
        let url = Url::parse("abfss://container@account.dfs.core.windows.net/t/").unwrap();
        let (target, o) = azure_target(&url, &opts(&[("azure_storage_sas_key", "sig")])).unwrap();
        assert_eq!(target, url);
        assert_eq!(
            o["azure_storage_endpoint"],
            "https://account.blob.core.windows.net"
        );
        assert!(build_store(&url, &opts(&[("azure_storage_sas_key", "sig=x")])).is_ok());
    }

    #[test]
    fn sovereign_cloud_urls_are_rewritten_to_the_az_form() {
        for (host, endpoint) in [
            (
                "acct.dfs.core.chinacloudapi.cn",
                "https://acct.blob.core.chinacloudapi.cn",
            ),
            (
                "acct.dfs.core.usgovcloudapi.net",
                "https://acct.blob.core.usgovcloudapi.net",
            ),
            (
                "acct.blob.core.cloudapi.de",
                "https://acct.blob.core.cloudapi.de",
            ),
        ] {
            let url = Url::parse(&format!("abfss://cont@{host}/a/b%20c/")).unwrap();
            let (target, o) =
                azure_target(&url, &opts(&[("azure_storage_sas_key", "sig")])).unwrap();
            assert_eq!(target.as_str(), "az://cont/a/b%20c/", "{host}");
            assert_eq!(o["azure_storage_account_name"], "acct");
            assert_eq!(o["azure_storage_endpoint"], endpoint);
            assert!(build_store(&url, &opts(&[("azure_storage_sas_key", "sig=x")])).is_ok());
        }
    }

    #[test]
    fn an_explicit_endpoint_and_account_win_over_the_host() {
        let url = Url::parse("abfss://cont@acct.dfs.core.chinacloudapi.cn/t/").unwrap();
        let given = opts(&[
            (
                "AZURE_STORAGE_ENDPOINT",
                "http://127.0.0.1:10000/devstoreaccount1",
            ),
            ("account_name", "devstoreaccount1"),
        ]);
        let (_, o) = azure_target(&url, &given).unwrap();
        assert!(!o.contains_key("azure_storage_endpoint"));
        assert!(!o.contains_key("azure_storage_account_name"));
    }

    #[test]
    fn a_host_outside_azure_storage_needs_an_explicit_endpoint() {
        // The connection's SAS went to https://acct.blob.evil.example.
        let url = Url::parse("abfss://c@acct.dfs.evil.example/t/").unwrap();
        let err = azure_target(&url, &opts(&[("azure_storage_sas_key", "sig")])).unwrap_err();
        assert!(
            err.to_string().contains("not an Azure Storage domain"),
            "{err}"
        );
        let given = opts(&[
            ("azure_storage_sas_key", "sig"),
            ("azure_storage_endpoint", "https://acct.blob.evil.example"),
        ]);
        assert!(azure_target(&url, &given).is_ok());
        let private = Url::parse("abfss://c@acct.privatelink.dfs.core.windows.net/t/").unwrap();
        assert!(azure_target(&private, &opts(&[("azure_storage_sas_key", "sig")])).is_ok());
    }

    #[test]
    fn another_account_than_the_options_name_is_refused() {
        let url = Url::parse("abfss://c@other.dfs.core.windows.net/t/").unwrap();
        let given = opts(&[
            ("azure_storage_sas_key", "sig"),
            ("azure_storage_account_name", "mine"),
        ]);
        let err = azure_target(&url, &given).unwrap_err();
        assert!(err.to_string().contains("another account"), "{err}");
        let same = Url::parse("abfss://c@mine.dfs.core.windows.net/t/").unwrap();
        assert!(azure_target(&same, &given).is_ok());
    }

    #[test]
    fn wasb_urls_are_refused_with_the_abfss_form() {
        let url = Url::parse("wasbs://cont@acct.blob.core.windows.net/t/").unwrap();
        let err = build_store(&url, &opts(&[("azure_storage_use_emulator", "true")])).unwrap_err();
        assert!(err.to_string().contains("abfss://"), "{err}");
    }

    #[test]
    fn the_probe_sees_put_if_absent_on_a_local_store() {
        let dir = std::env::temp_dir().join(format!("ds-probe-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let url = Url::from_directory_path(&dir).unwrap();
        assert!(probe_put_if_absent(&url, &HashMap::new()).unwrap());
        let log = dir.join("_delta_log");
        let left: Vec<_> = std::fs::read_dir(&log)
            .map(|d| d.count())
            .into_iter()
            .collect();
        assert_eq!(left, vec![0], "the sentinel was not removed");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn azure_with_explicit_endpoint_is_accepted() {
        let url = Url::parse("abfss://container@account.dfs.core.windows.net/t/").unwrap();
        let o = opts(&[
            ("azure_storage_sas_key", "sig=x"),
            ("azure_endpoint", "https://account.blob.core.windows.net"),
        ]);
        assert!(build_store(&url, &o).is_ok());
    }

    #[test]
    fn gcs_bearer_token_is_recognised_under_each_alias() {
        for key in GCS_BEARER_KEYS {
            let o = opts(&[(key, "ya29.token")]);
            assert_eq!(gcs_bearer_token(&o), Some("ya29.token"), "alias {key}");
        }
    }

    #[test]
    fn an_empty_alias_does_not_hide_a_token_under_another() {
        let o = opts(&[
            ("google_bearer_token", ""),
            ("gcp_oauth_token", "ya29.real"),
        ]);
        assert_eq!(gcs_bearer_token(&o), Some("ya29.real"));
        let o = opts(&[("GCP_OAUTH_TOKEN", "ya29.upper")]);
        assert_eq!(gcs_bearer_token(&o), Some("ya29.upper"));
    }

    #[test]
    fn azure_endpoint_aliases_and_the_emulator_are_accepted() {
        let url = Url::parse("abfss://container@account.dfs.core.windows.net/t/").unwrap();
        for (k, v) in [
            (
                "azure_storage_endpoint",
                "http://127.0.0.1:10000/devstoreaccount1",
            ),
            ("AZURE_ENDPOINT", "https://account.blob.core.windows.net"),
            ("azure_storage_use_emulator", "true"),
            ("use_emulator", "true"),
        ] {
            assert!(has_azure_endpoint(&opts(&[(k, v)])), "{k}");
        }
        assert!(!has_azure_endpoint(&opts(&[("use_emulator", "false")])));
        assert!(!has_azure_endpoint(&opts(&[("azure_endpoint", "")])));
        assert!(build_store(&url, &opts(&[("azure_storage_use_emulator", "true")])).is_ok());
    }

    #[test]
    fn empty_gcs_token_is_not_treated_as_a_credential() {
        let o = opts(&[("gcp_oauth_token", "")]);
        assert_eq!(gcs_bearer_token(&o), None);
    }

    #[test]
    fn gcs_with_bearer_token_builds_a_store() {
        // The delta-rs bug this avoids: routing the token through
        // google_application_credentials, which is read as a file path.
        let url = Url::parse("gs://bucket/table/").unwrap();
        let o = opts(&[("gcp_oauth_token", "ya29.token")]);
        assert!(build_store(&url, &o).is_ok());
    }

    /// The token must actually reach the wire as `Authorization: Bearer`,
    /// not just build a store. A one-shot local HTTP server stands in for GCS.
    #[test]
    fn gcs_bearer_token_is_sent_on_requests() {
        use delta_kernel::object_store::{path::Path, ObjectStoreExt};
        use std::io::{Read, Write};

        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let server = std::thread::spawn(move || {
            let (mut sock, _) = listener.accept().unwrap();
            let mut buf = vec![0u8; 8192];
            let n = sock.read(&mut buf).unwrap();
            let _ = sock.write_all(
                b"HTTP/1.1 404 Not Found\r\ncontent-length: 0\r\nconnection: close\r\n\r\n",
            );
            String::from_utf8_lossy(&buf[..n]).to_lowercase()
        });

        let url = Url::parse("gs://bucket/table/").unwrap();
        let base = format!("http://127.0.0.1:{port}");
        let o = opts(&[
            ("GOOGLE_BEARER_TOKEN", "ya29.secret"),
            ("google_base_url", base.as_str()),
            ("allow_http", "true"),
        ]);
        let store = build_store(&url, &o).unwrap();
        let _ = crate::runtime::block_on(async {
            store.head(&Path::from("table/_delta_log/x.json")).await
        });
        let request = server.join().unwrap();
        assert!(
            request.contains("authorization: bearer ya29.secret"),
            "request did not carry the bearer token:\n{request}"
        );
    }

    /// A server that answers `n` requests with 404 and returns each one's
    /// lower-cased text.
    fn recording_server(n: usize) -> (u16, std::thread::JoinHandle<Vec<String>>) {
        use std::io::{Read, Write};

        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let server = std::thread::spawn(move || {
            (0..n)
                .map(|_| {
                    let (mut sock, _) = listener.accept().unwrap();
                    let mut buf = vec![0u8; 8192];
                    let read = sock.read(&mut buf).unwrap();
                    let _ = sock.write_all(
                        b"HTTP/1.1 404 Not Found\r\ncontent-length: 0\r\nconnection: close\r\n\r\n",
                    );
                    String::from_utf8_lossy(&buf[..read]).to_lowercase()
                })
                .collect()
        });
        (port, server)
    }

    /// A server that answers with `statuses` in turn and returns each
    /// request's lower-cased text.
    fn scripted_server(
        statuses: &'static [&'static str],
    ) -> (u16, std::thread::JoinHandle<Vec<String>>) {
        use std::io::{Read, Write};

        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let server = std::thread::spawn(move || {
            statuses
                .iter()
                .map(|status| {
                    let (mut sock, _) = listener.accept().unwrap();
                    let mut buf = vec![0u8; 8192];
                    let read = sock.read(&mut buf).unwrap();
                    let reply = format!(
                        "HTTP/1.1 {status}\r\ncontent-length: 0\r\nlast-modified: \
                         Tue, 29 Sep 2026 10:00:00 GMT\r\netag: \"1\"\r\n\
                         connection: close\r\n\r\n"
                    );
                    let _ = sock.write_all(reply.as_bytes());
                    String::from_utf8_lossy(&buf[..read]).to_lowercase()
                })
                .collect()
        });
        (port, server)
    }

    fn far() -> Option<f64> {
        Some(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs_f64()
                + 3600.0,
        )
    }

    fn gcs_store(port: u16, slot: &str, token: &str) -> Arc<DynObjectStore> {
        let base = format!("http://127.0.0.1:{port}");
        let o = opts(&[
            ("google_bearer_token", token),
            ("google_base_url", base.as_str()),
            ("allow_http", "true"),
            ("max_retries", "0"),
            (SLOT_KEY, slot),
        ]);
        build_store(&Url::parse("gs://bucket/table/").unwrap(), &o).unwrap()
    }

    /// The Python refresher's part, for one slot: publish `token` when the
    /// slot's store asks. Requests for other slots (other tests running at
    /// once) are handed back, since each wait takes every pending request.
    fn answer(slot: &str, token: &'static str, force: bool) -> std::thread::JoinHandle<()> {
        let slot = slot.to_string();
        std::thread::spawn(move || {
            let deadline = std::time::Instant::now() + Duration::from_secs(20);
            while std::time::Instant::now() < deadline {
                let asked = crate::credential_slot::wait_requests(Duration::from_millis(200));
                let mine = asked.contains(&slot);
                for other in asked.into_iter().filter(|s| *s != slot) {
                    crate::credential_slot::request_refresh(&other);
                }
                if mine {
                    crate::credential_slot::publish(
                        &slot,
                        opts(&[("google_bearer_token", token)]),
                        far(),
                        force,
                    );
                    return;
                }
            }
        })
    }

    /// A request storage refuses asks the slot's publisher for a fresh
    /// credential and is sent again once, with it; the caller sees success.
    #[test]
    fn a_refused_request_is_retried_once_with_a_fresh_credential() {
        use delta_kernel::object_store::{path::Path, ObjectStoreExt};

        let slot = format!("test-{}", uuid::Uuid::new_v4());
        crate::credential_slot::set(&slot, opts(&[("google_bearer_token", "ya29.stale")]), far());
        let publisher = answer(&slot, "ya29.fresh", false);
        let (port, server) = scripted_server(&["403 Forbidden", "200 OK"]);
        let store = gcs_store(port, &slot, "ya29.stale");
        let head = crate::runtime::block_on(async {
            store.head(&Path::from("table/_delta_log/x.json")).await
        });
        publisher.join().unwrap();
        let requests = server.join().unwrap();
        crate::credential_slot::remove(&slot);
        assert!(head.is_ok(), "{head:?}");
        assert!(requests[0].contains("bearer ya29.stale"), "{}", requests[0]);
        assert!(requests[1].contains("bearer ya29.fresh"), "{}", requests[1]);
    }

    /// A second refusal is the store's own error: retried once, not forever.
    #[test]
    fn a_second_refusal_is_the_error() {
        use delta_kernel::object_store::{path::Path, ObjectStoreExt};

        let slot = format!("test-{}", uuid::Uuid::new_v4());
        crate::credential_slot::set(&slot, opts(&[("google_bearer_token", "ya29.a")]), far());
        // Vending gives the same credential: published as new anyway, so
        // the store retries at once rather than waiting out its timeout.
        let publisher = answer(&slot, "ya29.a", true);
        let (port, server) = scripted_server(&["403 Forbidden", "403 Forbidden"]);
        let store = gcs_store(port, &slot, "ya29.a");
        let started = std::time::Instant::now();
        let head = crate::runtime::block_on(async {
            store.head(&Path::from("table/_delta_log/x.json")).await
        });
        publisher.join().unwrap();
        assert_eq!(server.join().unwrap().len(), 2);
        crate::credential_slot::remove(&slot);
        assert!(head.is_err());
        assert!(
            started.elapsed() < Duration::from_secs(10),
            "{:?}",
            started.elapsed()
        );
    }

    #[test]
    fn refusals_are_told_from_other_errors() {
        use crate::auth_retry::refused;
        use delta_kernel::object_store::Error;

        let generic = |text: &str| Error::Generic {
            store: "S3",
            source: text.to_string().into(),
        };
        assert!(refused(&generic(
            "ExpiredToken: The provided token has expired"
        )));
        assert!(refused(&generic(
            "Server returned non-2xx status code: 403 Forbidden: AuthenticationFailed"
        )));
        assert!(!refused(&generic("connection reset by peer")));
        assert!(!refused(&Error::NotFound {
            path: "x".into(),
            source: "403 Forbidden".into(),
        }));
    }

    /// One store, two requests, a refresh published between them: the second
    /// request carries the new credential. Before slots, a store held the
    /// credential it was built with until it was dropped.
    #[test]
    fn a_store_with_a_slot_sends_the_refreshed_credential() {
        use delta_kernel::object_store::{path::Path, ObjectStoreExt};

        let far = || {
            Some(
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs_f64()
                    + 3600.0,
            )
        };
        let slot = format!("test-{}", uuid::Uuid::new_v4());
        crate::credential_slot::set(&slot, opts(&[("google_bearer_token", "ya29.first")]), far());
        let (port, server) = recording_server(2);
        let base = format!("http://127.0.0.1:{port}");
        let o = opts(&[
            ("google_bearer_token", "ya29.first"),
            ("google_base_url", base.as_str()),
            ("allow_http", "true"),
            ("max_retries", "0"),
            (SLOT_KEY, slot.as_str()),
        ]);
        let store = build_store(&Url::parse("gs://bucket/table/").unwrap(), &o).unwrap();
        let head = || {
            let _ = crate::runtime::block_on(async {
                store.head(&Path::from("table/_delta_log/x.json")).await
            });
        };
        head();
        crate::credential_slot::set(
            &slot,
            opts(&[("google_bearer_token", "ya29.second")]),
            far(),
        );
        head();
        let requests = server.join().unwrap();
        crate::credential_slot::remove(&slot);
        assert!(requests[0].contains("bearer ya29.first"), "{}", requests[0]);
        assert!(
            requests[1].contains("bearer ya29.second"),
            "{}",
            requests[1]
        );
    }

    #[test]
    fn s3_and_azure_stores_build_with_a_slot() {
        let slot = format!("test-{}", uuid::Uuid::new_v4());
        let s3 = opts(&[
            ("aws_access_key_id", "k"),
            ("aws_secret_access_key", "s"),
            ("aws_region", "us-west-2"),
            (SLOT_KEY, slot.as_str()),
        ]);
        build_store(&Url::parse("s3://b/t").unwrap(), &s3).unwrap();
        let azure = opts(&[
            ("azure_storage_sas_key", "sv=1&sig=x"),
            (SLOT_KEY, slot.as_str()),
        ]);
        build_store(
            &Url::parse("abfss://c@acct.dfs.core.windows.net/t").unwrap(),
            &azure,
        )
        .unwrap();
        // A slot on a local path is ignored, not an error.
        build_store(
            &Url::parse("file:///tmp/t").unwrap(),
            &opts(&[(SLOT_KEY, "x")]),
        )
        .unwrap();
    }

    #[test]
    fn gcs_without_a_token_falls_back_to_parse_url_opts() {
        let url = Url::parse("gs://bucket/table/").unwrap();
        // No credentials at all: object_store should still construct a store
        // (it resolves credentials lazily from the environment).
        assert!(build_store(&url, &HashMap::new()).is_ok());
    }
}
