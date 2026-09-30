//! A store that asks for a fresh credential, once, when storage refuses one.
//!
//! A credential slot is refreshed ahead of the vended credential's stated
//! expiry, but storage can refuse a credential before then: one revoked when
//! a grant changed, a token whose clock disagrees with the store's, an Azure
//! SAS cut short. Each of those failed the request, and with it a scan or a
//! write partway through. Wrapped here, a request refused with 401/403 (or
//! an expired-token error) asks the slot's publisher for a fresh credential
//! (`credential_slot::request_refresh`, answered by the Python refresher
//! thread), waits for it to be published, and is sent again once. A second
//! refusal is the store's own error: the principal really may not.
//!
//! What is retried is a single request: a put (nothing was written, so it is
//! safe to send again), a get, a head, a copy, a listing's first page, the
//! start of a multipart upload. A body already streaming, or a part of an
//! upload under way, fails as it did: storage checks the credential when a
//! request starts.

use std::fmt;
use std::ops::Range;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use bytes::Bytes;
use delta_kernel::object_store::path::Path;
use delta_kernel::object_store::{
    CopyOptions, Error, GetOptions, GetResult, ListResult, MultipartUpload, ObjectMeta,
    ObjectStore, PutMultipartOptions, PutOptions, PutPayload, PutResult, RenameOptions, Result,
};
use futures::stream::{BoxStream, StreamExt};

use crate::credential_slot;

/// How long a refused request waits for the publisher's fresh credential.
const REFRESH_WAIT: Duration = Duration::from_secs(30);

/// Words storage uses for a credential it no longer takes, where
/// object_store reports the refusal as a generic error rather than
/// `Unauthenticated` / `PermissionDenied`.
const REFUSALS: &[&str] = &[
    "expiredtoken",
    "token has expired",
    "token is expired",
    "request has expired",
    "authenticationfailed",
    "invalidauthenticationinfo",
    "invalid_token",
    "unauthorized",
    "status: 401",
    "status: 403",
    "401 unauthorized",
    "403 forbidden",
];

/// Whether `err` is storage refusing the credential.
pub(crate) fn refused(err: &Error) -> bool {
    match err {
        Error::Unauthenticated { .. } | Error::PermissionDenied { .. } => true,
        Error::NotFound { .. }
        | Error::AlreadyExists { .. }
        | Error::Precondition { .. }
        | Error::NotModified { .. }
        | Error::NotImplemented { .. }
        | Error::NotSupported { .. } => false,
        other => {
            let text = other.to_string().to_ascii_lowercase();
            REFUSALS.iter().any(|word| text.contains(word))
        }
    }
}

/// `inner`, retried once with a fresh credential from slot `slot`.
pub struct AuthRetryStore {
    inner: Arc<dyn ObjectStore>,
    slot: String,
}

impl AuthRetryStore {
    pub fn new(inner: Arc<dyn ObjectStore>, slot: &str) -> Self {
        Self {
            inner,
            slot: slot.to_string(),
        }
    }

    /// Ask for a fresh credential; whether one was published in time.
    async fn refreshed(&self) -> bool {
        refresh(&self.slot).await
    }
}

async fn refresh(slot: &str) -> bool {
    let Some(since) = credential_slot::generation(slot) else {
        return false;
    };
    credential_slot::request_refresh(slot);
    credential_slot::wait_newer(slot, since, REFRESH_WAIT).await
}

impl fmt::Debug for AuthRetryStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AuthRetryStore")
            .field("inner", &self.inner)
            .finish()
    }
}

impl fmt::Display for AuthRetryStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "AuthRetry({})", self.inner)
    }
}

/// `$call`, sent again once when storage refused the credential and a fresh
/// one arrived.
macro_rules! retried {
    ($self:ident, $call:expr) => {
        match $call.await {
            Err(err) if refused(&err) && $self.refreshed().await => $call.await,
            other => other,
        }
    };
}

#[async_trait]
impl ObjectStore for AuthRetryStore {
    async fn put_opts(
        &self,
        location: &Path,
        payload: PutPayload,
        opts: PutOptions,
    ) -> Result<PutResult> {
        retried!(
            self,
            self.inner.put_opts(location, payload.clone(), opts.clone())
        )
    }

    async fn put_multipart_opts(
        &self,
        location: &Path,
        opts: PutMultipartOptions,
    ) -> Result<Box<dyn MultipartUpload>> {
        retried!(self, self.inner.put_multipart_opts(location, opts.clone()))
    }

    async fn get_opts(&self, location: &Path, options: GetOptions) -> Result<GetResult> {
        retried!(self, self.inner.get_opts(location, options.clone()))
    }

    async fn get_ranges(&self, location: &Path, ranges: &[Range<u64>]) -> Result<Vec<Bytes>> {
        retried!(self, self.inner.get_ranges(location, ranges))
    }

    fn delete_stream(
        &self,
        locations: BoxStream<'static, Result<Path>>,
    ) -> BoxStream<'static, Result<Path>> {
        self.inner.delete_stream(locations)
    }

    fn list(&self, prefix: Option<&Path>) -> BoxStream<'static, Result<ObjectMeta>> {
        let inner = self.inner.clone();
        let slot = self.slot.clone();
        let prefix = prefix.cloned();
        // A refused listing fails on its first page; one after it has begun
        // yielding is not restarted, which would repeat what it yielded.
        futures::stream::once(async move {
            let mut listing = inner.list(prefix.as_ref());
            match listing.next().await {
                Some(Err(err)) if refused(&err) && refresh(&slot).await => {
                    inner.list(prefix.as_ref())
                }
                first => futures::stream::iter(first).chain(listing).boxed(),
            }
        })
        .flatten()
        .boxed()
    }

    fn list_with_offset(
        &self,
        prefix: Option<&Path>,
        offset: &Path,
    ) -> BoxStream<'static, Result<ObjectMeta>> {
        let inner = self.inner.clone();
        let slot = self.slot.clone();
        let (prefix, offset) = (prefix.cloned(), offset.clone());
        futures::stream::once(async move {
            let mut listing = inner.list_with_offset(prefix.as_ref(), &offset);
            match listing.next().await {
                Some(Err(err)) if refused(&err) && refresh(&slot).await => {
                    inner.list_with_offset(prefix.as_ref(), &offset)
                }
                first => futures::stream::iter(first).chain(listing).boxed(),
            }
        })
        .flatten()
        .boxed()
    }

    async fn list_with_delimiter(&self, prefix: Option<&Path>) -> Result<ListResult> {
        retried!(self, self.inner.list_with_delimiter(prefix))
    }

    async fn copy_opts(&self, from: &Path, to: &Path, options: CopyOptions) -> Result<()> {
        retried!(self, self.inner.copy_opts(from, to, options.clone()))
    }

    async fn rename_opts(&self, from: &Path, to: &Path, options: RenameOptions) -> Result<()> {
        retried!(self, self.inner.rename_opts(from, to, options.clone()))
    }
}
