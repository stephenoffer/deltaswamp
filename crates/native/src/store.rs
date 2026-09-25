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
//! DNS, and the sovereign clouds.

use std::collections::HashMap;
use std::sync::Arc;

use delta_kernel::object_store::client::StaticCredentialProvider;
use delta_kernel::object_store::gcp::{GcpCredential, GoogleCloudStorageBuilder};
use delta_kernel::object_store::{parse_url_opts, DynObjectStore};
use url::Url;

use crate::error::{NativeError, Result};

/// Option keys accepted for a GCS OAuth bearer token, most specific first.
///
/// `gcp_oauth_token` is the name Unity Catalog uses in its credential response,
/// so accepting it verbatim means callers can pass the vended block through
/// without renaming fields.
const GCS_BEARER_KEYS: &[&str] = &["google_bearer_token", "gcp_oauth_token", "bearer_token"];

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
    GCS_BEARER_KEYS
        .iter()
        .find_map(|k| options.get(*k))
        .map(String::as_str)
        .filter(|t| !t.is_empty())
}

/// Build an object store for `url`, honoring vended credentials.
pub fn build_store(url: &Url, options: &HashMap<String, String>) -> Result<Arc<DynObjectStore>> {
    if is_azure(url) && !options.contains_key("azure_endpoint") && !options.contains_key("endpoint")
    {
        // Refusing here rather than letting a confusing 403 surface later.
        return Err(NativeError::Invalid(format!(
            "Azure URL {url} has no explicit endpoint. Relying on account-name \
             inference breaks Azurite, private-link DNS and sovereign clouds \
             (.chinacloudapi.cn, .usgovcloudapi.net); pass azure_endpoint."
        )));
    }

    if is_gcs(url) {
        if let Some(token) = gcs_bearer_token(options) {
            return build_gcs_with_bearer(url, options, token);
        }
    }

    let pairs = options.iter().map(|(k, v)| (k.as_str(), v.as_str()));
    let (store, _path) = parse_url_opts(url, pairs)?;
    Ok(Arc::from(store))
}

/// Build a GCS store authenticated with a raw OAuth2 bearer token.
fn build_gcs_with_bearer(
    url: &Url,
    options: &HashMap<String, String>,
    token: &str,
) -> Result<Arc<DynObjectStore>> {
    let mut builder = GoogleCloudStorageBuilder::new().with_url(url.as_str());

    // Forward everything except our own token keys; object_store would reject
    // them as unknown configuration.
    for (k, v) in options {
        if GCS_INTERNAL_KEYS.contains(&k.as_str()) {
            continue;
        }
        if let Ok(key) = k.parse() {
            builder = builder.with_config(key, v);
        }
    }

    let provider = Arc::new(StaticCredentialProvider::new(GcpCredential {
        bearer: token.to_string(),
    }));
    let store = builder.with_credentials(provider).build()?;
    Ok(Arc::new(store))
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
    fn local_file_store_builds() {
        let url = Url::parse("file:///tmp/some-table/").unwrap();
        assert!(build_store(&url, &HashMap::new()).is_ok());
    }

    #[test]
    fn azure_without_explicit_endpoint_is_refused() {
        // The failure mode this prevents is a 403 several layers away, on
        // Azurite / private link / sovereign clouds.
        let url = Url::parse("abfss://container@account.dfs.core.windows.net/t/").unwrap();
        let err = build_store(&url, &opts(&[("azure_storage_sas_key", "sig")])).unwrap_err();
        assert!(err.to_string().contains("no explicit endpoint"), "{err}");
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

    #[test]
    fn gcs_without_a_token_falls_back_to_parse_url_opts() {
        let url = Url::parse("gs://bucket/table/").unwrap();
        // No credentials at all: object_store should still construct a store
        // (it resolves credentials lazily from the environment).
        assert!(build_store(&url, &HashMap::new()).is_ok());
    }
}
