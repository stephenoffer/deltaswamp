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

/// Build an object store for `url`, honoring vended credentials.
pub fn build_store(url: &Url, options: &HashMap<String, String>) -> Result<Arc<DynObjectStore>> {
    if is_azure(url) && !has_azure_endpoint(options) {
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
        let k = k.to_ascii_lowercase();
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

    #[test]
    fn gcs_without_a_token_falls_back_to_parse_url_opts() {
        let url = Url::parse("gs://bucket/table/").unwrap();
        // No credentials at all: object_store should still construct a store
        // (it resolves credentials lazily from the environment).
        assert!(build_store(&url, &HashMap::new()).is_ok());
    }
}
