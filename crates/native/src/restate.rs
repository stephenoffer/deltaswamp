//! A write's transaction on the table as the write leaves it.
//!
//! delta-kernel 0.28 checks every data write against the snapshot it starts
//! from, and has no way to change that snapshot's schema in the same commit as
//! the data: its ALTER TABLE transaction takes no data files, and its write
//! transaction emits no metadata. Two writes need exactly that:
//!
//! * A schema-evolving append or overwrite (`schema_mode="merge"`) writes its
//!   rows under the widened schema and commits the new `metaData` with them,
//!   one atomic commit -- a reader never sees the new column without its data,
//!   nor the data under the old schema.
//! * A write to a table with CHECK constraints, which the kernel refuses
//!   outright (`checkConstraints` is NotSupported for writes, whether or not
//!   the table holds a constraint). The caller evaluates every constraint over
//!   every row it writes, as Spark does, and says so.
//!
//! Both start the transaction on a restated snapshot: the same log segment and
//! version, with the evolved metadata and protocol, and with
//! `checkConstraints` set aside from the protocol the kernel checks the write
//! against when the caller has checked them. The metadata and protocol the
//! commit really writes go in as actions of their own, after the kernel's;
//! the protocol set aside is never written anywhere. The commit's conflict
//! check is the object store's put-if-absent of the next version, as for any
//! kernel write, and the post-commit snapshot kernel builds from the restated
//! one is not used for the version's checksum (see `commit::finish_commit_as`).

use std::sync::Arc;

use delta_kernel::actions::{Metadata, Protocol};
use delta_kernel::snapshot::SnapshotRef;
use delta_kernel::FilteredEngineData;

use crate::commit::{CommitInfoPatch, SharedEngine};
use crate::error::{NativeError, Result};

/// What a write commits beside its data, and what it leaves the kernel to check.
#[derive(Debug, Clone, Default)]
pub struct Restatement {
    /// The `metaData` the commit writes, as log JSON, in place of the table's.
    pub metadata: Option<String>,
    /// The `protocol` the commit writes, as log JSON, in place of the table's.
    pub protocol: Option<String>,
    /// Every CHECK constraint was evaluated over every row the commit writes.
    /// The kernel's refusal of `checkConstraints` is then set aside, and of
    /// `generatedColumns` where no column is generated (see [`checked_features`]).
    pub constraints_checked: bool,
}

impl Restatement {
    /// Whether the write goes ahead on the table's own snapshot.
    pub fn is_empty(&self) -> bool {
        self.metadata.is_none() && self.protocol.is_none() && !self.constraints_checked
    }
}

/// The writer features a legacy writer version implies, in protocol order.
pub(crate) fn legacy_writer_features(version: i32) -> Vec<&'static str> {
    let mut features = Vec::new();
    if version >= 2 {
        features.extend(["appendOnly", "invariants"]);
    }
    if version >= 3 {
        features.push("checkConstraints");
    }
    if version >= 4 {
        features.extend(["changeDataFeed", "generatedColumns"]);
    }
    if version >= 5 {
        features.push("columnMapping");
    }
    if version >= 6 {
        features.push("identityColumns");
    }
    features
}

/// `protocol` with `set_aside` left out of its writer features, or None when
/// it lists (or implies) none of them.
///
/// A legacy writer version becomes the same features listed explicitly
/// (writer version 7), the only form in which some can be left out.
fn without_features(protocol: &Protocol, set_aside: &[&str]) -> Result<Option<Protocol>> {
    let writer = protocol.min_writer_version();
    let listed: Vec<String> = match protocol.writer_features() {
        Some(features) => features.iter().map(|f| f.to_string()).collect(),
        None => legacy_writer_features(writer)
            .into_iter()
            .map(str::to_string)
            .collect(),
    };
    if !listed.iter().any(|f| set_aside.contains(&f.as_str())) {
        return Ok(None);
    }
    let kept: Vec<&String> = listed
        .iter()
        .filter(|f| !set_aside.contains(&f.as_str()))
        .collect();
    let reader_features: Option<Vec<String>> = protocol
        .reader_features()
        .map(|features| features.iter().map(|f| f.to_string()).collect());
    let restated = serde_json::from_value(serde_json::json!({
        "minReaderVersion": protocol.min_reader_version(),
        "minWriterVersion": 7,
        "readerFeatures": reader_features,
        "writerFeatures": kept,
    }))
    .map_err(|e| {
        NativeError::Invalid(format!(
            "could not restate the table's protocol for the write: {e}"
        ))
    })?;
    Ok(Some(restated))
}

/// `snapshot` with `metadata` and `protocol` in place of its own (each when
/// given), and `set_aside` left out of the protocol the transaction checks.
///
/// The snapshot itself when nothing changes.
pub(crate) fn restated_snapshot(
    snapshot: &SnapshotRef,
    set_aside: &[&str],
    metadata: Option<Metadata>,
    protocol: Option<Protocol>,
) -> Result<SnapshotRef> {
    use delta_kernel::snapshot::Snapshot;
    use delta_kernel::table_configuration::TableConfiguration;

    let config = snapshot.table_configuration();
    let protocol = protocol.unwrap_or_else(|| config.protocol().clone());
    let checked = without_features(&protocol, set_aside)?;
    if metadata.is_none() && checked.is_none() && &protocol == config.protocol() {
        return Ok(snapshot.clone());
    }
    let config = TableConfiguration::try_new(
        metadata.unwrap_or_else(|| config.metadata().clone()),
        checked.unwrap_or(protocol),
        snapshot.table_root().clone(),
        snapshot.version(),
    )?;
    Ok(Arc::new(Snapshot::new(
        snapshot.log_segment().clone(),
        config,
    )?))
}

/// The features a caller's own evaluation stands in for.
const CHECKED: &[&str] = &["checkConstraints"];

/// Whether any field of the schema in `schema_string` (nested ones included)
/// carries the metadata key `key`.
fn declares(schema_string: &str, key: &str) -> bool {
    fn walk(value: &serde_json::Value, key: &str) -> bool {
        match value {
            serde_json::Value::Object(map) => {
                let here = map
                    .get("metadata")
                    .and_then(|m| m.as_object())
                    .is_some_and(|m| m.contains_key(key));
                here || map.values().any(|v| walk(v, key))
            }
            serde_json::Value::Array(items) => items.iter().any(|v| walk(v, key)),
            _ => false,
        }
    }
    // Unreadable: taken as declaring it, so the kernel's own check stands.
    serde_json::from_str::<serde_json::Value>(schema_string)
        .map(|schema| walk(&schema, key))
        .unwrap_or(true)
}

/// What a checked write sets aside: [`CHECKED`], and `generatedColumns` when
/// no column of `schema_string` is generated.
///
/// Legacy writer versions 4 to 6 imply `generatedColumns` whatever the
/// schema, and the kernel refuses the feature: every column-mapped table
/// Databricks or delta-rs created at (2, 5) was refused, though no column of
/// it had a generation expression for the kernel to compute.
fn checked_features(schema_string: &str) -> Vec<&'static str> {
    let mut features = CHECKED.to_vec();
    if !declares(schema_string, "delta.generationExpression") {
        features.push("generatedColumns");
    }
    features
}

/// The snapshot `restatement`'s write starts its transaction on, with the
/// metadata and protocol it changes queued into `info` as actions of the
/// commit.
pub(crate) fn writing_snapshot(
    snapshot: &SnapshotRef,
    engine: &SharedEngine,
    restatement: &Restatement,
    info: &CommitInfoPatch,
) -> Result<SnapshotRef> {
    use delta_kernel::actions::{LOG_METADATA_SCHEMA, LOG_PROTOCOL_SCHEMA};
    use delta_kernel::IntoEngineData;

    if restatement.is_empty() {
        return Ok(snapshot.clone());
    }
    let parse = |what: &str, text: &str| -> Result<serde_json::Value> {
        serde_json::from_str(text)
            .map_err(|e| NativeError::Invalid(format!("the new {what} is not valid JSON: {e}")))
    };
    let protocol: Option<Protocol> = match &restatement.protocol {
        Some(text) => Some(
            serde_json::from_value(parse("protocol", text)?).map_err(|e| {
                NativeError::Invalid(format!("the new protocol is not a protocol action: {e}"))
            })?,
        ),
        None => None,
    };
    let metadata: Option<Metadata> = match &restatement.metadata {
        Some(text) => Some(
            serde_json::from_value(parse("metaData", text)?).map_err(|e| {
                NativeError::Invalid(format!("the new metaData is not a metaData action: {e}"))
            })?,
        ),
        None => None,
    };
    if let Some(new) = &metadata {
        let current = snapshot.table_configuration().metadata();
        if new.id() != current.id() {
            // A metaData with another id is another table: every reader
            // would take the log for a replaced one.
            return Err(NativeError::Invalid(
                "the new metaData names another table id than the table's".to_string(),
            ));
        }
    }
    let set_aside = if restatement.constraints_checked {
        let schema = match &metadata {
            Some(new) => new.schema_string().clone(),
            None => snapshot
                .table_configuration()
                .metadata()
                .schema_string()
                .clone(),
        };
        checked_features(&schema)
    } else {
        Vec::new()
    };
    let restated = restated_snapshot(snapshot, &set_aside, metadata.clone(), protocol.clone())?;
    // Queued only once the restated snapshot validated them: the kernel's
    // TableConfiguration refuses a schema its column mapping does not cover,
    // a protocol missing a feature the metadata enables, and the like.
    if let Some(protocol) = protocol {
        let data = protocol.into_engine_data(LOG_PROTOCOL_SCHEMA.clone(), engine.as_ref())?;
        info.extra_actions
            .push(FilteredEngineData::with_all_rows_selected(data));
    }
    if let Some(metadata) = metadata {
        let data = metadata.into_engine_data(LOG_METADATA_SCHEMA.clone(), engine.as_ref())?;
        info.extra_actions
            .push(FilteredEngineData::with_all_rows_selected(data));
    }
    Ok(restated)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn legacy_writer_versions_imply_their_features() {
        assert_eq!(legacy_writer_features(1), Vec::<&str>::new());
        assert_eq!(
            legacy_writer_features(4),
            [
                "appendOnly",
                "invariants",
                "checkConstraints",
                "changeDataFeed",
                "generatedColumns"
            ]
        );
        assert!(legacy_writer_features(6).contains(&"identityColumns"));
    }

    fn protocol(value: serde_json::Value) -> Protocol {
        serde_json::from_value(value).unwrap()
    }

    #[test]
    fn set_aside_features_leave_the_protocol() {
        let listed = protocol(serde_json::json!({
            "minReaderVersion": 1,
            "minWriterVersion": 7,
            "writerFeatures": ["checkConstraints", "inCommitTimestamp"],
        }));
        let restated = without_features(&listed, CHECKED).unwrap().unwrap();
        let features: Vec<String> = restated
            .writer_features()
            .unwrap()
            .iter()
            .map(|f| f.to_string())
            .collect();
        assert_eq!(features, ["inCommitTimestamp"]);

        // A legacy version is restated as the features it implies, less the
        // ones set aside.
        let legacy = protocol(serde_json::json!({"minReaderVersion": 1, "minWriterVersion": 3}));
        let restated = without_features(&legacy, CHECKED).unwrap().unwrap();
        assert_eq!(restated.min_writer_version(), 7);
        let features: Vec<String> = restated
            .writer_features()
            .unwrap()
            .iter()
            .map(|f| f.to_string())
            .collect();
        assert_eq!(features, ["appendOnly", "invariants"]);

        let plain = protocol(serde_json::json!({"minReaderVersion": 1, "minWriterVersion": 2}));
        assert!(without_features(&plain, CHECKED).unwrap().is_none());
    }

    #[test]
    fn generated_columns_are_set_aside_only_where_none_is_declared() {
        let plain = r#"{"type":"struct","fields":[{"name":"id","type":"long","nullable":true,"metadata":{}}]}"#;
        assert_eq!(
            checked_features(plain),
            ["checkConstraints", "generatedColumns"]
        );
        let nested = r#"{"type":"struct","fields":[{"name":"s","type":{"type":"struct","fields":[{"name":"g","type":"long","nullable":true,"metadata":{"delta.generationExpression":"1"}}]},"nullable":true,"metadata":{}}]}"#;
        assert_eq!(checked_features(nested), ["checkConstraints"]);
        assert_eq!(checked_features("not json"), ["checkConstraints"]);
    }
}
