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
//! * A write to a table whose generated columns, identity columns or
//!   invariants the caller computed and checked over every row (see
//!   [`Checked`]), all of which the kernel refuses to write.
//! * A write to a table carrying `checkpointProtection`, which kernel 0.28 does
//!   not know and so refuses (see [`HISTORY_ONLY`]).
//!
//! All start the transaction on a restated snapshot: the same log segment and
//! version, with the evolved metadata and protocol, and with the features the
//! caller evaluated set aside from the protocol the kernel checks the write
//! against. The metadata and protocol the
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

/// Which value constraints the caller evaluated over every row a commit writes.
///
/// Each stands in for the kernel's refusal of the matching feature:
///
/// * `constraints`: every CHECK constraint held (`checkConstraints`).
/// * `generated`: every generated column holds its generation expression's
///   value, computed where the data left it out (`generatedColumns`).
/// * `invariants`: every column invariant held (`invariants`).
/// * `identity`: every identity column's value was generated here, within a
///   range the table's high-water mark already covers, or given where the
///   column allows it (`identityColumns`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Checked {
    pub constraints: bool,
    pub generated: bool,
    pub invariants: bool,
    pub identity: bool,
}

impl Checked {
    /// `constraints_checked=` and `values_checked=` as the binding takes them.
    ///
    /// Unknown names are refused rather than ignored: a newer caller naming a
    /// check this build cannot set aside must not write past the kernel.
    pub fn from_args(constraints: bool, values: Option<Vec<String>>) -> Result<Self> {
        let mut checked = Checked {
            constraints,
            ..Default::default()
        };
        for name in values.unwrap_or_default() {
            match name.as_str() {
                "checkConstraints" => checked.constraints = true,
                "generatedColumns" => checked.generated = true,
                "invariants" => checked.invariants = true,
                "identityColumns" => checked.identity = true,
                other => {
                    return Err(NativeError::Invalid(format!(
                        "values_checked names {other:?}, which this build cannot set aside"
                    )))
                }
            }
        }
        Ok(checked)
    }

    /// Whether the caller checked anything at all.
    pub fn any(&self) -> bool {
        self.constraints || self.generated || self.invariants || self.identity
    }
}

impl From<bool> for Checked {
    fn from(constraints: bool) -> Self {
        Checked {
            constraints,
            ..Default::default()
        }
    }
}

/// What a write commits beside its data, and what it leaves the kernel to check.
#[derive(Debug, Clone, Default)]
pub struct Restatement {
    /// The `metaData` the commit writes, as log JSON, in place of the table's.
    pub metadata: Option<String>,
    /// The `protocol` the commit writes, as log JSON, in place of the table's.
    pub protocol: Option<String>,
    /// The value constraints the caller evaluated over every row the commit
    /// writes; the kernel's refusal of each is set aside (see
    /// [`checked_features`]).
    pub constraints_checked: Checked,
}

impl Restatement {
    /// Whether the write goes ahead on the table's own snapshot (unless the
    /// table carries a [`HISTORY_ONLY`] feature).
    pub fn is_empty(&self) -> bool {
        self.metadata.is_none() && self.protocol.is_none() && !self.constraints_checked.any()
    }
}

/// Features that bind only what a writer deletes from the log, never a commit.
///
/// `checkpointProtection` (Delta 4.0, added when a feature is dropped) says
/// that commits and checkpoints before
/// `delta.requireCheckpointProtectionBeforeVersion` may be removed only all
/// together, up to a checkpoint. A data, metadata or DML commit removes
/// nothing from the log, and a checkpoint of the latest version adds one;
/// kernel 0.28 has no variant for the feature, reads it as Unknown and
/// refuses every write. Expired log cleanup, the one operation the feature
/// binds, refuses it itself (`crate::logclean`).
pub(crate) const HISTORY_ONLY: &[&str] = &["checkpointProtection"];

/// Whether `protocol` lists any of `features` among its writer features.
fn lists_any(protocol: &Protocol, features: &[&str]) -> bool {
    protocol.writer_features().is_some_and(|listed| {
        listed
            .iter()
            .any(|f| features.contains(&f.to_string().as_str()))
    })
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
        // A legacy feature is supported only where both versions reach it:
        // columnMapping, the one reader-writer feature here, needs reader 2,
        // and delta-rs writes (1, 6) tables that have none of it.
        None => legacy_writer_features(writer)
            .into_iter()
            .filter(|f| *f != "columnMapping" || protocol.min_reader_version() >= 2)
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
    // Every restated snapshot sets these aside: no commit is bound by them.
    let set_aside: Vec<&str> = set_aside.iter().chain(HISTORY_ONLY).copied().collect();
    let checked = without_features(&protocol, &set_aside)?;
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

/// Features that constrain only the values a commit writes.
///
/// A compaction writes back exactly the values it read (`dataChange=false`),
/// already checked when they were first written: a CHECK constraint, an
/// invariant or a generation expression holds for them as it did, and an
/// identity column's values and high-water mark are untouched. A checkpoint
/// or a checksum writes no row at all. delta-kernel refuses any of these
/// on a table carrying one of the features -- a writer version 3 or 4 table
/// implies two of them, so every table delta-rs gave a change data feed was
/// refused -- which left those tables to delta-rs, or, where delta-rs lacks
/// another of the table's features (in-commit timestamps, row tracking,
/// clustering), to nothing.
pub(crate) const VALUE_CONSTRAINTS: &[&str] = &[
    "checkConstraints",
    "generatedColumns",
    "identityColumns",
    "invariants",
];

/// `snapshot` as a checkpoint or checksum writer sees it: the same log
/// segment, metadata and version, with [`VALUE_CONSTRAINTS`] left out of the
/// protocol the kernel checks the table against -- and `snapshot` itself
/// where the kernel writes the table as it is.
///
/// Both writers take the protocol and metadata they write from the log
/// (`LogSegment::read_actions` for a checkpoint, the log replay rooted at a
/// checkpoint or version 0 for a checksum), never from the snapshot's table
/// configuration, so what they write is the table's own protocol, the
/// features set aside here included. The restated configuration decides only
/// the kernel's refusal; the checkpoint's shape (classic or v2, stats as JSON
/// or struct) comes from features and properties it keeps.
///
/// A restated snapshot holds no checksum in memory (the kernel builds it from
/// the log segment alone), so the kernel resolves a checksum from it by
/// replay: see `crate::checksum::write`.
pub(crate) fn log_writing_snapshot(snapshot: &SnapshotRef) -> Result<SnapshotRef> {
    use delta_kernel::table_features::Operation;

    if snapshot
        .table_configuration()
        .ensure_operation_supported(Operation::Write)
        .is_ok()
    {
        return Ok(snapshot.clone());
    }
    // A checkpoint below the protected version would be one the feature's
    // all-or-nothing cleanup rule does not account for. That version is the
    // one a feature was dropped at, so the latest is never below it; refused
    // rather than assumed.
    let config = snapshot.table_configuration();
    if lists_any(config.protocol(), HISTORY_ONLY) {
        let protected = config
            .metadata()
            .configuration()
            .get("delta.requireCheckpointProtectionBeforeVersion")
            .and_then(|v| v.parse::<u64>().ok());
        if let Some(below) = protected.filter(|below| snapshot.version() < *below) {
            return Err(NativeError::Invalid(format!(
                "the table's checkpointProtection covers versions below {below}, and this \
                 snapshot is version {}",
                snapshot.version()
            )));
        }
    }
    restated_snapshot(snapshot, VALUE_CONSTRAINTS, None, None)
}

/// The features a caller's own evaluation of CHECK constraints stands in for.
#[cfg(test)]
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

/// What a checked write sets aside: each feature `checked` says the caller
/// evaluated, and `generatedColumns` / `identityColumns` when no column of
/// `schema_string` is generated / an identity column, as there is then no
/// value for anyone to compute.
///
/// Legacy writer versions 4 to 6 imply `generatedColumns` (and 6
/// `identityColumns`) whatever the schema, and the kernel refuses both: every
/// column-mapped table Databricks or delta-rs created at (2, 5) was refused,
/// though no column of it had a generation expression for the kernel to compute.
fn checked_features(schema_string: &str, checked: Checked) -> Vec<&'static str> {
    let mut features = Vec::new();
    if checked.constraints {
        features.push("checkConstraints");
    }
    if checked.generated || !declares(schema_string, "delta.generationExpression") {
        features.push("generatedColumns");
    }
    if checked.identity || !declares(schema_string, "delta.identity.start") {
        features.push("identityColumns");
    }
    if checked.invariants {
        features.push("invariants");
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

    if restatement.is_empty() && !lists_any(snapshot.table_configuration().protocol(), HISTORY_ONLY)
    {
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
    let set_aside = if restatement.constraints_checked.any() {
        let schema = match &metadata {
            Some(new) => new.schema_string().clone(),
            None => snapshot
                .table_configuration()
                .metadata()
                .schema_string()
                .clone(),
        };
        checked_features(&schema, restatement.constraints_checked)
    } else {
        // HISTORY_ONLY alone, which restated_snapshot always adds.
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

        // Writer 6 under reader 1 (as delta-rs writes it) has no column
        // mapping, which needs reader 2: the restatement does not claim it.
        let reader_one =
            protocol(serde_json::json!({"minReaderVersion": 1, "minWriterVersion": 6}));
        let restated = without_features(&reader_one, &["identityColumns"])
            .unwrap()
            .unwrap();
        let features: Vec<String> = restated
            .writer_features()
            .unwrap()
            .iter()
            .map(|f| f.to_string())
            .collect();
        assert!(
            !features.contains(&"columnMapping".to_string()),
            "{features:?}"
        );
    }

    /// A table carrying every value-constraint feature: CHECK constraints, a
    /// generated column, an identity column and an invariant, over two commits.
    fn value_constrained_table() -> (crate::commit::SharedEngine, SnapshotRef, serde_json::Value) {
        use delta_kernel::object_store::memory::InMemory;
        use delta_kernel::object_store::path::Path;
        use delta_kernel::object_store::ObjectStoreExt;
        use delta_kernel::snapshot::Snapshot;

        let protocol = serde_json::json!({
            "minReaderVersion": 1,
            "minWriterVersion": 7,
            "writerFeatures": [
                "appendOnly", "invariants", "checkConstraints", "generatedColumns",
                "identityColumns"
            ],
        });
        let schema = serde_json::json!({"type": "struct", "fields": [
            {"name": "id", "type": "long", "nullable": true,
             "metadata": {"delta.identity.start": 1, "delta.identity.step": 1,
                          "delta.identity.allowExplicitInsert": false,
                          "delta.identity.highWaterMark": 4}},
            {"name": "g", "type": "long", "nullable": true,
             "metadata": {"delta.generationExpression": "id * 2"}},
            {"name": "v", "type": "long", "nullable": true,
             "metadata": {"delta.invariants": "{\"expression\":{\"expression\":\"v > 0\"}}"}},
        ]});
        let metadata = serde_json::json!({
            "id": "5b3c0a1e-0000-4000-8000-000000000001",
            "format": {"provider": "parquet", "options": {}},
            "schemaString": schema.to_string(),
            "partitionColumns": [],
            "configuration": {"delta.constraints.pos": "id > 0"},
            "createdTime": 1,
        });
        let add = |name: &str, size: i64| {
            serde_json::json!({"add": {"path": name, "partitionValues": {}, "size": size,
                "modificationTime": 1, "dataChange": true}})
            .to_string()
        };
        let info = |op: &str| {
            serde_json::json!({"commitInfo": {"timestamp": 1, "operation": op}}).to_string()
        };
        let commits = [
            [
                info("CREATE TABLE"),
                serde_json::json!({"protocol": protocol}).to_string(),
                serde_json::json!({"metaData": metadata}).to_string(),
                add("a.parquet", 100),
            ]
            .join("\n"),
            [info("WRITE"), add("b.parquet", 200)].join("\n"),
        ];
        let store = Arc::new(InMemory::new());
        for (version, body) in commits.iter().enumerate() {
            let key = Path::from(format!("t/_delta_log/{version:020}.json"));
            crate::runtime::block_on(store.put(&key, body.clone().into())).unwrap();
        }
        let engine = crate::commit::new_engine(store);
        let snapshot = Snapshot::builder_for("memory:///t/")
            .build(engine.as_ref())
            .unwrap();
        (engine, snapshot, protocol)
    }

    #[test]
    fn a_checkpoint_of_a_value_constrained_table_keeps_its_protocol() {
        use delta_kernel::snapshot::{CheckpointWriteResult, Snapshot};

        let (engine, snapshot, protocol) = value_constrained_table();
        // The kernel refuses the table as it is ...
        assert!(snapshot.checkpoint(engine.as_ref(), None).is_err());
        // ... and checkpoints it restated.
        let restated = log_writing_snapshot(&snapshot).unwrap();
        assert!(!Arc::ptr_eq(&restated, &snapshot));
        let (result, checkpointed) = restated.checkpoint(engine.as_ref(), None).unwrap();
        assert!(matches!(result, CheckpointWriteResult::Written));
        assert!(crate::checksum::write_at_checkpoint(
            &checkpointed,
            engine.as_ref()
        ));

        // A fresh snapshot, loaded from the checkpoint, has the table's own
        // protocol and metadata.
        let fresh = Snapshot::builder_for("memory:///t/")
            .build(engine.as_ref())
            .unwrap();
        assert_eq!(fresh.log_segment().checkpoint_version, Some(1));
        assert_eq!(
            serde_json::to_value(fresh.table_configuration().protocol()).unwrap(),
            serde_json::to_value(snapshot.table_configuration().protocol()).unwrap(),
        );
        assert_eq!(
            serde_json::to_value(fresh.table_configuration().protocol()).unwrap()["writerFeatures"],
            protocol["writerFeatures"],
        );
        assert_eq!(
            fresh.table_configuration().metadata(),
            snapshot.table_configuration().metadata()
        );
        assert_eq!(
            fresh
                .get_file_stats_if_present()
                .map(|s| (s.num_files(), s.table_size_bytes())),
            Some((2, 300))
        );
    }

    #[test]
    fn a_writable_table_is_checkpointed_as_it_is() {
        use delta_kernel::object_store::memory::InMemory;
        use delta_kernel::object_store::path::Path;
        use delta_kernel::object_store::ObjectStoreExt;
        use delta_kernel::snapshot::Snapshot;

        let body = [
            r#"{"commitInfo":{"timestamp":1,"operation":"CREATE TABLE"}}"#,
            r#"{"protocol":{"minReaderVersion":1,"minWriterVersion":2}}"#,
            r#"{"metaData":{"id":"5b3c0a1e-0000-4000-8000-000000000002","format":{"provider":"parquet","options":{}},"schemaString":"{\"type\":\"struct\",\"fields\":[{\"name\":\"id\",\"type\":\"long\",\"nullable\":true,\"metadata\":{}}]}","partitionColumns":[],"configuration":{},"createdTime":1}}"#,
        ]
        .join("\n");
        let store = Arc::new(InMemory::new());
        let key = Path::from(format!("t/_delta_log/{:020}.json", 0));
        crate::runtime::block_on(store.put(&key, body.into())).unwrap();
        let engine = crate::commit::new_engine(store);
        let snapshot = Snapshot::builder_for("memory:///t/")
            .build(engine.as_ref())
            .unwrap();
        // Writer version 2 implies invariants, but the schema declares none:
        // the kernel writes the table, and the snapshot (checksum and all) is kept.
        assert!(Arc::ptr_eq(
            &log_writing_snapshot(&snapshot).unwrap(),
            &snapshot
        ));
    }

    #[test]
    fn generated_columns_are_set_aside_only_where_none_is_declared() {
        let plain = r#"{"type":"struct","fields":[{"name":"id","type":"long","nullable":true,"metadata":{}}]}"#;
        let constraints = Checked::from(true);
        assert_eq!(
            checked_features(plain, constraints),
            ["checkConstraints", "generatedColumns", "identityColumns"]
        );
        let nested = r#"{"type":"struct","fields":[{"name":"s","type":{"type":"struct","fields":[{"name":"g","type":"long","nullable":true,"metadata":{"delta.generationExpression":"1"}}]},"nullable":true,"metadata":{}}]}"#;
        assert_eq!(
            checked_features(nested, constraints),
            ["checkConstraints", "identityColumns"]
        );
        assert_eq!(
            checked_features("not json", constraints),
            ["checkConstraints"]
        );
        // A declared generated column is set aside only when the caller computed it.
        let computed = Checked {
            generated: true,
            invariants: true,
            ..Default::default()
        };
        assert_eq!(
            checked_features(nested, computed),
            ["generatedColumns", "identityColumns", "invariants"]
        );
        assert!(Checked::from_args(false, Some(vec!["nope".into()])).is_err());
    }
}
