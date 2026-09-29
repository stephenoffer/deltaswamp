//! Geospatial columns as the kernel's default engine can read and write them.
//!
//! A `geospatial` table (Databricks' GEOMETRY and GEOGRAPHY types, protocol
//! RFC delta-io/delta#4725) stores each value as WKB in a Parquet BYTE_ARRAY
//! annotated with the GEOMETRY or GEOGRAPHY logical type. delta-kernel 0.28
//! parses the types (`geo-type-in-dev`) but its default engine converts no geo
//! type to Arrow, so every scan failed, and it refuses every write.
//!
//! Here the kernel sees each geo column as `binary` -- which is what the
//! Parquet column physically is, and what the engine reads it as -- with its
//! real type kept in the field's metadata under [`TYPE_KEY`]. The protocol is
//! the table's own, `geospatial` included: the kernel accepts it on a table
//! whose schema holds no geo type, and the table's features read as they are.
//! A write sets the feature aside (`crate::restate`) and annotates each geo
//! column's Parquet type with its logical type ([`parquet_schema`]); it
//! writes no min/max statistics for them (binary columns get none), which the
//! RFC allows ("Writers, if writing statistics, ..."), so readers skip no file
//! by a geo column. The table's own `metaData` is never replaced by this
//! view: `PySnapshot::metadata_json` returns what the log holds.

use std::sync::Arc;

use delta_kernel::actions::Metadata;
use delta_kernel::parquet::basic::{EdgeInterpolationAlgorithm, LogicalType, Type as PhysicalType};
use delta_kernel::parquet::schema::types::{SchemaDescriptor, Type, TypePtr};
use delta_kernel::schema::{DataType, StructType};
use delta_kernel::snapshot::{Snapshot, SnapshotRef};

use crate::error::{NativeError, Result};

/// The field metadata key a geo column's real Delta type is kept under.
pub(crate) const TYPE_KEY: &str = "deltaswamp.geospatial.type";

/// The field metadata key marking a field whose array elements or map keys or
/// values hold a geo type, which has no field of its own to carry it.
pub(crate) const NESTED_KEY: &str = "deltaswamp.geospatial.nested";

/// The table feature.
pub(crate) const FEATURE: &str = "geospatial";

/// Whether a Delta type string is a geo type.
fn is_geo(type_name: &str) -> bool {
    let lower = type_name.trim().to_ascii_lowercase();
    (lower.starts_with("geometry(") || lower.starts_with("geography(")) && lower.ends_with(')')
}

/// Rewrite every geo type in a schema JSON value to `binary`, marking the
/// struct fields that held one. Returns (changed, geo values a write could
/// not annotate: an array element, a map key or value, or any field inside
/// one of those -- arrow-rs's Parquet schema is annotated through structs
/// only).
fn rewrite(value: &mut serde_json::Value) -> (bool, bool) {
    rewrite_in(value, false)
}

fn rewrite_in(value: &mut serde_json::Value, inside: bool) -> (bool, bool) {
    use serde_json::Value;
    let mut changed = false;
    let mut unmarked = false;
    match value {
        Value::Object(map) => {
            // A struct field: {"name", "type", "nullable", "metadata"}.
            let is_field = map.contains_key("name") && map.contains_key("nullable");
            if is_field {
                let mark = |map: &mut serde_json::Map<String, Value>, key: &str, value: Value| {
                    let metadata = map
                        .entry("metadata")
                        .or_insert_with(|| Value::Object(Default::default()));
                    if let Value::Object(md) = metadata {
                        md.insert(key.into(), value);
                    }
                };
                if let Some(Value::String(t)) = map.get("type") {
                    if is_geo(t) {
                        let original = t.clone();
                        map.insert("type".into(), Value::String("binary".into()));
                        mark(map, TYPE_KEY, Value::String(original));
                        return (true, inside);
                    }
                }
                if let Some(ty) = map.get_mut("type") {
                    let (c, u) = rewrite_in(ty, inside);
                    if u {
                        mark(map, NESTED_KEY, Value::Bool(true));
                    }
                    return (c, u);
                }
            }
            // An array's element type and a map's key and value types: bare
            // strings for primitives, objects for nested types.
            let container = ["elementType", "keyType", "valueType"]
                .iter()
                .any(|k| map.contains_key(*k));
            for key in ["elementType", "keyType", "valueType"] {
                if let Some(Value::String(t)) = map.get(key) {
                    if is_geo(t) {
                        map.insert(key.into(), Value::String("binary".into()));
                        changed = true;
                        unmarked = true;
                    }
                }
            }
            for v in map.values_mut() {
                let (c, u) = rewrite_in(v, inside || container);
                changed |= c;
                unmarked |= u;
            }
        }
        Value::Array(items) => {
            for v in items {
                let (c, u) = rewrite_in(v, inside);
                changed |= c;
                unmarked |= u;
            }
        }
        _ => {}
    }
    (changed, unmarked)
}

/// Whether the table's logged schema holds a geo type anywhere.
pub(crate) fn schema_has_geo(schema_string: &str) -> bool {
    serde_json::from_str::<serde_json::Value>(schema_string)
        .map(|mut v| rewrite(&mut v).0)
        .unwrap_or(false)
}

/// `metadata` with its geo columns as marked binary, or None when it has none.
pub(crate) fn view_metadata(metadata: &Metadata) -> Result<Option<Metadata>> {
    let mut schema: serde_json::Value = serde_json::from_str(metadata.schema_string())
        .map_err(|e| NativeError::Invalid(format!("the table's schema is not JSON: {e}")))?;
    let (changed, _) = rewrite(&mut schema);
    if !changed {
        return Ok(None);
    }
    let mut action = serde_json::to_value(metadata)
        .map_err(|e| NativeError::Invalid(format!("could not serialize metadata: {e}")))?;
    action["schemaString"] = serde_json::Value::String(schema.to_string());
    let viewed = serde_json::from_value(action)
        .map_err(|e| NativeError::Invalid(format!("could not restate the geo schema: {e}")))?;
    Ok(Some(viewed))
}

/// `snapshot` with its geo columns as marked binary; `snapshot` itself when
/// it has none. The protocol and log segment are unchanged.
pub(crate) fn view(snapshot: SnapshotRef) -> Result<SnapshotRef> {
    use delta_kernel::table_configuration::TableConfiguration;

    let config = snapshot.table_configuration();
    let Some(metadata) = view_metadata(config.metadata())? else {
        return Ok(snapshot);
    };
    let config = TableConfiguration::try_new(
        metadata,
        config.protocol().clone(),
        snapshot.table_root().clone(),
        snapshot.version(),
    )?;
    Ok(Arc::new(Snapshot::new(
        snapshot.log_segment().clone(),
        config,
    )?))
}

/// Whether a schema -- the table's own, or the view of it -- carries a geo
/// type inside an array or map, which no write can annotate.
pub(crate) fn has_unmarked(schema_string: &str) -> bool {
    schema_string.contains(NESTED_KEY)
        || serde_json::from_str::<serde_json::Value>(schema_string)
            .map(|mut v| rewrite(&mut v).1)
            .unwrap_or(false)
}

/// The Parquet logical type for a Delta geo type string.
fn logical_type(delta_type: &str) -> Result<LogicalType> {
    let inner = |prefix: &str| -> String {
        let t = delta_type.trim();
        t[prefix.len()..t.len() - 1].to_string()
    };
    let lower = delta_type.trim().to_ascii_lowercase();
    if lower.starts_with("geometry(") {
        return Ok(LogicalType::geometry(Some(
            inner("geometry(").trim().to_string(),
        )));
    }
    let args = inner("geography(");
    let mut parts = args.splitn(2, ',');
    let crs = parts.next().unwrap_or_default().trim().to_string();
    let algorithm = match parts
        .next()
        .map(|a| a.trim().to_ascii_lowercase())
        .as_deref()
    {
        None | Some("spherical") => EdgeInterpolationAlgorithm::SPHERICAL,
        Some("vincenty") => EdgeInterpolationAlgorithm::VINCENTY,
        Some("thomas") => EdgeInterpolationAlgorithm::THOMAS,
        Some("andoyer") => EdgeInterpolationAlgorithm::ANDOYER,
        Some("karney") => EdgeInterpolationAlgorithm::KARNEY,
        Some(other) => {
            return Err(NativeError::Invalid(format!(
                "geography edge interpolation algorithm {other:?} is not one Parquet knows"
            )))
        }
    };
    Ok(LogicalType::geography(Some(crs), Some(algorithm)))
}

fn parquet_error(e: delta_kernel::parquet::errors::ParquetError) -> NativeError {
    NativeError::Invalid(format!(
        "could not type the geospatial columns for Parquet: {e}"
    ))
}

/// Parquet's schema for `converted` (the one arrow-rs derives for a data
/// file) with each column `physical` marks as geo annotated with its
/// GEOMETRY or GEOGRAPHY logical type; None when no column is geo.
///
/// Struct fields are matched by position, as the Arrow batch is the physical
/// schema's; a geo value in an array or map has no field to carry its type,
/// and is refused.
pub(crate) fn parquet_schema(
    physical: &StructType,
    converted: &SchemaDescriptor,
) -> Result<Option<SchemaDescriptor>> {
    fn marked(schema: &StructType) -> bool {
        schema.fields().any(|f| {
            f.metadata().contains_key(TYPE_KEY)
                || matches!(f.data_type(), DataType::Struct(s) if marked(s))
        })
    }
    if !marked(physical) {
        return Ok(None);
    }
    fn annotate(schema: &StructType, group: &TypePtr) -> Result<TypePtr> {
        let children = group.get_fields();
        if children.len() != schema.fields().len() {
            return Err(NativeError::Invalid(
                "the data file's schema does not line up with the table's".into(),
            ));
        }
        let mut fields = Vec::with_capacity(children.len());
        for (field, child) in schema.fields().zip(children) {
            let annotated = if let Some(delta_type) = field.metadata().get(TYPE_KEY) {
                let delta_type = delta_type.to_string();
                let delta_type = delta_type.trim_matches('"');
                let info = child.get_basic_info();
                if !child.is_primitive() || child.get_physical_type() != PhysicalType::BYTE_ARRAY {
                    return Err(NativeError::Invalid(format!(
                        "geospatial column {} is not written as a byte array",
                        field.name()
                    )));
                }
                let mut builder =
                    Type::primitive_type_builder(info.name(), PhysicalType::BYTE_ARRAY)
                        .with_repetition(info.repetition())
                        .with_logical_type(Some(logical_type(delta_type)?));
                if info.has_id() {
                    builder = builder.with_id(Some(info.id()));
                }
                Arc::new(builder.build().map_err(parquet_error)?)
            } else if let DataType::Struct(inner) = field.data_type() {
                if marked(inner) {
                    annotate(inner, child)?
                } else {
                    child.clone()
                }
            } else {
                child.clone()
            };
            fields.push(annotated);
        }
        let info = group.get_basic_info();
        let mut builder = Type::group_type_builder(info.name()).with_fields(fields);
        if info.has_repetition() {
            builder = builder.with_repetition(info.repetition());
        }
        if info.has_id() {
            builder = builder.with_id(Some(info.id()));
        }
        Ok(Arc::new(builder.build().map_err(parquet_error)?))
    }
    let root = annotate(physical, &converted.root_schema_ptr())?;
    Ok(Some(SchemaDescriptor::new(root)))
}

#[cfg(test)]
mod tests {
    use super::*;

    const SCHEMA: &str = r#"{"type":"struct","fields":[
        {"name":"id","type":"long","nullable":true,"metadata":{}},
        {"name":"g","type":"geometry(OGC:CRS84)","nullable":true,"metadata":{}},
        {"name":"h","type":"geography(OGC:CRS84, SPHERICAL)","nullable":true,"metadata":{}},
        {"name":"s","type":{"type":"struct","fields":[
            {"name":"inner","type":"geometry(EPSG:3857)","nullable":true,"metadata":{}}]},
         "nullable":true,"metadata":{}}]}"#;

    #[test]
    fn geo_columns_become_marked_binary() {
        let mut value: serde_json::Value = serde_json::from_str(SCHEMA).unwrap();
        assert_eq!(rewrite(&mut value), (true, false));
        let fields = value["fields"].as_array().unwrap();
        assert_eq!(fields[1]["type"], "binary");
        assert_eq!(fields[1]["metadata"][TYPE_KEY], "geometry(OGC:CRS84)");
        assert_eq!(
            fields[2]["metadata"][TYPE_KEY],
            "geography(OGC:CRS84, SPHERICAL)"
        );
        assert_eq!(fields[3]["type"]["fields"][0]["type"], "binary");
        assert!(schema_has_geo(SCHEMA));
        assert!(!schema_has_geo(&value.to_string()));
    }

    #[test]
    fn a_struct_in_an_array_holding_geo_is_unmarked() {
        let schema = r#"{"type":"struct","fields":[{"name":"a","type":{"type":"array",
            "elementType":{"type":"struct","fields":[{"name":"g","type":"geometry(OGC:CRS84)",
            "nullable":true,"metadata":{}}]},"containsNull":true},"nullable":true,
            "metadata":{}}]}"#;
        assert!(has_unmarked(schema));
        let mut value: serde_json::Value = serde_json::from_str(schema).unwrap();
        rewrite(&mut value);
        assert!(has_unmarked(&value.to_string()));
    }

    #[test]
    fn geo_in_an_array_is_unmarked() {
        let schema = r#"{"type":"struct","fields":[{"name":"a","type":{"type":"array",
            "elementType":"geometry(OGC:CRS84)","containsNull":true},"nullable":true,
            "metadata":{}}]}"#;
        assert!(has_unmarked(schema));
        assert!(!has_unmarked(SCHEMA));
    }

    #[test]
    fn logical_types_follow_the_delta_type() {
        assert_eq!(
            logical_type("geometry(OGC:CRS84)").unwrap(),
            LogicalType::geometry(Some("OGC:CRS84".into()))
        );
        assert_eq!(
            logical_type("geography(OGC:CRS84, SPHERICAL)").unwrap(),
            LogicalType::geography(
                Some("OGC:CRS84".into()),
                Some(EdgeInterpolationAlgorithm::SPHERICAL)
            )
        );
        assert_eq!(
            logical_type("geography(EPSG:4326, vincenty)").unwrap(),
            LogicalType::geography(
                Some("EPSG:4326".into()),
                Some(EdgeInterpolationAlgorithm::VINCENTY)
            )
        );
        assert!(logical_type("geography(EPSG:4326, flat)").is_err());
    }
}
