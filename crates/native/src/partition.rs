//! Splitting an Arrow batch by partition values for a partitioned append.
//!
//! Kernel binds partition values per write context
//! (`WriteState::partitioned_write_context`), and the data passed to the
//! writer must *omit* the partition columns -- the kernel serialises the values
//! into the Add action and materialises them into the file only when the
//! protocol demands it. So each incoming batch is grouped by its distinct
//! partition-value tuples, each group loses its partition columns, and each
//! group is written through its own context.
//!
//! Values are handed to the kernel as typed `Scalar`s, not strings: the kernel
//! owns the Delta partition-value serialisation (dates as `YYYY-MM-DD`,
//! timestamps in UTC, NULL as a JSON null), so we never have to get it right
//! twice. Row order within a group is preserved; order across groups is not,
//! which is harmless on a write (there is no deletion vector to line up).

use std::collections::HashMap;
use std::sync::Arc;

use arrow::array::{Array, ArrayRef, AsArray, RecordBatch, UInt32Array};
use arrow::datatypes::{
    DataType as ArrowType, Date32Type, Decimal128Type, Float32Type, Float64Type, Int16Type,
    Int32Type, Int64Type, Int8Type, Schema as ArrowSchema, TimestampMicrosecondType,
};
use arrow::row::{RowConverter, SortField};
use delta_kernel::engine::arrow_conversion::TryIntoArrow;
use delta_kernel::expressions::Scalar;
use delta_kernel::schema::{DataType, PrimitiveType, StructType};

use crate::error::{NativeError, Result};

/// One partition's slice of a batch: its typed values and its data columns.
pub struct PartitionGroup {
    pub values: HashMap<String, Scalar>,
    pub data: RecordBatch,
}

/// Group `batch` by the values of `partition_columns` (logical names).
///
/// Column names are matched case-insensitively, as Delta does.
pub fn split_by_partition(
    batch: &RecordBatch,
    partition_columns: &[String],
    table_schema: &StructType,
) -> Result<Vec<PartitionGroup>> {
    let schema = batch.schema();

    // Locate each partition column in the batch and its kernel type.
    let mut part_indices = Vec::with_capacity(partition_columns.len());
    let mut part_arrays: Vec<ArrayRef> = Vec::with_capacity(partition_columns.len());
    let mut part_types: Vec<PrimitiveType> = Vec::with_capacity(partition_columns.len());
    for name in partition_columns {
        let index = schema
            .fields()
            .iter()
            .position(|f| f.name().eq_ignore_ascii_case(name))
            .ok_or_else(|| {
                NativeError::Invalid(format!(
                    "the table is partitioned by {name:?}, but the data has no such column; \
                     a partitioned append must supply every partition column"
                ))
            })?;
        let field = table_schema
            .fields()
            .find(|f| f.name().eq_ignore_ascii_case(name))
            .ok_or_else(|| {
                NativeError::Invalid(format!(
                    "partition column {name:?} is not in the table schema"
                ))
            })?;
        let DataType::Primitive(ptype) = field.data_type() else {
            return Err(NativeError::Invalid(format!(
                "partition column {name:?} has non-primitive type {:?}",
                field.data_type()
            )));
        };
        // Cast to the table's own Arrow type, so value extraction below only
        // has to handle canonical types (and a mismatched input is refused
        // here rather than written with the wrong value).
        let target: ArrowType = field.data_type().try_into_arrow()?;
        let array = null_partition_markers(batch.column(index), &target)
            .and_then(|a| lossless_cast(&a, &target))
            .map_err(|e| {
                NativeError::Invalid(format!(
                "partition column {name:?} cannot be converted to the table's type {target}: {e}"
            ))
            })?;
        part_indices.push(index);
        part_arrays.push(array);
        part_types.push(ptype.clone());
    }

    // Data columns: everything but the partition columns, in input order.
    let data_indices: Vec<usize> = (0..schema.fields().len())
        .filter(|i| !part_indices.contains(i))
        .collect();
    let data_schema = Arc::new(ArrowSchema::new_with_metadata(
        data_indices
            .iter()
            .map(|&i| schema.field(i).clone())
            .collect::<Vec<_>>(),
        schema.metadata().clone(),
    ));

    // Group rows by the row-format encoding of their partition tuple, keeping
    // groups in first-seen order so output is deterministic.
    let converter = RowConverter::new(
        part_arrays
            .iter()
            .map(|a| SortField::new(a.data_type().clone()))
            .collect(),
    )?;
    let rows = converter.convert_columns(&part_arrays)?;
    let mut order: Vec<Vec<u8>> = Vec::new();
    let mut groups: HashMap<Vec<u8>, Vec<u32>> = HashMap::new();
    for i in 0..batch.num_rows() {
        let key = rows.row(i).as_ref().to_vec();
        let row = u32::try_from(i).map_err(|_| {
            NativeError::Invalid("a single batch has more than 2^32 rows".to_string())
        })?;
        groups
            .entry(key.clone())
            .or_insert_with(|| {
                order.push(key);
                Vec::new()
            })
            .push(row);
    }

    let mut out = Vec::with_capacity(order.len());
    for key in order {
        let indices = UInt32Array::from(groups.remove(&key).unwrap_or_default());
        let first = indices.value(0) as usize;

        let mut values = HashMap::with_capacity(partition_columns.len());
        for ((name, array), ptype) in partition_columns.iter().zip(&part_arrays).zip(&part_types) {
            values.insert(name.clone(), scalar_at(array.as_ref(), first, ptype)?);
        }

        let columns = data_indices
            .iter()
            .map(|&i| arrow::compute::take(batch.column(i).as_ref(), &indices, None))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let data = RecordBatch::try_new_with_options(
            data_schema.clone(),
            columns,
            &arrow::array::RecordBatchOptions::new().with_row_count(Some(indices.len())),
        )?;
        out.push(PartitionGroup { values, data });
    }
    Ok(out)
}

/// Cast `array` to `target`, refusing any value the cast would change.
///
/// `arrow::compute::cast` defaults to "safe" mode, which turns a value it
/// cannot convert into NULL -- so an int64 partition value of 3e9 against an
/// int32 column, or the string "abc" against an integer one, was silently
/// written to the `__HIVE_DEFAULT_PARTITION__` (NULL) partition. Unsafe mode
/// errors on overflow and parse failures but still truncates (1.7 -> 1,
/// nanoseconds -> microseconds), so non-string inputs are also cast back and
/// compared. A partition value is the file's identity; it must not drift.
fn lossless_cast(array: &ArrayRef, target: &ArrowType) -> std::result::Result<ArrayRef, String> {
    use arrow::compute::{cast_with_options, CastOptions};
    if array.data_type() == target {
        return Ok(array.clone());
    }
    let strict = CastOptions {
        safe: false,
        ..Default::default()
    };
    let cast = cast_with_options(array, target, &strict).map_err(|e| e.to_string())?;
    if cast.null_count() != array.null_count() {
        return Err("the conversion would turn some values into NULL".to_string());
    }
    let source = array.data_type();
    let textual = matches!(
        source,
        ArrowType::Utf8 | ArrowType::LargeUtf8 | ArrowType::Utf8View
    );
    // Dictionary inputs are decoded, not converted; a string target keeps
    // whatever text the source rendered to.
    let exempt = textual
        || matches!(source, ArrowType::Dictionary(..))
        || matches!(
            target,
            ArrowType::Utf8 | ArrowType::LargeUtf8 | ArrowType::Utf8View
        );
    // Narrowing between floats (float64 data into a FLOAT column, pandas'
    // default) rounds to the nearest float as Spark's cast does; only an
    // overflow to infinity changes a value beyond that.
    let float_narrowing = source.is_floating() && target.is_floating();
    if float_narrowing {
        if non_finite_count(&cast) != non_finite_count(array) {
            return Err(format!(
                "converting {source} to {target} would overflow some values to infinity"
            ));
        }
    } else if !exempt {
        let back = cast_with_options(&cast, source, &strict).map_err(|e| e.to_string())?;
        // Timezone-only differences are not a value change.
        let same = back.to_data() == array.to_data()
            || arrow::compute::kernels::cmp::distinct(&back, array)
                .map(|d| d.true_count() == 0)
                .unwrap_or(false);
        if !same {
            return Err(format!(
                "converting {source} to {target} would change some values (lost precision \
                 or range); cast the data to the table's type explicitly"
            ));
        }
    }
    Ok(cast)
}

/// How many non-null values of a float array are infinite or NaN.
fn non_finite_count(array: &ArrayRef) -> usize {
    let Ok(wide) = arrow::compute::cast(array, &ArrowType::Float64) else {
        return 0;
    };
    wide.as_primitive::<Float64Type>()
        .iter()
        .flatten()
        .filter(|v| !v.is_finite())
        .count()
}

/// `""` and `__HIVE_DEFAULT_PARTITION__` in a text partition column bound
/// for a non-text type are NULL, as Delta serialises a NULL partition value;
/// every other string is parsed (and refused if it does not parse).
fn null_partition_markers(
    array: &ArrayRef,
    target: &ArrowType,
) -> std::result::Result<ArrayRef, String> {
    let textual = |t: &ArrowType| {
        matches!(
            t,
            ArrowType::Utf8 | ArrowType::LargeUtf8 | ArrowType::Utf8View
        )
    };
    if !textual(array.data_type()) || textual(target) {
        return Ok(array.clone());
    }
    let text = arrow::compute::cast(array, &ArrowType::Utf8).map_err(|e| e.to_string())?;
    let strings = text.as_string::<i32>();
    let marker: arrow::array::BooleanArray = strings
        .iter()
        .map(|v| v.map(|s| s.is_empty() || s == "__HIVE_DEFAULT_PARTITION__"))
        .collect();
    if marker.true_count() == 0 {
        return Ok(array.clone());
    }
    arrow::compute::nullif(&text, &marker).map_err(|e| e.to_string())
}

/// Reorder, rename and cast `batch`'s columns to match `table_schema`.
///
/// Kernel's write path binds input columns to the table schema by
/// *position*, struct fields included: data with columns `(b, a)` against a
/// table `(a, b)` of the same types was written with the values swapped, and
/// data missing a leading column had every later column shifted into the
/// wrong one. It also writes the *input's* physical types, so int32 data went
/// into a `long` column's files as INT32 and nanosecond timestamps as
/// TIMESTAMP(NANOS) -- files other Delta readers refuse. Delta matches by name,
/// case-insensitively, so this does too, at every struct level: extra columns
/// are refused, missing nullable ones are filled with NULL, missing
/// non-nullable ones are refused, and leaves are cast to the table's type
/// only when that is lossless (see [`lossless_cast`]).
pub fn conform_to_table(
    batch: &RecordBatch,
    table_schema: &StructType,
    partition_columns: &[String],
) -> Result<RecordBatch> {
    let schema = batch.schema();
    // A missing partition column is refused rather than NULL-filled: that
    // would silently route every row to the NULL partition.
    for name in partition_columns {
        if !schema
            .fields()
            .iter()
            .any(|f| f.name().eq_ignore_ascii_case(name))
        {
            return Err(NativeError::Invalid(format!(
                "the table is partitioned by {name:?}, but the data has no such column; \
                 a partitioned append must supply every partition column"
            )));
        }
    }
    // Text partition values spelled the way Delta serialises NULL become NULL
    // before the strict cast below refuses them as unparsable.
    let mut arrays = batch.columns().to_vec();
    for (i, input) in schema.fields().iter().enumerate() {
        if !partition_columns
            .iter()
            .any(|p| p.eq_ignore_ascii_case(input.name()))
        {
            continue;
        }
        let Some(field) = table_schema
            .fields()
            .find(|f| f.name().eq_ignore_ascii_case(input.name()))
        else {
            continue;
        };
        if let DataType::Primitive(_) = field.data_type() {
            let target: ArrowType = field.data_type().try_into_arrow()?;
            arrays[i] = null_partition_markers(&arrays[i], &target).map_err(|e| {
                NativeError::Invalid(format!("partition column {:?}: {e}", input.name()))
            })?;
        }
    }
    let (fields, columns) =
        conform_fields(schema.fields(), &arrays, batch.num_rows(), table_schema, "")?;
    Ok(RecordBatch::try_new_with_options(
        Arc::new(ArrowSchema::new_with_metadata(
            fields,
            schema.metadata().clone(),
        )),
        columns,
        &arrow::array::RecordBatchOptions::new().with_row_count(Some(batch.num_rows())),
    )?)
}

fn conform_fields(
    input: &arrow::datatypes::Fields,
    arrays: &[ArrayRef],
    len: usize,
    target: &StructType,
    prefix: &str,
) -> Result<(Vec<arrow::datatypes::Field>, Vec<ArrayRef>)> {
    use arrow::datatypes::Field;
    // Delta names are case-insensitive. A second `region` (or `REGION`) was
    // reported as a column "not in the table schema", which it plainly is.
    let mut names = std::collections::HashSet::new();
    if let Some(dup) = input
        .iter()
        .find(|f| !names.insert(f.name().to_lowercase()))
    {
        return Err(NativeError::Invalid(format!(
            "the data has column {:?} more than once (Delta column names are \
             case-insensitive); drop or rename the duplicate",
            format!("{prefix}{}", dup.name())
        )));
    }
    let mut used = vec![false; input.len()];
    let mut fields: Vec<Field> = Vec::new();
    let mut columns: Vec<ArrayRef> = Vec::new();
    for table_field in target.fields() {
        let name = table_field.name();
        let path = format!("{prefix}{name}");
        let exact = input.iter().position(|f| f.name() == name);
        let index = exact.or_else(|| {
            let mut hits = input
                .iter()
                .enumerate()
                .filter(|(_, f)| f.name().eq_ignore_ascii_case(name));
            let first = hits.next().map(|(i, _)| i);
            if hits.next().is_some() {
                None
            } else {
                first
            }
        });
        match index {
            Some(i) if used[i] => {
                return Err(NativeError::Invalid(format!(
                    "data column {:?} matches more than one table column",
                    input[i].name()
                )))
            }
            Some(i) => {
                used[i] = true;
                let array = conform_array(&arrays[i], table_field.data_type(), &path)?;
                fields.push(Field::new(
                    name.clone(),
                    array.data_type().clone(),
                    // A NULL partition marker may have introduced nulls.
                    input[i].is_nullable() || array.null_count() > 0,
                ));
                columns.push(array);
            }
            None if table_field.is_nullable() && !has_computed_value(table_field) => {
                let arrow_type: ArrowType = table_field.data_type().try_into_arrow()?;
                fields.push(Field::new(name.clone(), arrow_type.clone(), true));
                columns.push(arrow::array::new_null_array(&arrow_type, len));
            }
            None if has_computed_value(table_field) => {
                // Kernel leaves column defaults to the connector
                // (`Transaction::top_level_column_defaults`); NULL would be
                // the wrong value, silently.
                return Err(NativeError::Invalid(format!(
                    "the data has no column {path:?}, which the table fills from a default, \
                     generation or identity expression that this writer does not evaluate; \
                     supply the column's values"
                )));
            }
            None => {
                return Err(NativeError::Invalid(format!(
                    "the data has no column {path:?}, which the table declares NOT NULL"
                )))
            }
        }
    }
    let extra: Vec<String> = input
        .iter()
        .zip(&used)
        .filter(|(_, u)| !**u)
        .map(|(f, _)| format!("{prefix}{}", f.name()))
        .collect();
    if !extra.is_empty() {
        return Err(NativeError::Invalid(format!(
            "the data has columns {extra:?} that are not in the table schema; \
             evolve the schema first or drop them"
        )));
    }
    Ok((fields, columns))
}

/// Whether the table computes `field`'s value when a writer omits it.
fn has_computed_value(field: &delta_kernel::schema::StructField) -> bool {
    use delta_kernel::schema::ColumnMetadataKey as Key;
    [
        Key::CurrentDefault,
        Key::GenerationExpression,
        Key::IdentityStart,
    ]
    .iter()
    .any(|key| field.get_config_value(key).is_some())
}

/// Conform one column to its table type; see [`conform_to_table`].
fn conform_array(array: &ArrayRef, target: &DataType, path: &str) -> Result<ArrayRef> {
    use arrow::array::{ListArray, MapArray, StructArray};
    use arrow::datatypes::Field;
    let mismatch = |what: &str| {
        NativeError::Invalid(format!(
            "column {path:?} is {} in the data but {what} in the table",
            array.data_type()
        ))
    };
    match target {
        DataType::Struct(st) => {
            let sa = array.as_struct_opt().ok_or_else(|| mismatch("a struct"))?;
            let (fields, columns) =
                conform_fields(sa.fields(), sa.columns(), sa.len(), st, &format!("{path}."))?;
            let rebuilt = if fields.is_empty() {
                StructArray::new_empty_fields(sa.len(), sa.nulls().cloned())
            } else {
                StructArray::try_new(fields.into(), columns, sa.nulls().cloned())?
            };
            Ok(Arc::new(rebuilt))
        }
        DataType::Array(at) => {
            let array = match array.data_type() {
                ArrowType::List(_) => array.clone(),
                ArrowType::LargeList(f)
                | ArrowType::ListView(f)
                | ArrowType::FixedSizeList(f, _) => {
                    arrow::compute::cast(array, &ArrowType::List(f.clone()))?
                }
                _ => return Err(mismatch("an array")),
            };
            let list = array.as_list::<i32>();
            let ArrowType::List(item) = list.data_type() else {
                return Err(mismatch("an array"));
            };
            let values = conform_array(list.values(), at.element_type(), &format!("{path}[]"))?;
            let item = Field::new(item.name(), values.data_type().clone(), item.is_nullable());
            Ok(Arc::new(ListArray::try_new(
                Arc::new(item),
                list.offsets().clone(),
                values,
                list.nulls().cloned(),
            )?))
        }
        DataType::Map(mt) => {
            let map = array.as_map_opt().ok_or_else(|| mismatch("a map"))?;
            let ArrowType::Map(entries_field, sorted) = map.data_type() else {
                return Err(mismatch("a map"));
            };
            let ArrowType::Struct(entry_fields) = entries_field.data_type() else {
                return Err(mismatch("a map"));
            };
            let keys = conform_array(map.keys(), mt.key_type(), &format!("{path}.key"))?;
            let values = conform_array(map.values(), mt.value_type(), &format!("{path}.value"))?;
            let fields: Vec<Field> = [&keys, &values]
                .iter()
                .zip(entry_fields.iter())
                .map(|(a, f)| Field::new(f.name(), a.data_type().clone(), f.is_nullable()))
                .collect();
            let entries = StructArray::try_new(fields.clone().into(), vec![keys, values], None)?;
            let entries_field = Field::new(
                entries_field.name(),
                entries.data_type().clone(),
                entries_field.is_nullable(),
            );
            Ok(Arc::new(MapArray::try_new(
                Arc::new(entries_field),
                map.offsets().clone(),
                entries,
                map.nulls().cloned(),
                *sorted,
            )?))
        }
        DataType::Primitive(_) => {
            let arrow_type: ArrowType = target.try_into_arrow()?;
            if array.data_type() == &arrow_type {
                return Ok(array.clone());
            }
            lossless_cast(array, &arrow_type).map_err(|e| {
                NativeError::Invalid(format!(
                    "column {path:?} cannot be written as the table's type {arrow_type}: {e}"
                ))
            })
        }
        // Variant and other kernel-specific types: leave them to kernel.
        _ => Ok(array.clone()),
    }
}

/// Arrow memory per written file above which input batches stop merging.
const COALESCE_TARGET_BYTES: usize = 128 * 1024 * 1024;

/// Merge consecutive small batches, preserving row order.
///
/// Kernel writes one Parquet file per `write_parquet` call, and the write
/// path makes one call per input batch (per partition): a 10,000-row table
/// arriving as 1-row batches became 10,000 files, plus 10,000 add actions in
/// the log. Empty batches are dropped; they would each write an empty file.
pub fn coalesce(batches: Vec<RecordBatch>) -> Result<Vec<RecordBatch>> {
    let mut out = Vec::new();
    let mut pending: Vec<RecordBatch> = Vec::new();
    let mut pending_bytes = 0usize;
    let flush = |pending: &mut Vec<RecordBatch>, out: &mut Vec<RecordBatch>| -> Result<()> {
        match pending.len() {
            0 => {}
            1 => out.push(pending.pop().expect("one pending batch")),
            _ => {
                let schema = pending[0].schema();
                out.push(arrow::compute::concat_batches(&schema, pending.iter())?);
                pending.clear();
            }
        }
        Ok(())
    };
    for batch in batches {
        if batch.num_rows() == 0 {
            continue;
        }
        let size = batch.get_array_memory_size();
        // Only batches with an identical schema can be concatenated.
        let compatible = pending.first().is_none_or(|p| p.schema() == batch.schema());
        if !compatible || (pending_bytes + size > COALESCE_TARGET_BYTES && !pending.is_empty()) {
            flush(&mut pending, &mut out)?;
            pending_bytes = 0;
        }
        pending_bytes += size;
        pending.push(batch);
    }
    flush(&mut pending, &mut out)?;
    Ok(out)
}

/// Widen unsigned Arrow integers to a signed type that holds every value.
///
/// Kernel maps `uint8` to `byte`, `uint16` to `short` and `uint32` to
/// `integer` -- same width, half the range -- so a table created from a
/// `uint8` schema could not store 200. Delta has no unsigned types; the next
/// signed width up is what Spark and pandas users expect. `uint64` has no
/// wider integer and stays `long`; out-of-range values are refused on write.
pub fn widen_unsigned(data_type: &ArrowType) -> ArrowType {
    use arrow::datatypes::Field;
    let field = |f: &arrow::datatypes::FieldRef| -> arrow::datatypes::FieldRef {
        Arc::new(
            Field::new(f.name(), widen_unsigned(f.data_type()), f.is_nullable())
                .with_metadata(f.metadata().clone()),
        )
    };
    match data_type {
        ArrowType::UInt8 => ArrowType::Int16,
        ArrowType::UInt16 => ArrowType::Int32,
        ArrowType::UInt32 | ArrowType::UInt64 => ArrowType::Int64,
        ArrowType::Struct(fields) => ArrowType::Struct(fields.iter().map(field).collect()),
        ArrowType::List(f) => ArrowType::List(field(f)),
        ArrowType::LargeList(f) => ArrowType::LargeList(field(f)),
        ArrowType::FixedSizeList(f, n) => ArrowType::FixedSizeList(field(f), *n),
        ArrowType::Map(f, sorted) => ArrowType::Map(field(f), *sorted),
        other => other.clone(),
    }
}

/// The value at `row` as a kernel scalar of `ptype`.
///
/// `array` has already been cast to the table's canonical Arrow type.
fn scalar_at(array: &dyn Array, row: usize, ptype: &PrimitiveType) -> Result<Scalar> {
    if array.is_null(row) {
        return Ok(Scalar::Null(DataType::Primitive(ptype.clone())));
    }
    use PrimitiveType as P;
    let scalar = match ptype {
        P::String => Scalar::String(string_at(array, row)?),
        P::Long => Scalar::Long(array.as_primitive::<Int64Type>().value(row)),
        P::Integer => Scalar::Integer(array.as_primitive::<Int32Type>().value(row)),
        P::Short => Scalar::Short(array.as_primitive::<Int16Type>().value(row)),
        P::Byte => Scalar::Byte(array.as_primitive::<Int8Type>().value(row)),
        P::Float => Scalar::Float(array.as_primitive::<Float32Type>().value(row)),
        P::Double => Scalar::Double(array.as_primitive::<Float64Type>().value(row)),
        P::Boolean => Scalar::Boolean(array.as_boolean().value(row)),
        P::Date => Scalar::Date(array.as_primitive::<Date32Type>().value(row)),
        P::Timestamp => {
            Scalar::Timestamp(array.as_primitive::<TimestampMicrosecondType>().value(row))
        }
        P::TimestampNtz => {
            Scalar::TimestampNtz(array.as_primitive::<TimestampMicrosecondType>().value(row))
        }
        P::Decimal(dt) => Scalar::Decimal(delta_kernel::expressions::DecimalData::try_new(
            array.as_primitive::<Decimal128Type>().value(row),
            *dt,
        )?),
        P::Binary => Scalar::Binary(binary_at(array, row)?),
        other => {
            return Err(NativeError::Invalid(format!(
                "partition columns of type {other:?} are not supported on the kernel append path"
            )))
        }
    };
    Ok(scalar)
}

fn string_at(array: &dyn Array, row: usize) -> Result<String> {
    match array.data_type() {
        ArrowType::Utf8 => Ok(array.as_string::<i32>().value(row).to_string()),
        ArrowType::LargeUtf8 => Ok(array.as_string::<i64>().value(row).to_string()),
        ArrowType::Utf8View => Ok(array.as_string_view().value(row).to_string()),
        other => Err(NativeError::Invalid(format!(
            "unexpected Arrow type {other} for a string partition column"
        ))),
    }
}

fn binary_at(array: &dyn Array, row: usize) -> Result<Vec<u8>> {
    match array.data_type() {
        ArrowType::Binary => Ok(array.as_binary::<i32>().value(row).to_vec()),
        ArrowType::LargeBinary => Ok(array.as_binary::<i64>().value(row).to_vec()),
        ArrowType::BinaryView => Ok(array.as_binary_view().value(row).to_vec()),
        other => Err(NativeError::Invalid(format!(
            "unexpected Arrow type {other} for a binary partition column"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Int64Array, StringArray};
    use arrow::datatypes::Field;
    use delta_kernel::schema::StructField;

    fn table_schema() -> StructType {
        StructType::try_new([
            StructField::nullable("id", DataType::LONG),
            StructField::nullable("region", DataType::STRING),
        ])
        .unwrap()
    }

    fn batch() -> RecordBatch {
        RecordBatch::try_new(
            Arc::new(ArrowSchema::new(vec![
                Field::new("id", ArrowType::Int64, true),
                Field::new("region", ArrowType::Utf8, true),
            ])),
            vec![
                Arc::new(Int64Array::from(vec![1, 2, 3, 4, 5])),
                Arc::new(StringArray::from(vec![
                    Some("eu"),
                    Some("us"),
                    Some("eu"),
                    None,
                    Some("us"),
                ])),
            ],
        )
        .unwrap()
    }

    #[test]
    fn groups_by_value_in_first_seen_order_and_strips_partition_columns() {
        let groups =
            split_by_partition(&batch(), &["region".to_string()], &table_schema()).unwrap();
        assert_eq!(groups.len(), 3);

        let ids = |g: &PartitionGroup| -> Vec<i64> {
            g.data
                .column(0)
                .as_primitive::<Int64Type>()
                .values()
                .to_vec()
        };
        assert_eq!(groups[0].values["region"], Scalar::String("eu".into()));
        assert_eq!(ids(&groups[0]), [1, 3]);
        assert_eq!(groups[1].values["region"], Scalar::String("us".into()));
        assert_eq!(ids(&groups[1]), [2, 5]);
        assert!(matches!(groups[2].values["region"], Scalar::Null(_)));
        assert_eq!(ids(&groups[2]), [4]);

        for g in &groups {
            assert_eq!(g.data.num_columns(), 1, "partition column must be removed");
            assert_eq!(g.data.schema().field(0).name(), "id");
        }
    }

    #[test]
    fn missing_partition_column_is_refused() {
        let schema = StructType::try_new([
            StructField::nullable("id", DataType::LONG),
            StructField::nullable("day", DataType::DATE),
        ])
        .unwrap();
        let err = split_by_partition(&batch(), &["day".to_string()], &schema)
            .err()
            .unwrap();
        assert!(err.to_string().contains("no such column"), "{err}");
    }

    fn two_longs() -> StructType {
        StructType::try_new([
            StructField::nullable("a", DataType::LONG),
            StructField::nullable("b", DataType::LONG),
        ])
        .unwrap()
    }

    fn longs(names: &[&str], values: &[i64]) -> RecordBatch {
        RecordBatch::try_new(
            Arc::new(ArrowSchema::new(
                names
                    .iter()
                    .map(|n| Field::new(*n, ArrowType::Int64, true))
                    .collect::<Vec<_>>(),
            )),
            values
                .iter()
                .map(|v| Arc::new(Int64Array::from(vec![*v])) as ArrayRef)
                .collect(),
        )
        .unwrap()
    }

    #[test]
    fn columns_are_matched_by_name_not_position() {
        let out = conform_to_table(&longs(&["b", "a"], &[1, 2]), &two_longs(), &[]).unwrap();
        assert_eq!(out.schema().field(0).name(), "a");
        assert_eq!(out.column(0).as_primitive::<Int64Type>().value(0), 2);
        assert_eq!(out.column(1).as_primitive::<Int64Type>().value(0), 1);
    }

    #[test]
    fn float64_data_narrows_into_a_float_column_unless_it_overflows() {
        let table = StructType::try_new([StructField::nullable("f", DataType::FLOAT)]).unwrap();
        let batch = |v: f64| {
            RecordBatch::try_new(
                Arc::new(ArrowSchema::new(vec![Field::new(
                    "f",
                    ArrowType::Float64,
                    true,
                )])),
                vec![Arc::new(arrow::array::Float64Array::from(vec![v])) as ArrayRef],
            )
            .unwrap()
        };
        let out = conform_to_table(&batch(0.1), &table, &[]).unwrap();
        assert_eq!(out.column(0).data_type(), &ArrowType::Float32);
        let err = conform_to_table(&batch(1e300), &table, &[]).unwrap_err();
        assert!(err.to_string().contains("infinity"), "{err}");
        assert!(conform_to_table(&batch(f64::INFINITY), &table, &[]).is_ok());
    }

    #[test]
    fn a_missing_column_with_a_default_is_refused_not_nulled() {
        use delta_kernel::schema::MetadataValue;
        let table = StructType::try_new([
            StructField::nullable("a", DataType::LONG),
            StructField::nullable("b", DataType::LONG).with_metadata([(
                "CURRENT_DEFAULT".to_string(),
                MetadataValue::String("42".to_string()),
            )]),
        ])
        .unwrap();
        let err = conform_to_table(&longs(&["a"], &[1]), &table, &[]).unwrap_err();
        assert!(err.to_string().contains("default"), "{err}");
    }

    #[test]
    fn serialised_null_partition_strings_are_null_and_timestamps_parse() {
        let table = StructType::try_new([
            StructField::nullable("id", DataType::LONG),
            StructField::nullable("p", DataType::INTEGER),
            StructField::nullable("ts", DataType::TIMESTAMP),
        ])
        .unwrap();
        let batch = RecordBatch::try_new(
            Arc::new(ArrowSchema::new(vec![
                Field::new("id", ArrowType::Int64, false),
                Field::new("p", ArrowType::Utf8, false),
                Field::new("ts", ArrowType::Utf8, false),
            ])),
            vec![
                Arc::new(Int64Array::from(vec![1, 2, 3])) as ArrayRef,
                Arc::new(StringArray::from(vec![
                    "",
                    "__HIVE_DEFAULT_PARTITION__",
                    "7",
                ])),
                Arc::new(StringArray::from(vec![
                    "2024-01-01 00:00:00",
                    "2024-01-01T00:00:00.000000Z",
                    "__HIVE_DEFAULT_PARTITION__",
                ])),
            ],
        )
        .unwrap();
        let parts = ["p".to_string(), "ts".to_string()];
        let out = conform_to_table(&batch, &table, &parts).unwrap();
        let p = out.column(1).as_primitive::<Int32Type>();
        assert!(p.is_null(0) && p.is_null(1) && p.value(2) == 7);
        let ts = out.column(2).as_primitive::<TimestampMicrosecondType>();
        assert_eq!(ts.value(0), 1_704_067_200_000_000);
        assert_eq!(ts.value(1), 1_704_067_200_000_000);
        assert!(ts.is_null(2));
        let groups = split_by_partition(&out, &parts, &table).unwrap();
        assert_eq!(groups.len(), 2); // (NULL, midnight) twice, then (7, NULL)
                                     // Still refused: a string that is neither a value nor a NULL marker.
        let bad = RecordBatch::try_new(
            Arc::new(ArrowSchema::new(vec![
                Field::new("id", ArrowType::Int64, true),
                Field::new("p", ArrowType::Utf8, true),
            ])),
            vec![
                Arc::new(Int64Array::from(vec![1])) as ArrayRef,
                Arc::new(StringArray::from(vec!["abc"])),
            ],
        )
        .unwrap();
        assert!(conform_to_table(&bad, &table, &["p".to_string()]).is_err());
    }

    #[test]
    fn a_missing_nullable_column_is_null_not_a_shift() {
        let out = conform_to_table(&longs(&["b"], &[9]), &two_longs(), &[]).unwrap();
        assert!(out.column(0).is_null(0));
        assert_eq!(out.column(1).as_primitive::<Int64Type>().value(0), 9);
    }

    #[test]
    fn extra_and_missing_required_columns_are_refused() {
        let err =
            conform_to_table(&longs(&["a", "b", "c"], &[1, 2, 3]), &two_longs(), &[]).unwrap_err();
        assert!(err.to_string().contains("\"c\""), "{err}");
        let required = StructType::try_new([StructField::not_null("a", DataType::LONG)]).unwrap();
        let err = conform_to_table(&longs(&["b"], &[1]), &required, &[]).unwrap_err();
        assert!(err.to_string().contains("NOT NULL"), "{err}");
    }

    #[test]
    fn narrower_input_types_are_cast_to_the_table_type() {
        let batch = RecordBatch::try_new(
            Arc::new(ArrowSchema::new(vec![Field::new(
                "a",
                ArrowType::Int32,
                true,
            )])),
            vec![Arc::new(arrow::array::Int32Array::from(vec![7]))],
        )
        .unwrap();
        let out = conform_to_table(&batch, &two_longs(), &[]).unwrap();
        assert_eq!(out.column(0).data_type(), &ArrowType::Int64);
    }

    #[test]
    fn struct_fields_are_matched_by_name() {
        use arrow::array::StructArray;
        let table = StructType::try_new([StructField::nullable("s", two_longs())]).unwrap();
        let inner = StructArray::from(vec![
            (
                Arc::new(Field::new("b", ArrowType::Int64, true)),
                Arc::new(Int64Array::from(vec![1])) as ArrayRef,
            ),
            (
                Arc::new(Field::new("a", ArrowType::Int64, true)),
                Arc::new(Int64Array::from(vec![2])) as ArrayRef,
            ),
        ]);
        let batch = RecordBatch::try_new(
            Arc::new(ArrowSchema::new(vec![Field::new(
                "s",
                inner.data_type().clone(),
                true,
            )])),
            vec![Arc::new(inner)],
        )
        .unwrap();
        let out = conform_to_table(&batch, &table, &[]).unwrap();
        let s = out.column(0).as_struct();
        assert_eq!(
            s.column_by_name("a")
                .unwrap()
                .as_primitive::<Int64Type>()
                .value(0),
            2
        );
        assert_eq!(s.fields()[0].name(), "a");
    }

    #[test]
    fn lossy_casts_are_refused() {
        let big: ArrayRef = Arc::new(Int64Array::from(vec![3_000_000_000]));
        assert!(lossless_cast(&big, &ArrowType::Int32).is_err());
        let text: ArrayRef = Arc::new(StringArray::from(vec!["abc"]));
        assert!(lossless_cast(&text, &ArrowType::Int32).is_err());
        let frac: ArrayRef = Arc::new(arrow::array::Float64Array::from(vec![1.7]));
        assert!(lossless_cast(&frac, &ArrowType::Int32).is_err());
        let whole: ArrayRef = Arc::new(arrow::array::Float64Array::from(vec![2.0]));
        assert!(lossless_cast(&whole, &ArrowType::Int32).is_ok());
        let nanos: ArrayRef = Arc::new(arrow::array::TimestampNanosecondArray::from(vec![1_001]));
        let micros = ArrowType::Timestamp(arrow::datatypes::TimeUnit::Microsecond, None);
        assert!(lossless_cast(&nanos, &micros).is_err());
        let exact: ArrayRef = Arc::new(arrow::array::TimestampNanosecondArray::from(vec![2_000]));
        assert!(lossless_cast(&exact, &micros).is_ok());
    }

    #[test]
    fn unsigned_types_widen() {
        assert_eq!(widen_unsigned(&ArrowType::UInt8), ArrowType::Int16);
        assert_eq!(widen_unsigned(&ArrowType::UInt16), ArrowType::Int32);
        assert_eq!(widen_unsigned(&ArrowType::UInt32), ArrowType::Int64);
        let list = ArrowType::List(Arc::new(Field::new("item", ArrowType::UInt8, true)));
        assert_eq!(
            widen_unsigned(&list),
            ArrowType::List(Arc::new(Field::new("item", ArrowType::Int16, true)))
        );
    }

    #[test]
    fn small_batches_coalesce_in_order_and_empty_ones_vanish() {
        let batches = vec![
            longs(&["a"], &[1]),
            longs(&["a"], &[2]).slice(0, 0),
            longs(&["a"], &[3]),
        ];
        let out = coalesce(batches).unwrap();
        assert_eq!(out.len(), 1);
        let values = out[0]
            .column(0)
            .as_primitive::<Int64Type>()
            .values()
            .to_vec();
        assert_eq!(values, [1, 3]);
        assert!(coalesce(vec![longs(&["a"], &[1]).slice(0, 0)])
            .unwrap()
            .is_empty());
    }

    #[test]
    fn overflowing_partition_value_is_refused_not_nulled() {
        let schema = StructType::try_new([
            StructField::nullable("id", DataType::LONG),
            StructField::nullable("p", DataType::INTEGER),
        ])
        .unwrap();
        let batch = longs(&["id", "p"], &[1, 3_000_000_000]);
        let err = split_by_partition(&batch, &["p".to_string()], &schema)
            .err()
            .unwrap();
        assert!(err.to_string().contains("Int32"), "{err}");
    }

    #[test]
    fn partition_columns_match_case_insensitively() {
        let groups =
            split_by_partition(&batch(), &["REGION".to_string()], &table_schema()).unwrap();
        assert_eq!(groups.len(), 3);
    }
}
