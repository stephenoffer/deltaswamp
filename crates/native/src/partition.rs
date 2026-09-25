//! Splitting an Arrow batch by partition values for a partitioned append.
//!
//! Kernel binds partition values per write context
//! (`WriteState::partitioned_write_context`), and the data passed to the
//! writer must *omit* the partition columns -- the kernel serializes the values
//! into the Add action and materializes them into the file only when the
//! protocol demands it. So each incoming batch is grouped by its distinct
//! partition-value tuples, each group loses its partition columns, and each
//! group is written through its own context.
//!
//! Values are handed to the kernel as typed `Scalar`s, not strings: the kernel
//! owns the Delta partition-value serialization (dates as `YYYY-MM-DD`,
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
        let array = arrow::compute::cast(batch.column(index), &target).map_err(|e| {
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

    #[test]
    fn partition_columns_match_case_insensitively() {
        let groups =
            split_by_partition(&batch(), &["REGION".to_string()], &table_schema()).unwrap();
        assert_eq!(groups.len(), 3);
    }
}
