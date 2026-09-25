//! A JSON predicate AST, converted into a kernel `Predicate` for file skipping.
//!
//! The predicate we hand the kernel is **only an optimization**. It lets the
//! scan drop whole files whose statistics or partition values prove no row can
//! match; it never filters rows (kernel disables Parquet pushdown until it can
//! track row indexes, which is also what keeps deletion vectors positional).
//! Python applies the exact row filter afterwards.
//!
//! That framing dictates the one rule this module lives by: **a converted
//! predicate may be weaker than the original, never stronger.** A weaker
//! predicate skips fewer files -- slower, still correct. A stronger one skips a
//! file that holds matching rows, and those rows silently vanish. So:
//!
//! * An unconvertible conjunct of an `AND` is dropped (`a AND ?` becomes `a`,
//!   which is weaker) -- but only when the caller tolerates weakening.
//! * `OR` needs every child: `a OR ?` weakened to `a` would be *stronger*.
//! * `NOT` flips polarity: `NOT (a AND ?)` weakened to `NOT a` is stronger, so a
//!   `NOT` converts its child in *exact* mode, where nothing may be dropped.
//! * A literal that cannot be coerced to the column's type *exactly* (a double
//!   that is not representable as `float`, a decimal with too many fraction
//!   digits, a naive timestamp against a zoned one) makes its comparison
//!   unconvertible rather than rounded, because rounding moves the boundary.
//! * An unknown column, a non-primitive column, an unknown `op` or literal type:
//!   all unconvertible. Never an error.
//!
//! The only error is JSON that does not parse at all, because that is a caller
//! bug rather than an unsupported predicate.

use delta_kernel::expressions::{ColumnName, Expression, Predicate, Scalar};
use delta_kernel::schema::{DataType, PrimitiveType, StructType};
use serde_json::Value;

use crate::error::{NativeError, Result};

/// Parse a JSON predicate and convert it for data skipping against `schema`.
///
/// `Ok(None)` means "no skipping": either nothing was supplied or nothing in it
/// could be converted safely. Only malformed JSON is an error.
pub fn parse_predicate(json: Option<&str>, schema: &StructType) -> Result<Option<Predicate>> {
    let Some(json) = json else {
        return Ok(None);
    };
    let value: Value = serde_json::from_str(json).map_err(|e| {
        NativeError::Invalid(format!(
            "predicate is not valid JSON ({e}); expected an object such as \
             {{\"op\":\"eq\",\"args\":[{{\"column\":[\"a\"]}},{{\"literal\":1,\"type\":\"long\"}}]}}"
        ))
    })?;
    Ok(convert(&value, schema, false))
}

/// Convert one AST node.
///
/// With `exact`, the result must be logically equivalent to the input (or
/// `None`); without it, the result may be weaker. See the module docs.
fn convert(node: &Value, schema: &StructType, exact: bool) -> Option<Predicate> {
    let op = node.get("op")?.as_str()?;
    let args = node.get("args")?.as_array()?;
    match op {
        "and" => {
            let converted: Vec<Option<Predicate>> =
                args.iter().map(|a| convert(a, schema, exact)).collect();
            if exact && converted.iter().any(Option::is_none) {
                return None;
            }
            let mut kept: Vec<Predicate> = converted.into_iter().flatten().collect();
            match kept.len() {
                0 => None,
                1 => kept.pop(),
                _ => Some(Predicate::and_from(kept)),
            }
        }
        "or" => {
            // Every child is needed; a weakened child is still fine (an OR of
            // weaker predicates is weaker), so `exact` passes through.
            let children: Option<Vec<Predicate>> =
                args.iter().map(|a| convert(a, schema, exact)).collect();
            let mut children = children?;
            match children.len() {
                0 => None,
                1 => children.pop(),
                _ => Some(Predicate::or_from(children)),
            }
        }
        "not" => {
            let [child] = args.as_slice() else {
                return None;
            };
            convert(child, schema, true).map(Predicate::not)
        }
        "is_null" | "is_not_null" => {
            let [arg] = args.as_slice() else {
                return None;
            };
            let (column, _) = resolve_column(arg, schema)?;
            let expr = Expression::column(column);
            Some(if op == "is_null" {
                Predicate::is_null(expr)
            } else {
                Predicate::is_not_null(expr)
            })
        }
        "eq" | "ne" | "lt" | "le" | "gt" | "ge" => {
            let [left, right] = args.as_slice() else {
                return None;
            };
            let (l, r) = comparison_operands(left, right, schema)?;
            Some(match op {
                "eq" => Predicate::eq(l, r),
                "ne" => Predicate::ne(l, r),
                "lt" => Predicate::lt(l, r),
                "le" => Predicate::le(l, r),
                "gt" => Predicate::gt(l, r),
                _ => Predicate::ge(l, r),
            })
        }
        _ => None,
    }
}

/// Resolve both sides of a comparison, coercing a literal to its column's type.
fn comparison_operands(
    left: &Value,
    right: &Value,
    schema: &StructType,
) -> Option<(Expression, Expression)> {
    match (resolve_column(left, schema), resolve_column(right, schema)) {
        (Some((lc, lt)), Some((rc, rt))) => {
            // Column-to-column: only when kernel can compare them without a cast.
            (lt == rt).then(|| (Expression::column(lc), Expression::column(rc)))
        }
        (Some((col, ty)), None) => {
            let scalar = coerce_literal(right, &ty)?;
            Some((Expression::column(col), Expression::literal(scalar)))
        }
        (None, Some((col, ty))) => {
            let scalar = coerce_literal(left, &ty)?;
            Some((Expression::literal(scalar), Expression::column(col)))
        }
        // Two literals (a constant), or an unknown column on either side.
        (None, None) => None,
    }
}

/// Resolve `{"column": [...]}` against the schema, case-insensitively.
///
/// Returns the column with the schema's own spelling (kernel resolves names
/// exactly) and its primitive type. Non-primitive leaves are unconvertible:
/// kernel keeps no min/max statistics for them.
fn resolve_column(node: &Value, schema: &StructType) -> Option<(ColumnName, PrimitiveType)> {
    let path = node.get("column")?;
    let parts: Vec<&str> = match path {
        Value::Array(items) => items.iter().map(Value::as_str).collect::<Option<_>>()?,
        Value::String(s) => vec![s.as_str()],
        _ => return None,
    };
    if parts.is_empty() {
        return None;
    }

    let mut current = schema;
    let mut resolved = Vec::with_capacity(parts.len());
    let mut leaf: Option<&DataType> = None;
    for (i, part) in parts.iter().enumerate() {
        let field = current.field(part).or_else(|| {
            current
                .fields()
                .find(|f| f.name().eq_ignore_ascii_case(part))
        })?;
        resolved.push(field.name().clone());
        if i + 1 < parts.len() {
            match field.data_type() {
                DataType::Struct(inner) => current = inner,
                _ => return None,
            }
        } else {
            leaf = Some(field.data_type());
        }
    }
    match leaf? {
        DataType::Primitive(p) => Some((ColumnName::new(resolved), p.clone())),
        _ => None,
    }
}

/// Coerce a `{"literal": v, "type": t}` node to exactly `target`, or give up.
fn coerce_literal(node: &Value, target: &PrimitiveType) -> Option<Scalar> {
    let value = node.get("literal")?;
    let declared = node.get("type").and_then(Value::as_str).unwrap_or("");
    use PrimitiveType as P;
    match value {
        // A comparison with NULL is never true; there is nothing to skip on.
        Value::Null => None,
        Value::Bool(b) => matches!(target, P::Boolean).then_some(Scalar::Boolean(*b)),
        Value::Number(n) => match target {
            P::Byte | P::Short | P::Integer | P::Long => {
                // `parse_scalar` range-checks; "2.0" fails, which is fine.
                target.parse_scalar(&n.to_string()).ok()
            }
            P::Float => {
                let f = n.as_f64()?;
                let narrowed = f as f32;
                // Narrowing that moves the value moves the comparison boundary.
                (f64::from(narrowed) == f && f.is_finite()).then_some(Scalar::Float(narrowed))
            }
            P::Double => {
                if let Some(i) = n.as_i64() {
                    // Integers beyond 2^53 are not exactly representable.
                    let d = i as f64;
                    (d as i64 == i).then_some(Scalar::Double(d))
                } else {
                    n.as_f64().map(Scalar::Double)
                }
            }
            P::Decimal(dt) => normalize_decimal(&n.to_string(), dt.scale())
                .and_then(|raw| target.parse_scalar(&raw).ok()),
            P::Timestamp if matches!(declared, "timestamp" | "timestamp_ntz") => {
                n.as_i64().map(Scalar::Timestamp)
            }
            P::TimestampNtz if matches!(declared, "timestamp" | "timestamp_ntz") => {
                n.as_i64().map(Scalar::TimestampNtz)
            }
            _ => None,
        },
        Value::String(s) => match target {
            // Never via parse_scalar: it maps "" to NULL, which is a different
            // predicate from `= ''`.
            P::String => Some(Scalar::String(s.clone())),
            P::Boolean | P::Byte | P::Short | P::Integer | P::Long | P::Date | P::Timestamp => {
                if s.is_empty() {
                    None
                } else {
                    target.parse_scalar(s).ok()
                }
            }
            P::TimestampNtz => parse_timestamp_ntz(s),
            P::Decimal(dt) => {
                normalize_decimal(s, dt.scale()).and_then(|raw| target.parse_scalar(&raw).ok())
            }
            // A decimal string against a binary float column cannot be compared
            // exactly, and binary literals have no agreed JSON encoding.
            _ => None,
        },
        _ => None,
    }
}

/// Parse a zone-less timestamp; anything carrying a zone is ambiguous here.
fn parse_timestamp_ntz(s: &str) -> Option<Scalar> {
    if s.is_empty() {
        return None;
    }
    let target = PrimitiveType::TimestampNtz;
    if let Ok(v) = target.parse_scalar(s) {
        return Some(v);
    }
    // Accept the ISO `T` separator, but not an offset or `Z`: an NTZ column has
    // no zone, so converting one would guess at the caller's intent.
    let time = s.split_once('T').map(|(_, t)| t)?;
    if time.contains(['Z', 'z', '+', '-']) {
        return None;
    }
    target.parse_scalar(&s.replacen('T', " ", 1)).ok()
}

/// Rewrite a plain decimal string to exactly `scale` fraction digits.
///
/// Kernel's decimal parser demands the scale match exactly. Padding with zeros
/// is exact; dropping non-zero digits would round, so that is refused.
fn normalize_decimal(raw: &str, scale: u8) -> Option<String> {
    let raw = raw.trim();
    if raw.contains(['e', 'E']) || raw.is_empty() {
        return None;
    }
    let (sign, body) = match raw.strip_prefix('-') {
        Some(rest) => ("-", rest),
        None => ("", raw.strip_prefix('+').unwrap_or(raw)),
    };
    let (int_part, frac_part) = body.split_once('.').unwrap_or((body, ""));
    let int_part = if int_part.is_empty() { "0" } else { int_part };
    if !int_part.bytes().all(|b| b.is_ascii_digit())
        || !frac_part.bytes().all(|b| b.is_ascii_digit())
    {
        return None;
    }
    let scale = usize::from(scale);
    let frac = if frac_part.len() > scale {
        let (kept, dropped) = frac_part.split_at(scale);
        if dropped.bytes().any(|b| b != b'0') {
            return None;
        }
        kept.to_string()
    } else {
        format!("{frac_part:0<scale$}")
    };
    Some(if scale == 0 {
        format!("{sign}{int_part}")
    } else {
        format!("{sign}{int_part}.{frac}")
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use delta_kernel::schema::{DecimalType, StructField};

    fn schema() -> StructType {
        StructType::try_new([
            StructField::nullable("id", DataType::LONG),
            StructField::nullable("name", DataType::STRING),
            StructField::nullable("f", DataType::FLOAT),
            StructField::nullable("d", DataType::DOUBLE),
            StructField::nullable("day", DataType::DATE),
            StructField::nullable("ts", DataType::TIMESTAMP),
            StructField::nullable("ntz", DataType::TIMESTAMP_NTZ),
            StructField::nullable(
                "amount",
                DataType::Primitive(PrimitiveType::Decimal(DecimalType::try_new(10, 2).unwrap())),
            ),
            StructField::nullable(
                "nested",
                StructType::try_new([StructField::nullable("x", DataType::INTEGER)]).unwrap(),
            ),
            StructField::nullable(
                "tags",
                delta_kernel::schema::ArrayType::new(DataType::STRING, true),
            ),
        ])
        .unwrap()
    }

    fn conv(json: &str) -> Option<String> {
        parse_predicate(Some(json), &schema())
            .unwrap()
            .map(|p| p.to_string())
    }

    const UNCONVERTIBLE: &str =
        r#"{"op":"eq","args":[{"column":["nope"]},{"literal":1,"type":"long"}]}"#;
    const ID_GT_5: &str = r#"{"op":"gt","args":[{"column":["id"]},{"literal":5,"type":"long"}]}"#;

    #[test]
    fn malformed_json_is_the_only_error() {
        assert!(parse_predicate(Some("{not json"), &schema()).is_err());
        assert!(
            parse_predicate(Some(r#"{"op":"like","args":[]}"#), &schema())
                .unwrap()
                .is_none()
        );
        assert!(parse_predicate(Some("[1,2]"), &schema()).unwrap().is_none());
        assert!(parse_predicate(None, &schema()).unwrap().is_none());
    }

    #[test]
    fn simple_comparison_converts() {
        assert_eq!(conv(ID_GT_5).unwrap(), "Column(id) > 5");
    }

    #[test]
    fn literal_on_the_left_is_kept_on_the_left() {
        let p = conv(r#"{"op":"lt","args":[{"literal":5,"type":"long"},{"column":["id"]}]}"#);
        assert_eq!(p.unwrap(), "5 < Column(id)");
    }

    #[test]
    fn column_names_resolve_case_insensitively() {
        let p = conv(r#"{"op":"eq","args":[{"column":["ID"]},{"literal":1,"type":"long"}]}"#);
        assert_eq!(p.unwrap(), "Column(id) = 1");
    }

    #[test]
    fn nested_column_path() {
        let p =
            conv(r#"{"op":"ge","args":[{"column":["nested","x"]},{"literal":3,"type":"long"}]}"#);
        assert_eq!(p.unwrap(), "NOT(Column(nested.x) < 3)");
    }

    #[test]
    fn and_drops_only_the_unconvertible_conjunct() {
        let json = format!(r#"{{"op":"and","args":[{ID_GT_5},{UNCONVERTIBLE}]}}"#);
        assert_eq!(conv(&json).unwrap(), "Column(id) > 5");
    }

    #[test]
    fn or_with_an_unconvertible_child_is_no_predicate() {
        let json = format!(r#"{{"op":"or","args":[{ID_GT_5},{UNCONVERTIBLE}]}}"#);
        assert_eq!(conv(&json), None);
    }

    #[test]
    fn not_of_a_partially_convertible_and_is_no_predicate() {
        // NOT(a AND ?) weakened to NOT(a) would be STRONGER and skip matching
        // files; exact mode under NOT must refuse instead.
        let json =
            format!(r#"{{"op":"not","args":[{{"op":"and","args":[{ID_GT_5},{UNCONVERTIBLE}]}}]}}"#);
        assert_eq!(conv(&json), None);
    }

    #[test]
    fn or_inside_and_inside_or_propagates_none() {
        let inner_or = format!(r#"{{"op":"or","args":[{ID_GT_5},{UNCONVERTIBLE}]}}"#);
        let json = format!(r#"{{"op":"and","args":[{inner_or},{ID_GT_5}]}}"#);
        assert_eq!(conv(&json).unwrap(), "Column(id) > 5");
    }

    #[test]
    fn not_of_a_fully_convertible_predicate_converts() {
        let json = format!(r#"{{"op":"not","args":[{ID_GT_5}]}}"#);
        assert_eq!(conv(&json).unwrap(), "NOT(Column(id) > 5)");
    }

    #[test]
    fn unknown_column_and_non_primitive_column_are_unconvertible() {
        assert_eq!(conv(UNCONVERTIBLE), None);
        let p = r#"{"op":"is_null","args":[{"column":["tags"]}]}"#;
        assert_eq!(conv(p), None);
        let p = r#"{"op":"is_null","args":[{"column":["nested"]}]}"#;
        assert_eq!(conv(p), None);
    }

    #[test]
    fn null_checks_convert() {
        let p = r#"{"op":"is_not_null","args":[{"column":["name"]}]}"#;
        assert_eq!(conv(p).unwrap(), "NOT(Column(name) IS NULL)");
    }

    #[test]
    fn inexact_float_literal_is_refused() {
        // f32(1.1) > 1.1, so `f > 1.1` rewritten as `f > f32(1.1)` would skip
        // a file whose max is exactly f32(1.1).
        let p = r#"{"op":"gt","args":[{"column":["f"]},{"literal":1.1,"type":"double"}]}"#;
        assert_eq!(conv(p), None);
        let p = r#"{"op":"gt","args":[{"column":["f"]},{"literal":1.5,"type":"double"}]}"#;
        assert!(conv(p).is_some());
    }

    #[test]
    fn fractional_literal_against_integer_column_is_refused() {
        let p = r#"{"op":"gt","args":[{"column":["id"]},{"literal":1.5,"type":"double"}]}"#;
        assert_eq!(conv(p), None);
    }

    #[test]
    fn date_timestamp_and_decimal_literals_coerce() {
        let p = r#"{"op":"eq","args":[{"column":["day"]},{"literal":"2024-01-02","type":"date"}]}"#;
        assert!(conv(p).is_some());
        let p = r#"{"op":"lt","args":[{"column":["ts"]},{"literal":"2024-01-02T03:04:05Z","type":"timestamp"}]}"#;
        assert!(conv(p).is_some());
        let p = r#"{"op":"lt","args":[{"column":["ts"]},{"literal":1700000000000000,"type":"timestamp"}]}"#;
        assert!(conv(p).is_some());
        let p = r#"{"op":"lt","args":[{"column":["ntz"]},{"literal":"2024-01-02T03:04:05","type":"timestamp_ntz"}]}"#;
        assert!(conv(p).is_some());
        let p = r#"{"op":"lt","args":[{"column":["ntz"]},{"literal":"2024-01-02T03:04:05Z","type":"timestamp"}]}"#;
        assert_eq!(conv(p), None);
        let p = r#"{"op":"lt","args":[{"column":["amount"]},{"literal":"1.5","type":"decimal"}]}"#;
        assert!(conv(p).is_some());
        let p =
            r#"{"op":"lt","args":[{"column":["amount"]},{"literal":"1.555","type":"decimal"}]}"#;
        assert_eq!(conv(p), None);
    }

    #[test]
    fn null_literal_and_constant_comparisons_are_unconvertible() {
        let p = r#"{"op":"eq","args":[{"column":["id"]},{"literal":null,"type":"null"}]}"#;
        assert_eq!(conv(p), None);
        let p = r#"{"op":"eq","args":[{"literal":1,"type":"long"},{"literal":1,"type":"long"}]}"#;
        assert_eq!(conv(p), None);
    }

    #[test]
    fn empty_string_literal_stays_an_empty_string() {
        let p = r#"{"op":"eq","args":[{"column":["name"]},{"literal":"","type":"string"}]}"#;
        assert!(conv(p).unwrap().contains("= ''") || conv(p).unwrap().contains("= \"\""));
    }

    #[test]
    fn decimal_normalization() {
        assert_eq!(normalize_decimal("1.5", 2).as_deref(), Some("1.50"));
        assert_eq!(normalize_decimal("-3", 2).as_deref(), Some("-3.00"));
        assert_eq!(normalize_decimal("1.500", 2).as_deref(), Some("1.50"));
        assert_eq!(normalize_decimal("1.505", 2), None);
        assert_eq!(normalize_decimal("7.0", 0).as_deref(), Some("7"));
        assert_eq!(normalize_decimal("1e3", 0), None);
    }
}
