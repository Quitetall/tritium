use std::{
    collections::{BTreeMap, BTreeSet},
    env,
    error::Error,
    fs,
    path::{Path, PathBuf},
};

use schemars::JsonSchema;
use serde_json::{Map, Value};
use tritium_schema::{
    AdditiveLayout, AdmittedLaw, Basis, BlobId, EvidenceEnvelope, LayoutError, ModelId, PackageId,
    PlaneAllocation, PlaneCodec, PlaneRelation, ScaleAnchor, ScaleLaw, ScalePrecision, SchemaId,
    SemanticTensorDigest, Transport, UnknownReason, Verdict,
};

fn main() -> Result<(), Box<dyn Error>> {
    let check = match env::args().nth(1).as_deref() {
        None => false,
        Some("--check") => true,
        Some(other) => return Err(format!("unknown argument {other:?}; expected --check").into()),
    };
    if env::args().nth(2).is_some() {
        return Err("accepts at most one argument (--check)".into());
    }

    let workspace = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let output_dir = workspace.join("schemas/json/v1");
    if !check {
        fs::create_dir_all(&output_dir)?;
    }

    macro_rules! project {
        ($type:ty, $name:literal) => {
            write_projection::<$type>(&output_dir, $name, check)?;
        };
    }

    project!(AdditiveLayout, "additive-layout");
    project!(AdmittedLaw, "admitted-law");
    project!(Basis, "basis");
    project!(BlobId, "blob-id");
    project!(EvidenceEnvelope<serde_json::Value>, "evidence-envelope");
    project!(LayoutError, "layout-error");
    project!(ModelId, "model-id");
    project!(PackageId, "package-id");
    project!(PlaneAllocation, "plane-allocation");
    project!(PlaneCodec, "plane-codec");
    project!(PlaneRelation, "plane-relation");
    project!(ScaleAnchor, "scale-anchor");
    project!(ScaleLaw, "scale-law");
    project!(ScalePrecision, "scale-precision");
    project!(SchemaId, "schema-id");
    project!(SemanticTensorDigest, "semantic-tensor-digest");
    project!(Transport, "transport");
    project!(UnknownReason, "unknown-reason");
    project!(Verdict, "verdict");

    let schemas = load_schemas(&output_dir)?;
    let typescript = render_typescript(&schemas).map_err(invalid_schema)?;
    let python = render_python(&schemas).map_err(invalid_schema)?;
    write_text_projection(
        &workspace.join("schemas/typescript/v1.d.ts"),
        &typescript,
        check,
    )?;
    write_text_projection(&workspace.join("schemas/python/v1.pyi"), &python, check)?;

    Ok(())
}

fn invalid_schema(message: String) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidData, message)
}

fn write_text_projection(path: &Path, rendered: &str, check: bool) -> Result<(), Box<dyn Error>> {
    if check {
        let existing = fs::read_to_string(path)?;
        if existing != rendered {
            return Err(format!(
                "{} is stale; rerun the projection generator",
                path.display()
            )
            .into());
        }
    } else {
        let parent = path.parent().ok_or("projection path has no parent")?;
        fs::create_dir_all(parent)?;
        fs::write(path, rendered)?;
    }
    Ok(())
}

#[derive(Debug)]
struct ProjectionSchema {
    name: String,
    schema: Value,
}

fn load_schemas(directory: &Path) -> Result<Vec<ProjectionSchema>, Box<dyn Error>> {
    let mut paths = fs::read_dir(directory)?
        .map(|entry| entry.map(|entry| entry.path()))
        .collect::<Result<Vec<_>, _>>()?;
    paths.retain(|path| {
        path.extension()
            .is_some_and(|extension| extension == "json")
    });
    paths.sort();

    let mut schemas = Vec::with_capacity(paths.len());
    for path in paths {
        let schema: Value = serde_json::from_slice(&fs::read(path)?)?;
        let name = schema
            .get("title")
            .and_then(Value::as_str)
            .ok_or("generated schema has no title")?
            .to_owned();
        schemas.push(ProjectionSchema { name, schema });
    }
    Ok(schemas)
}

fn render_typescript(schemas: &[ProjectionSchema]) -> Result<String, String> {
    let roots = schemas
        .iter()
        .map(|schema| (schema.name.clone(), root_definition(&schema.schema)))
        .collect::<BTreeMap<_, _>>();
    let definitions = collect_definitions(schemas, roots.keys())?;
    let mut output = String::from(
        "// Generated from tritium-schema Rust types; do not edit.\n\
         // JavaScript numbers cannot exactly represent every uint64 JSON integer.\n\
         // Use a lossless JSON parser when consuming values above Number.MAX_SAFE_INTEGER.\n\n",
    );

    for (name, schema) in roots.iter().chain(definitions.iter()) {
        output.push_str(&render_typescript_definition(name, schema)?);
        output.push('\n');
    }
    trim_blank_lines_at_eof(&mut output);
    Ok(output)
}

fn render_typescript_definition(name: &str, schema: &Value) -> Result<String, String> {
    if object_properties(schema).is_some() {
        return render_typescript_object(name, schema);
    }
    Ok(format!(
        "export type {name} = {};\n",
        typescript_type(schema, name)?
    ))
}

fn render_typescript_object(name: &str, schema: &Value) -> Result<String, String> {
    let properties = object_properties(schema).ok_or_else(|| format!("{name} is not an object"))?;
    let required = required_properties(schema);
    let mut output = format!("export interface {name} {{\n");
    for (field, value) in properties {
        let optional = if required.contains(field.as_str()) {
            ""
        } else {
            "?"
        };
        let field = typescript_property_name(field)?;
        output.push_str(&format!(
            "  readonly {field}{optional}: {};\n",
            typescript_type(value, name)?
        ));
    }
    if schema.get("additionalProperties") == Some(&Value::Bool(true)) {
        output.push_str("  readonly [key: string]: unknown;\n");
    }
    output.push_str("}\n");
    Ok(output)
}

fn typescript_type(schema: &Value, path: &str) -> Result<String, String> {
    if let Some(reference) = schema.get("$ref").and_then(Value::as_str) {
        return local_reference(reference);
    }
    if let Some(value) = schema.get("const") {
        return json_literal(value);
    }
    if let Some(values) = schema.get("enum").and_then(Value::as_array) {
        return join_types(
            values
                .iter()
                .map(json_literal)
                .collect::<Result<Vec<_>, _>>()?,
        );
    }
    for keyword in ["oneOf", "anyOf"] {
        if let Some(variants) = schema.get(keyword).and_then(Value::as_array) {
            return join_types(
                variants
                    .iter()
                    .enumerate()
                    .map(|(index, variant)| {
                        typescript_type(variant, &format!("{path}Variant{index}"))
                    })
                    .collect::<Result<Vec<_>, _>>()?,
            );
        }
    }
    if let Some(parts) = schema.get("allOf").and_then(Value::as_array) {
        return join_intersections(
            parts
                .iter()
                .map(|part| typescript_type(part, path))
                .collect::<Result<Vec<_>, _>>()?,
        );
    }
    if let Some(types) = schema.get("type").and_then(Value::as_array) {
        return join_types(
            types
                .iter()
                .map(|kind| {
                    let kind = kind
                        .as_str()
                        .ok_or_else(|| format!("{path} has a non-string schema type"))?;
                    typescript_primitive(schema, kind, path)
                })
                .collect::<Result<Vec<_>, _>>()?,
        );
    }
    let Some(kind) = schema.get("type").and_then(Value::as_str) else {
        if object_properties(schema).is_some() {
            return typescript_inline_object(schema, path);
        }
        return Ok("unknown".into());
    };
    typescript_primitive(schema, kind, path)
}

fn typescript_primitive(schema: &Value, kind: &str, path: &str) -> Result<String, String> {
    match kind {
        "string" => Ok("string".into()),
        "integer" | "number" => Ok("number".into()),
        "boolean" => Ok("boolean".into()),
        "null" => Ok("null".into()),
        "array" => {
            let item = schema
                .get("items")
                .map(|item| typescript_type(item, &format!("{path}Item")))
                .transpose()?
                .unwrap_or_else(|| "unknown".into());
            if let (Some(minimum), Some(maximum)) = (
                schema.get("minItems").and_then(Value::as_u64),
                schema.get("maxItems").and_then(Value::as_u64),
            ) && minimum == maximum
                && minimum <= 64
            {
                let tuple = std::iter::repeat_n(item, minimum as usize)
                    .collect::<Vec<_>>()
                    .join(", ");
                return Ok(format!("[{tuple}]"));
            }
            Ok(format!("ReadonlyArray<{item}>"))
        }
        "object" => {
            if object_properties(schema).is_some() {
                typescript_inline_object(schema, path)
            } else if let Some(value_schema) = schema
                .get("additionalProperties")
                .filter(|value| value.is_object())
            {
                Ok(format!(
                    "Readonly<Record<string, {}>>",
                    typescript_type(value_schema, &format!("{path}Value"))?
                ))
            } else {
                Ok("Readonly<Record<string, unknown>>".into())
            }
        }
        other => Err(format!("unsupported JSON Schema type {other:?} at {path}")),
    }
}

fn typescript_inline_object(schema: &Value, path: &str) -> Result<String, String> {
    let properties =
        object_properties(schema).ok_or_else(|| format!("{path} has no properties"))?;
    let required = required_properties(schema);
    let mut fields = Vec::new();
    for (field, value) in properties {
        let optional = if required.contains(field.as_str()) {
            ""
        } else {
            "?"
        };
        let field = typescript_property_name(field)?;
        fields.push(format!(
            "readonly {field}{optional}: {}",
            typescript_type(value, path)?
        ));
    }
    if schema.get("additionalProperties") == Some(&Value::Bool(true)) {
        fields.push("readonly [key: string]: unknown".into());
    }
    Ok(format!("{{ {} }}", fields.join("; ")))
}

fn render_python(schemas: &[ProjectionSchema]) -> Result<String, String> {
    let roots = schemas
        .iter()
        .map(|schema| (schema.name.clone(), root_definition(&schema.schema)))
        .collect::<BTreeMap<_, _>>();
    let definitions = collect_definitions(schemas, roots.keys())?;
    let mut helpers = BTreeMap::new();
    let mut aliases = BTreeMap::new();
    for (name, schema) in roots.iter().chain(definitions.iter()) {
        if object_properties(schema).is_some() {
            aliases.insert(
                name.clone(),
                render_python_object(name, schema, &mut helpers)?,
            );
        } else {
            aliases.insert(
                name.clone(),
                format!(
                    "{name}: TypeAlias = {}\n",
                    python_type(schema, name, &mut helpers)?
                ),
            );
        }
    }

    let mut output = String::from(
        "# Generated from tritium-schema Rust types; do not edit.\n\
         from __future__ import annotations\n\n\
         from typing import Any, Literal, Union\n\
         from typing_extensions import NotRequired, TypeAlias, TypedDict\n\n",
    );
    for code in helpers.values().chain(aliases.values()) {
        output.push_str(code);
        output.push('\n');
    }
    trim_blank_lines_at_eof(&mut output);
    Ok(output)
}

fn trim_blank_lines_at_eof(output: &mut String) {
    while output.ends_with("\n\n") {
        output.pop();
    }
}

fn render_python_object(
    name: &str,
    schema: &Value,
    helpers: &mut BTreeMap<String, String>,
) -> Result<String, String> {
    let properties = object_properties(schema).ok_or_else(|| format!("{name} is not an object"))?;
    let required = required_properties(schema);
    let mut output = format!("class {name}(TypedDict):\n");
    if properties.is_empty() {
        output.push_str("    pass\n");
    }
    for (field, value) in properties {
        if !is_python_identifier(field) {
            return Err(format!(
                "{name} has unsupported Python field name {field:?}"
            ));
        }
        let field_type = python_type(value, &format!("{name}{}", pascal_case(field)), helpers)?;
        let field_type = if required.contains(field.as_str()) {
            field_type
        } else {
            format!("NotRequired[{field_type}]")
        };
        output.push_str(&format!("    {field}: {field_type}\n"));
    }
    if schema.get("additionalProperties") == Some(&Value::Bool(true)) {
        output.push_str("    __extra__: NotRequired[dict[str, Any]]\n");
    }
    Ok(output)
}

fn python_type(
    schema: &Value,
    path: &str,
    helpers: &mut BTreeMap<String, String>,
) -> Result<String, String> {
    if let Some(reference) = schema.get("$ref").and_then(Value::as_str) {
        return local_reference(reference);
    }
    if let Some(value) = schema.get("const") {
        return Ok(format!("Literal[{}]", python_literal(value)?));
    }
    if let Some(values) = schema.get("enum").and_then(Value::as_array) {
        let literals = values
            .iter()
            .map(python_literal)
            .map(|value| value.map(|value| format!("Literal[{value}]")))
            .collect::<Result<Vec<_>, _>>()?;
        return join_python_types(literals);
    }
    for keyword in ["oneOf", "anyOf"] {
        if let Some(variants) = schema.get(keyword).and_then(Value::as_array) {
            return join_python_types(
                variants
                    .iter()
                    .enumerate()
                    .map(|(index, variant)| {
                        python_type(variant, &format!("{path}Variant{index}"), helpers)
                    })
                    .collect::<Result<Vec<_>, _>>()?,
            );
        }
    }
    if let Some(parts) = schema.get("allOf").and_then(Value::as_array) {
        return join_python_types(
            parts
                .iter()
                .map(|part| python_type(part, path, helpers))
                .collect::<Result<Vec<_>, _>>()?,
        );
    }
    if let Some(types) = schema.get("type").and_then(Value::as_array) {
        return join_python_types(
            types
                .iter()
                .map(|kind| {
                    let kind = kind
                        .as_str()
                        .ok_or_else(|| format!("{path} has a non-string schema type"))?;
                    python_primitive(schema, kind, path, helpers)
                })
                .collect::<Result<Vec<_>, _>>()?,
        );
    }
    let Some(kind) = schema.get("type").and_then(Value::as_str) else {
        if object_properties(schema).is_some() {
            let name = pascal_case(path);
            let rendered = render_python_object(&name, schema, helpers)?;
            insert_helper(helpers, name.clone(), rendered)?;
            return Ok(name);
        }
        return Ok("Any".into());
    };
    python_primitive(schema, kind, path, helpers)
}

fn python_primitive(
    schema: &Value,
    kind: &str,
    path: &str,
    helpers: &mut BTreeMap<String, String>,
) -> Result<String, String> {
    match kind {
        "string" => Ok("str".into()),
        "integer" => Ok("int".into()),
        "number" => Ok("float".into()),
        "boolean" => Ok("bool".into()),
        "null" => Ok("None".into()),
        "array" => {
            let item = schema
                .get("items")
                .map(|item| python_type(item, &format!("{path}Item"), helpers))
                .transpose()?
                .unwrap_or_else(|| "Any".into());
            if let (Some(minimum), Some(maximum)) = (
                schema.get("minItems").and_then(Value::as_u64),
                schema.get("maxItems").and_then(Value::as_u64),
            ) && minimum == maximum
                && minimum <= 64
            {
                let tuple = std::iter::repeat_n(item, minimum as usize)
                    .collect::<Vec<_>>()
                    .join(", ");
                return Ok(format!("tuple[{tuple}]"));
            }
            Ok(format!("list[{item}]"))
        }
        "object" => {
            if object_properties(schema).is_some() {
                let name = pascal_case(path);
                let rendered = render_python_object(&name, schema, helpers)?;
                insert_helper(helpers, name.clone(), rendered)?;
                Ok(name)
            } else if let Some(value_schema) = schema
                .get("additionalProperties")
                .filter(|value| value.is_object())
            {
                Ok(format!(
                    "dict[str, {}]",
                    python_type(value_schema, &format!("{path}Value"), helpers)?
                ))
            } else {
                Ok("dict[str, Any]".into())
            }
        }
        other => Err(format!("unsupported JSON Schema type {other:?} at {path}")),
    }
}

fn insert_helper(
    helpers: &mut BTreeMap<String, String>,
    name: String,
    code: String,
) -> Result<(), String> {
    if helpers.get(&name).is_some_and(|existing| existing != &code) {
        return Err(format!("generated helper name collision for {name}"));
    }
    helpers.insert(name, code);
    Ok(())
}

fn collect_definitions<'a>(
    schemas: &'a [ProjectionSchema],
    root_names: impl Iterator<Item = &'a String>,
) -> Result<BTreeMap<String, Value>, String> {
    let root_names = root_names.cloned().collect::<BTreeSet<_>>();
    let mut definitions = BTreeMap::new();
    for schema in schemas {
        let Some(items) = schema.schema.get("$defs").and_then(Value::as_object) else {
            continue;
        };
        for (name, definition) in items {
            if root_names.contains(name) {
                continue;
            }
            if definitions
                .get(name)
                .is_some_and(|existing| existing != definition)
            {
                return Err(format!("generated schemas disagree on definition {name}"));
            }
            definitions.insert(name.clone(), definition.clone());
        }
    }
    Ok(definitions)
}

fn root_definition(schema: &Value) -> Value {
    let mut root = schema.clone();
    if let Some(object) = root.as_object_mut() {
        object.remove("$defs");
    }
    root
}

fn object_properties(schema: &Value) -> Option<&Map<String, Value>> {
    schema.get("properties").and_then(Value::as_object)
}

fn required_properties(schema: &Value) -> BTreeSet<String> {
    schema
        .get("required")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .map(str::to_owned)
        .collect()
}

fn local_reference(reference: &str) -> Result<String, String> {
    let name = reference
        .strip_prefix("#/$defs/")
        .ok_or_else(|| format!("unsupported external JSON Schema reference {reference:?}"))?;
    Ok(name.replace("~1", "/").replace("~0", "~"))
}

fn typescript_property_name(value: &str) -> Result<String, String> {
    let mut chars = value.chars();
    if chars
        .next()
        .is_some_and(|first| first == '_' || first == '$' || first.is_ascii_alphabetic())
        && chars.all(|character| {
            character == '_' || character == '$' || character.is_ascii_alphanumeric()
        })
    {
        Ok(value.to_owned())
    } else {
        serde_json::to_string(value).map_err(|error| error.to_string())
    }
}

fn json_literal(value: &Value) -> Result<String, String> {
    match value {
        Value::String(_) | Value::Number(_) | Value::Bool(_) | Value::Null => {
            serde_json::to_string(value).map_err(|error| error.to_string())
        }
        _ => Err("only scalar JSON Schema constants are supported".into()),
    }
}

fn python_literal(value: &Value) -> Result<String, String> {
    match value {
        Value::String(value) => Ok(format!("{value:?}")),
        Value::Number(value) => Ok(value.to_string()),
        Value::Bool(true) => Ok("True".into()),
        Value::Bool(false) => Ok("False".into()),
        Value::Null => Ok("None".into()),
        _ => Err("only scalar JSON Schema constants are supported".into()),
    }
}

fn join_types(types: Vec<String>) -> Result<String, String> {
    let types = types.into_iter().collect::<BTreeSet<_>>();
    if types.is_empty() {
        return Err("schema union has no variants".into());
    }
    Ok(types.into_iter().collect::<Vec<_>>().join(" | "))
}

fn join_intersections(types: Vec<String>) -> Result<String, String> {
    if types.is_empty() {
        return Err("schema intersection has no members".into());
    }
    Ok(types.join(" & "))
}

fn join_python_types(types: Vec<String>) -> Result<String, String> {
    let types = types.into_iter().collect::<BTreeSet<_>>();
    if types.is_empty() {
        return Err("schema union has no variants".into());
    }
    let types = types.into_iter().collect::<Vec<_>>();
    Ok(if types.len() == 1 {
        types[0].clone()
    } else {
        format!("Union[{}]", types.join(", "))
    })
}

fn pascal_case(value: &str) -> String {
    value
        .split(|character: char| !character.is_ascii_alphanumeric())
        .filter(|part| !part.is_empty())
        .map(|part| {
            let mut chars = part.chars();
            chars
                .next()
                .map(|first| first.to_ascii_uppercase().to_string() + chars.as_str())
                .unwrap_or_default()
        })
        .collect()
}

fn is_python_identifier(value: &str) -> bool {
    let mut chars = value.chars();
    chars
        .next()
        .is_some_and(|first| first == '_' || first.is_ascii_alphabetic())
        && chars.all(|character| character == '_' || character.is_ascii_alphanumeric())
}

fn write_projection<T: JsonSchema>(
    directory: &Path,
    name: &str,
    check: bool,
) -> Result<(), Box<dyn Error>> {
    let path = directory.join(format!("{name}.schema.json"));
    let schema = schemars::schema_for!(T);
    let rendered = format!("{}\n", serde_json::to_string_pretty(&schema)?);
    if check {
        let existing = fs::read_to_string(&path)?;
        if existing != rendered {
            return Err(format!(
                "{} is stale; rerun the projection generator",
                path.display()
            )
            .into());
        }
    } else {
        fs::write(&path, rendered)?;
    }
    Ok(())
}
