use std::{env, error::Error, fs, path::PathBuf};

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
    let output = workspace.join("schemas/cddl/v1-byte-layout.md");
    let rendered = tritium_format::trit_package_layout_spec();
    if check {
        let existing = fs::read_to_string(&output)?;
        if existing != rendered {
            return Err(format!(
                "{} is stale; rerun tritium-format-projections",
                output.display()
            )
            .into());
        }
    } else {
        let parent = output.parent().ok_or("projection path has no parent")?;
        fs::create_dir_all(parent)?;
        fs::write(output, rendered)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::{fs, path::PathBuf};

    #[test]
    fn byte_layout_projection_is_current() {
        let workspace = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
        let projection = fs::read_to_string(workspace.join("schemas/cddl/v1-byte-layout.md"))
            .expect("generated .trit byte-layout projection exists");
        assert_eq!(
            projection,
            tritium_format::trit_package_layout_spec(),
            "generated .trit byte-layout projection is stale"
        );
    }
}
