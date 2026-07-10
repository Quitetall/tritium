//! `tritium pull <repo>` — fetch a GGUF from the HuggingFace hub into the
//! local model cache, with resume.
//!
//! Zero-config path from "I have a repo name" to "I can serve it":
//!
//! ```text
//! tritium pull microsoft/bitnet-b1.58-2B-4T-gguf
//! tritium-serve --model ~/.cache/tritium-models/microsoft--bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf
//! ```
//!
//! The tokenizer travels INSIDE the GGUF (`tokenizer.ggml.*` metadata — see
//! `tritium_nn::GgufBpeTokenizer`), so one file is the whole model. Downloads
//! land as `<file>.part` and resume via HTTP `Range` on re-run; the final
//! rename is atomic, so a `.gguf` in the cache is always complete.

use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, bail};

/// One file entry in the hub's model-info response.
#[derive(serde::Deserialize)]
struct Sibling {
    rfilename: String,
}

/// The subset of `https://huggingface.co/api/models/{repo}` we consume.
#[derive(serde::Deserialize)]
struct ModelInfo {
    siblings: Vec<Sibling>,
}

/// Default cache root (shared with the docs' quickstart examples).
fn cache_root() -> PathBuf {
    if let Ok(dir) = std::env::var("TRITIUM_MODEL_CACHE") {
        return PathBuf::from(dir);
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    Path::new(&home).join(".cache/tritium-models")
}

/// Run `tritium pull`.
pub(crate) fn run(repo: &str, file: Option<&str>, revision: &str) -> anyhow::Result<()> {
    if repo.split('/').count() != 2 || repo.contains("..") {
        bail!("repo must look like `owner/name`, got {repo:?}");
    }

    // Which file? Ask the hub unless --file pinned one.
    let filename = match file {
        Some(f) => f.to_owned(),
        None => {
            let url = format!("https://huggingface.co/api/models/{repo}");
            let mut req = ureq::get(&url);
            // Gated repos need the token on the LISTING too, or pull dies on
            // an opaque 401 before ever reaching the download path's hint.
            if let Ok(token) = std::env::var("HF_TOKEN")
                && !token.is_empty()
            {
                req = req.header("Authorization", format!("Bearer {token}"));
            }
            let info: ModelInfo = req
                .call()
                .with_context(|| format!("querying {url}"))?
                .body_mut()
                .read_json()
                .context("parsing hub model info")?;
            let mut ggufs: Vec<String> = info
                .siblings
                .into_iter()
                .map(|s| s.rfilename)
                .filter(|f| f.ends_with(".gguf"))
                .collect();
            ggufs.sort();
            match ggufs.len() {
                0 => bail!(
                    "{repo} has no .gguf files — tritium serves GGUF; pick a \
                     -gguf repo (or convert and use --file)"
                ),
                1 => ggufs.remove(0),
                _ => {
                    eprintln!("{repo} has {} .gguf files:", ggufs.len());
                    for f in &ggufs {
                        eprintln!("  {f}");
                    }
                    bail!("pick one with --file <name>");
                }
            }
        }
    };

    let dir = cache_root().join(repo.replace('/', "--"));
    std::fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;
    let dest = dir.join(
        Path::new(&filename)
            .file_name()
            .context("bad --file name")?,
    );
    if dest.exists() {
        eprintln!("already present: {}", dest.display());
        print_next_steps(&dest);
        return Ok(());
    }

    let url = format!("https://huggingface.co/{repo}/resolve/{revision}/{filename}");
    // NOT with_extension (it would replace a non-.gguf final extension and
    // collide same-stem files): always append.
    let file_name = dest.file_name().and_then(|n| n.to_str()).unwrap_or("model");
    let part = dir.join(format!("{file_name}.part"));
    // Sidecar recording the upstream validator: "<etag> <total_bytes>".
    // Resume without it is a splice hazard (old-file prefix + new-file
    // tail both parse as GGUF and serve garbage), so no meta = restart.
    let meta_path = dir.join(format!("{file_name}.part.meta"));
    let meta: Option<(String, u64)> = std::fs::read_to_string(&meta_path)
        .ok()
        .and_then(|m| {
            let (etag, total) = m.trim().rsplit_once(' ')?;
            Some((etag.to_owned(), total.parse().ok()?))
        });
    let have = match (&meta, part.metadata().map(|m| m.len())) {
        (Some(_), Ok(len)) => len,
        _ => 0, // no validator (or no .part): restart from scratch
    };
    if let Some((_, total)) = &meta
        && have == *total
        && have > 0
    {
        // Complete but unpromoted (died between EOF and rename): promote.
        std::fs::rename(&part, &dest)
            .with_context(|| format!("renaming into {}", dest.display()))?;
        let _ = std::fs::remove_file(&meta_path);
        eprintln!("pulled {} (recovered complete download)", dest.display());
        print_next_steps(&dest);
        return Ok(());
    }

    eprintln!("pulling {url}");
    let mut req = ureq::get(&url)
        .config()
        // Handle 416 ourselves instead of erroring in .call().
        .http_status_as_error(false)
        .build();
    // Gated repos: HF_TOKEN if present. ureq strips Authorization on
    // redirect, so the token never leaks to the CDN host (HF's design).
    if let Ok(token) = std::env::var("HF_TOKEN")
        && !token.is_empty()
    {
        req = req.header("Authorization", format!("Bearer {token}"));
    }
    if have > 0
        && let Some((etag, _)) = &meta
    {
        eprintln!("resuming at {} MiB", have / (1024 * 1024));
        req = req
            .header("Range", format!("bytes={have}-"))
            // If upstream changed since the .part was written, the server
            // answers 200 with the full new file (never splices).
            .header("If-Range", etag);
    }
    let mut resp = req.call().with_context(|| format!("GET {url}"))?;
    let status = resp.status();
    let etag = resp
        .headers()
        .get("ETag")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("-")
        .to_owned();
    // 206 = validated resume; 200 = fresh start or upstream changed →
    // truncate; 416 with a matching validator = nothing left to fetch.
    let (mut out, mut done) = if status == 206 && have > 0 {
        let f = std::fs::OpenOptions::new()
            .append(true)
            .open(&part)
            .with_context(|| format!("opening {}", part.display()))?;
        (f, have)
    } else if status == 200 {
        let f = std::fs::File::create(&part)
            .with_context(|| format!("creating {}", part.display()))?;
        (f, 0)
    } else if status == 416 {
        // If-Range guarantees a changed file comes back as 200, so a 416
        // means our range start >= total for the SAME file — stale state
        // beyond recovery; clear it and ask for a re-run.
        let _ = std::fs::remove_file(&part);
        let _ = std::fs::remove_file(&meta_path);
        bail!("range not satisfiable — cleared stale partial state, re-run to restart");
    } else {
        bail!("GET {url} -> HTTP {status} (gated repo? set HF_TOKEN)");
    };
    let total = resp
        .headers()
        .get("Content-Length")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse::<u64>().ok())
        .map(|len| len + done);
    if let Some(total) = total {
        std::fs::write(&meta_path, format!("{etag} {total}"))
            .with_context(|| format!("writing {}", meta_path.display()))?;
    } else {
        // No Content-Length: no validator worth keeping — a stale sidecar
        // from a previous attempt must not authorize a future resume.
        let _ = std::fs::remove_file(&meta_path);
    }

    let mut reader = resp.body_mut().with_config().limit(u64::MAX).reader();
    let mut buf = vec![0u8; 1 << 20];
    let mut last_pct = u64::MAX;
    loop {
        let n = reader.read(&mut buf).context("network read")?;
        if n == 0 {
            break;
        }
        out.write_all(&buf[..n]).context("cache write")?;
        done += n as u64;
        if let Some(total) = total {
            let pct = done * 100 / total.max(1);
            if pct != last_pct {
                eprint!("\r{pct:>3}%  {} / {} MiB", done >> 20, total >> 20);
                last_pct = pct;
            }
        } else if done.trailing_zeros() >= 24 {
            eprint!("\r{} MiB", done >> 20);
        }
    }
    eprintln!();
    out.flush().ok();
    drop(out);
    if let Some(total) = total
        && done != total
    {
        bail!(
            "short read: got {done} of {total} bytes — re-run to resume \
             ({} kept)",
            part.display()
        );
    }
    std::fs::rename(&part, &dest)
        .with_context(|| format!("renaming into {}", dest.display()))?;
    let _ = std::fs::remove_file(&meta_path);
    eprintln!("pulled {}", dest.display());
    print_next_steps(&dest);
    Ok(())
}

fn print_next_steps(dest: &Path) {
    eprintln!("\nnext:");
    eprintln!("  tritium inspect {}", dest.display());
    eprintln!(
        "  tritium-serve --model {} --backend cuda   # text chat on /v1/chat/completions",
        dest.display()
    );
}
