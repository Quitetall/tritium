//! ADR 0036 L2 quality gate for the opt-in int8 LM-head rung
//! (`TRITIUM_LM_HEAD=i8`): teacher-forced perplexity on WT-103 windows must be
//! within **1.001×** of the default f16 head on the SAME windows (the L2 bar;
//! the rung is explicitly not token-identical — OPTIMIZATION-LOG round-25
//! addendum records group-64 i8 at +0.067% ppl, 99.27% argmax).
//!
//! Run (quiet 4090 box, real 2B4T bundle):
//! ```sh
//! cargo test -p tritium-nn --features cuda --release --test lm_head_ppl \
//!   -- --ignored --nocapture --test-threads=1
//! ```
//!
//! Corpus: `~/blut/data/corpus_wt103.jsonl` (override with `TRITIUM_CORPUS`),
//! `{"tokens": [...]}` per line. 50 windows × 103 tokens = 5100 scored
//! positions — the same budget the round-25 grading used.

static GGUF_PATH: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
    let dir = std::env::var("TRITIUM_MODEL_DIR").unwrap_or_else(|_| {
        format!(
            "{}/.cache/tritium-models",
            std::env::var("HOME").unwrap_or_default()
        )
    });
    format!("{dir}/bitnet-2b4t-gguf/ggml-model-i2_s.gguf")
});

static CORPUS_PATH: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
    std::env::var("TRITIUM_CORPUS").unwrap_or_else(|_| {
        format!(
            "{}/blut/data/corpus_wt103.jsonl",
            std::env::var("HOME").unwrap_or_default()
        )
    })
});

const SEQ_LEN: usize = 103;
const WINDOWS: usize = 50;

struct HeadEnvGuard(Option<std::ffi::OsString>);
impl HeadEnvGuard {
    fn set(rung: &str) -> Self {
        let prev = std::env::var_os("TRITIUM_LM_HEAD");
        // SAFETY: this ignored gate runs single-threaded (--test-threads=1 per
        // the header command); no other thread touches the environment.
        #[allow(unsafe_code)]
        unsafe {
            std::env::set_var("TRITIUM_LM_HEAD", rung);
        }
        Self(prev)
    }
}
impl Drop for HeadEnvGuard {
    fn drop(&mut self) {
        // SAFETY: single-threaded gate (see `HeadEnvGuard::set`).
        #[allow(unsafe_code)]
        unsafe {
            match self.0.take() {
                Some(v) => std::env::set_var("TRITIUM_LM_HEAD", v),
                None => std::env::remove_var("TRITIUM_LM_HEAD"),
            }
        }
    }
}

/// Deterministic window selection: first `WINDOWS` corpus docs with at least
/// `SEQ_LEN` tokens contribute their leading `SEQ_LEN` tokens each.
fn corpus_windows() -> Option<Vec<u32>> {
    let text = match std::fs::read_to_string(&*CORPUS_PATH) {
        Ok(t) => t,
        Err(e) => {
            eprintln!(
                "skipping lm_head ppl gate: no corpus at {} ({e})",
                *CORPUS_PATH
            );
            return None;
        }
    };
    let mut out = Vec::with_capacity(WINDOWS * SEQ_LEN);
    let mut taken = 0usize;
    for line in text.lines() {
        if taken == WINDOWS {
            break;
        }
        // Minimal parse of {"tokens":[...]}: avoid a serde dependency here.
        let Some(start) = line.find('[') else {
            continue;
        };
        let Some(end) = line.rfind(']') else { continue };
        let ids: Vec<u32> = line[start + 1..end]
            .split(',')
            .filter_map(|t| t.trim().parse().ok())
            .collect();
        if ids.len() >= SEQ_LEN {
            out.extend_from_slice(&ids[..SEQ_LEN]);
            taken += 1;
        }
    }
    if taken < WINDOWS {
        eprintln!("skipping lm_head ppl gate: only {taken}/{WINDOWS} usable docs");
        return None;
    }
    Some(out)
}

fn score(rung: &'static str, bytes: &[u8], ids: &[u32]) -> Option<f64> {
    let _env = HeadEnvGuard::set(rung);
    let init = tritium_runtime::BACKENDS
        .iter()
        .find(|e| e.name == "cuda")?
        .init;
    let backend = match init() {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping lm_head ppl gate: cuda init failed ({e})");
            return None;
        }
    };
    let file = tritium_format::read_gguf(bytes).expect("parse gguf");
    let mut runner = tritium_nn::ModelRunner::load(&file, bytes, backend).expect("load model");
    let r = tritium_nn::teacher_forced_perplexity_windows(&mut runner, ids, SEQ_LEN)
        .expect("teacher-forced scoring");
    eprintln!(
        "lm_head={rung}: ppl {:.6} over {} positions ({} windows x {})",
        r.perplexity,
        r.token_count,
        r.window_count,
        SEQ_LEN - 1
    );
    Some(r.perplexity)
}

#[test]
#[ignore = "real-model quality gate: needs the 2B4T bundle + WT-103 corpus + a CUDA device"]
fn lm_head_i8_ppl_within_1_001x_of_f16() {
    let bytes = match std::fs::read(&*GGUF_PATH) {
        Ok(b) => b,
        Err(e) => {
            eprintln!(
                "skipping lm_head ppl gate: no model at {} ({e})",
                *GGUF_PATH
            );
            return;
        }
    };
    let Some(ids) = corpus_windows() else { return };
    let Some(ppl_f16) = score("f16", &bytes, &ids) else {
        return;
    };
    let Some(ppl_i8) = score("i8", &bytes, &ids) else {
        return;
    };
    let ratio = ppl_i8 / ppl_f16;
    eprintln!(
        "lm_head i8/f16 ppl ratio: {ratio:.6} (delta {:+.4}%; gate <= 1.001)",
        (ratio - 1.0) * 100.0
    );
    assert!(
        ratio <= 1.001,
        "i8 LM-head ppl ratio {ratio:.6} exceeds the ADR 0036 L2 bar (1.001x)"
    );
}
