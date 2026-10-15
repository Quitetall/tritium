//! Model-free request admission at the actual HTTP and source-tokenizer seams.
#![cfg(feature = "serve")]
use axum::{
    Router,
    body::Body,
    http::{Request, StatusCode},
};
use http_body_util::BodyExt;
use serde_json::{Value, json};
use std::{
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
    time::Duration,
};
use tower::ServiceExt;
use tritium_nn::{ChatRenderOptions, HfJsonTokenizer, NnError, Tokenizer};
use tritium_serve::{
    GenError, GenRequest, Generator, IdPassthroughTokenizer, MockGenerator, RequestLimits,
    ServeConfig, Step, build_router, build_router_with_limits,
};

const TEMPLATE: &str = "{% if enable_thinking is not defined %}default{% elif enable_thinking %}enabled{% else %}disabled{% endif %}{% for m in messages %} {{ m.content }}{% endfor %}";
type Prompts = Arc<Mutex<Vec<Vec<u32>>>>;
struct RecordedGenerator(Prompts, MockGenerator);
impl Generator for RecordedGenerator {
    fn generate(
        &mut self,
        req: &GenRequest,
        step: &mut dyn FnMut(Step) -> bool,
    ) -> Result<(), GenError> {
        self.0.lock().unwrap().push(req.prompt_tokens.clone());
        self.1.generate(req, step)
    }
    fn n_ctx(&self) -> usize {
        64
    }
    fn vocab(&self) -> usize {
        16
    }
}
struct CountedTokenizer<T>(T, Arc<AtomicUsize>);
impl<T: Tokenizer> Tokenizer for CountedTokenizer<T> {
    fn render_chat_prompt(&self, m: &[(&str, &str)], n: usize) -> Result<Option<String>, NnError> {
        self.0.render_chat_prompt(m, n)
    }
    fn render_chat_prompt_with_options(
        &self,
        m: &[(&str, &str)],
        o: ChatRenderOptions,
        n: usize,
    ) -> Result<Option<String>, NnError> {
        self.0.render_chat_prompt_with_options(m, o, n)
    }
    fn encode(&self, t: &str) -> Result<Vec<u32>, NnError> {
        self.1.fetch_add(1, Ordering::SeqCst);
        self.0.encode(t)
    }
    fn decode(&self, t: &[u32]) -> Result<String, NnError> {
        self.0.decode(t)
    }
    fn bos(&self) -> u32 {
        self.0.bos()
    }
    fn eos(&self) -> u32 {
        self.0.eos()
    }
}
fn source(template: Option<&str>) -> HfJsonTokenizer {
    use tokenizers::{
        Tokenizer as Inner, models::wordlevel::WordLevel, pre_tokenizers::whitespace::Whitespace,
    };
    let words = [
        "[UNK]", "eos", "default", "enabled", "disabled", "Hi", "Hello", "Next", "out",
    ];
    let model = WordLevel::builder()
        .vocab(
            words
                .into_iter()
                .enumerate()
                .map(|(i, w)| (w.to_owned(), i as u32))
                .collect(),
        )
        .unk_token("[UNK]".into())
        .build()
        .unwrap();
    let mut tokenizer = Inner::new(model);
    tokenizer.with_pre_tokenizer(Some(Whitespace));
    HfJsonTokenizer::from_bytes(
        tokenizer.to_string(false).unwrap().as_bytes(),
        &serde_json::to_vec(&json!({"eos_token":"eos","chat_template":template})).unwrap(),
    )
    .unwrap()
}
fn router<T: Tokenizer + Send + Sync + 'static>(
    tokenizer: T,
    limits: Option<RequestLimits>,
) -> (Router, Prompts, Arc<AtomicUsize>) {
    let prompts = Arc::new(Mutex::new(Vec::new()));
    let encoded = Arc::new(AtomicUsize::new(0));
    let generator = Box::new(RecordedGenerator(
        prompts.clone(),
        MockGenerator::new(vec![8]),
    ));
    let tokenizer = Arc::new(CountedTokenizer(tokenizer, encoded.clone()));
    let (router, _) = match limits {
        Some(limits) => {
            build_router_with_limits(generator, tokenizer, ServeConfig::default(), limits)
        }
        None => build_router(generator, tokenizer, ServeConfig::default()),
    };
    (router, prompts, encoded)
}
async fn request(router: &Router, body: String) -> (StatusCode, String, String) {
    tokio::time::timeout(Duration::from_secs(5), async {
        let response = router
            .clone()
            .oneshot(
                Request::post("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        let status = response.status();
        let content_type = response.headers()["content-type"]
            .to_str()
            .unwrap()
            .to_owned();
        let bytes = response.into_body().collect().await.unwrap().to_bytes();
        (
            status,
            content_type,
            String::from_utf8(bytes.to_vec()).unwrap(),
        )
    })
    .await
    .expect("bounded in-process chat request")
}
fn body(stream: bool) -> Value {
    json!({"model":"default","messages":[{"role":"user","content":"Hi"},{"role":"assistant","content":"Hello"},{"role":"user","content":"Next"}],"max_tokens":1,"stream":stream})
}

#[tokio::test]
async fn actual_hf_controls_reach_buffered_and_sse_generation_without_request_leakage() {
    let (router, prompts, encoded) = router(source(Some(TEMPLATE)), None);
    let mut expected = Vec::new();
    for stream in [false, true] {
        for (kwargs, prefix) in [
            (None, 2),
            (Some(Value::Null), 2),
            (Some(json!({})), 2),
            (Some(json!({"enable_thinking":false})), 4),
            (Some(json!({"enable_thinking":true})), 3),
            (None, 2),
        ] {
            let mut input = body(stream);
            if let Some(kwargs) = kwargs {
                input["chat_template_kwargs"] = kwargs;
            }
            let (status, content_type, text) = request(&router, input.to_string()).await;
            assert_eq!(status, StatusCode::OK, "{text}");
            if stream {
                assert!(content_type.starts_with("text/event-stream"));
                assert!(text.contains("data: [DONE]"));
                assert!(text.contains("\"content\":\"out\""), "{text}");
            } else {
                assert_eq!(
                    serde_json::from_str::<Value>(&text).unwrap()["choices"][0]["message"]["content"],
                    "out"
                );
            }
            expected.push(vec![prefix, 5, 6, 7]);
        }
    }
    assert_eq!(*prompts.lock().unwrap(), expected);
    assert_eq!(encoded.load(Ordering::SeqCst), 12);
}

#[tokio::test]
async fn malformed_control_objects_fail_as_json_before_encoding_or_dispatch() {
    let (router, prompts, encoded) = router(IdPassthroughTokenizer::default(), None);
    for stream in [false, true] {
        for kwargs in [
            json!(true),
            json!([]),
            json!({"enable_thinking":null}),
            json!({"enable_thinking":"false"}),
            json!({"enable_thinking":0}),
            json!({"messages":[]}),
            json!({"enable_thinking":false,"tools":[]}),
        ] {
            let mut input = body(stream);
            input["chat_template_kwargs"] = kwargs;
            let (status, content_type, text) = request(&router, input.to_string()).await;
            assert_eq!(status, StatusCode::BAD_REQUEST, "{text}");
            assert!(content_type.starts_with("application/json"));
            assert_eq!(
                serde_json::from_str::<Value>(&text).unwrap()["error"]["type"],
                "invalid_request_error"
            );
        }
    }
    let duplicate = r#"{"model":"default","messages":[{"role":"user","content":"Hi"}],"chat_template_kwargs":{"enable_thinking":false,"enable_thinking":true}}"#;
    assert_eq!(
        request(&router, duplicate.into()).await.0,
        StatusCode::BAD_REQUEST
    );
    assert_eq!(encoded.load(Ordering::SeqCst), 0);
    assert!(prompts.lock().unwrap().is_empty());
}

struct AbsentAdapter;
impl Tokenizer for AbsentAdapter {
    fn render_chat_prompt_with_options(
        &self,
        _: &[(&str, &str)],
        _: ChatRenderOptions,
        _: usize,
    ) -> Result<Option<String>, NnError> {
        Ok(None)
    }
    fn encode(&self, _: &str) -> Result<Vec<u32>, NnError> {
        panic!("absence must not trigger legacy encoding")
    }
    fn decode(&self, _: &[u32]) -> Result<String, NnError> {
        Ok("out".into())
    }
    fn bos(&self) -> u32 {
        0
    }
    fn eos(&self) -> u32 {
        1
    }
}
async fn refuses_before_encode<T: Tokenizer + Send + Sync + 'static>(tokenizer: T) {
    let (router, prompts, encoded) = router(tokenizer, None);
    for stream in [false, true] {
        for enabled in [false, true] {
            let mut input = body(stream);
            input["chat_template_kwargs"] = json!({"enable_thinking":enabled});
            let (status, content_type, text) = request(&router, input.to_string()).await;
            assert_eq!(status, StatusCode::BAD_REQUEST, "{text}");
            assert!(content_type.starts_with("application/json"));
            let error = serde_json::from_str::<Value>(&text).unwrap();
            assert_eq!(error["error"]["type"], "invalid_request_error");
            assert_eq!(error["error"]["param"], "chat_template_kwargs");
        }
    }
    assert_eq!(encoded.load(Ordering::SeqCst), 0);
    assert!(prompts.lock().unwrap().is_empty());
}
#[tokio::test]
async fn unsupported_absent_and_source_error_controls_never_use_legacy_fallback() {
    refuses_before_encode(IdPassthroughTokenizer::default()).await;
    refuses_before_encode(AbsentAdapter).await;
    refuses_before_encode(source(None)).await;
    refuses_before_encode(source(Some("{{ raise_exception('source refused') }}"))).await;
}

#[tokio::test]
async fn source_controls_do_not_bypass_tokenized_prompt_limits() {
    let limits = RequestLimits {
        max_prompt_tokens: 3,
        ..RequestLimits::default()
    };
    let (router, prompts, encoded) = router(source(Some(TEMPLATE)), Some(limits));
    let mut input = body(true);
    input["chat_template_kwargs"] = json!({"enable_thinking":false});
    let (status, _, text) = request(&router, input.to_string()).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{text}");
    assert_eq!(encoded.load(Ordering::SeqCst), 1);
    assert!(prompts.lock().unwrap().is_empty());
}
