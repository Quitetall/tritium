//! Fixed WebAssembly renderer; untrusted Jinja never runs in the native heap.

use std::sync::{
    OnceLock,
    atomic::{AtomicUsize, Ordering},
};

use serde_json::{Value, json};
use wasmi::{Config, Engine, Linker, Module, Store, StoreLimits, StoreLimitsBuilder};

use crate::error::NnError;

#[derive(serde::Serialize)]
struct ChatMessage<'a> {
    role: &'a str,
    content: &'a str,
}

#[derive(serde::Serialize)]
struct RenderContext<'a> {
    messages: Vec<ChatMessage<'a>>,
    add_generation_prompt: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    enable_thinking: Option<bool>,
    #[serde(flatten)]
    special_tokens: &'a serde_json::Map<String, Value>,
}

#[derive(serde::Serialize)]
struct RenderRequest<'a> {
    template: &'a str,
    context: RenderContext<'a>,
    max_output_bytes: usize,
}

#[derive(Default)]
struct BoundedInput(Vec<u8>);

impl std::io::Write for BoundedInput {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        if bytes.len() > MAX_INPUT_BYTES.saturating_sub(self.0.len()) {
            return Err(std::io::Error::other(
                "serialized chat input byte limit exceeded",
            ));
        }
        self.0.extend_from_slice(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

const MAX_TEMPLATE_BYTES: usize = 32 * 1024;
const MAX_MESSAGES: usize = 256;
const MAX_INPUT_BYTES: usize = 1024 * 1024;
const MAX_OUTPUT_BYTES: usize = 1024 * 1024;
const MAX_CONCURRENT: usize = 4;
const GUEST_MEMORY_BYTES: usize = 32 * 1024 * 1024;
const GUEST_FUEL: u64 = 50_000_000;

static ACTIVE: AtomicUsize = AtomicUsize::new(0);
#[cfg(test)]
pub(super) static TEST_SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn error(message: impl std::fmt::Display) -> NnError {
    NnError::Tokenizer(format!("source chat template: {message}"))
}

struct Admission;

impl Admission {
    fn acquire() -> Result<Self, NnError> {
        ACTIVE
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |count| {
                (count < MAX_CONCURRENT).then_some(count + 1)
            })
            .map_err(|_| error("renderer concurrency limit reached"))?;
        Ok(Self)
    }
}

impl Drop for Admission {
    fn drop(&mut self) {
        ACTIVE.fetch_sub(1, Ordering::Release);
    }
}

struct Renderer {
    engine: Engine,
    module: Module,
}

impl Renderer {
    fn shared() -> Result<&'static Self, NnError> {
        static RENDERER: OnceLock<Result<Renderer, String>> = OnceLock::new();
        RENDERER
            .get_or_init(|| {
                let mut config = Config::default();
                config.consume_fuel(true);
                config.allow_start_fn(false);
                config.set_max_recursion_depth(256);
                config.set_min_stack_height(16 * 1024);
                config.set_max_stack_height(256 * 1024);
                config.set_max_cached_stacks(0);
                let engine = Engine::new(&config);
                let module = Module::new(&engine, include_bytes!("chat_renderer.wasm").as_slice())
                    .map_err(|error| error.to_string())?;
                if module.imports().next().is_some() {
                    return Err("packaged renderer must have no imports".into());
                }
                Ok(Self { engine, module })
            })
            .as_ref()
            .map_err(error)
    }

    fn render(&self, input: &[u8], max_output_bytes: usize) -> Result<String, NnError> {
        let _admission = Admission::acquire()?;
        let limits = StoreLimitsBuilder::new()
            .memory_size(GUEST_MEMORY_BYTES)
            .memories(1)
            .tables(1)
            .table_elements(4096)
            .instances(1)
            .trap_on_grow_failure(true)
            .build();
        let mut store = Store::new(&self.engine, limits);
        store.limiter(|limits: &mut StoreLimits| limits);
        store.set_fuel(GUEST_FUEL).map_err(error)?;
        let instance = Linker::<StoreLimits>::new(&self.engine)
            .instantiate_and_start(&mut store, &self.module)
            .map_err(error)?;
        let memory = instance
            .get_memory(&store, "memory")
            .ok_or_else(|| error("missing guest memory"))?;
        let allocate = instance
            .get_typed_func::<u32, u32>(&store, "allocate")
            .map_err(error)?;
        let render = instance
            .get_typed_func::<(u32, u32), u64>(&store, "render_chat")
            .map_err(error)?;
        let len = u32::try_from(input.len()).map_err(error)?;
        let ptr = allocate.call(&mut store, len).map_err(error)?;
        memory
            .write(&mut store, ptr as usize, input)
            .map_err(error)?;
        let result = render.call(&mut store, (ptr, len)).map_err(error)?;
        let ptr = (result >> 32) as usize;
        let len = (result & u64::from(u32::MAX)) as usize;
        if len == 0 || len > max_output_bytes.max(4096) + 1 {
            return Err(error("invalid or oversized guest result"));
        }
        let end = ptr
            .checked_add(len)
            .ok_or_else(|| error("guest result range overflow"))?;
        let bytes = memory
            .data(&store)
            .get(ptr..end)
            .ok_or_else(|| error("guest result out of range"))?;
        let text = std::str::from_utf8(&bytes[1..]).map_err(error)?;
        match bytes[0] {
            0 if text.len() <= max_output_bytes => Ok(text.to_owned()),
            1 => Err(error(text)),
            _ => Err(error("invalid or oversized guest result")),
        }
    }
}

/// Immutable source selected from the authenticated tokenizer configuration.
#[derive(Debug)]
pub(super) struct SourceChatTemplate {
    text: String,
    special_tokens: serde_json::Map<String, Value>,
}

impl SourceChatTemplate {
    pub(super) fn from_config(
        config: &serde_json::Map<String, Value>,
    ) -> Result<Option<Self>, NnError> {
        let text = match config.get("chat_template") {
            None | Some(Value::Null) => return Ok(None),
            Some(Value::String(text)) => text.as_str(),
            Some(Value::Object(templates)) => {
                if templates.len() > 32 || templates.values().any(|value| !value.is_string()) {
                    return Err(error("invalid named templates"));
                }
                templates
                    .get("default")
                    .and_then(Value::as_str)
                    .ok_or_else(|| error("named templates require a string default"))?
            }
            Some(Value::Array(templates)) => {
                if templates.len() > 32 {
                    return Err(error("too many named templates"));
                }
                let mut names = std::collections::BTreeSet::new();
                let mut default = None;
                for entry in templates {
                    let name = entry
                        .get("name")
                        .and_then(Value::as_str)
                        .ok_or_else(|| error("invalid named template name"))?;
                    let text = entry
                        .get("template")
                        .and_then(Value::as_str)
                        .ok_or_else(|| error("invalid named template text"))?;
                    if !names.insert(name) {
                        return Err(error("duplicate named template"));
                    }
                    if name == "default" {
                        default = Some(text);
                    }
                }
                default.ok_or_else(|| error("named templates require a default"))?
            }
            _ => return Err(error("chat_template must be a string or named templates")),
        };
        if text.is_empty() || text.len() > MAX_TEMPLATE_BYTES {
            return Err(error("template must contain 1..32768 UTF-8 bytes"));
        }
        let mut special_tokens = serde_json::Map::new();
        let mut special_bytes = 0_usize;
        for (name, value) in config {
            if matches!(
                name.as_str(),
                "bos_token"
                    | "eos_token"
                    | "unk_token"
                    | "sep_token"
                    | "pad_token"
                    | "cls_token"
                    | "mask_token"
            ) {
                let text = match value {
                    Value::String(text) => Some(text.as_str()),
                    Value::Object(object) => object.get("content").and_then(Value::as_str),
                    Value::Null => None,
                    _ => return Err(error("invalid special token")),
                };
                if let Some(text) = text {
                    special_bytes = special_bytes
                        .checked_add(name.len())
                        .and_then(|n| n.checked_add(text.len()))
                        .ok_or_else(|| error("special token size overflow"))?;
                    if special_bytes > 16 * 1024 {
                        return Err(error("special tokens exceed byte budget"));
                    }
                    special_tokens.insert(name.clone(), Value::String(text.into()));
                }
            }
        }
        let template = Self {
            text: text.into(),
            special_tokens,
        };
        let input = serde_json::to_vec(&json!({"template":template.text,"context":{},"max_output_bytes":1,"validate_only":true})).map_err(error)?;
        Renderer::shared()?.render(&input, 1)?;
        Ok(Some(template))
    }

    pub(super) fn render(
        &self,
        messages: &[(&str, &str)],
        max_output_bytes: usize,
    ) -> Result<String, NnError> {
        self.render_with_options(
            messages,
            super::ChatRenderOptions::default(),
            max_output_bytes,
        )
    }

    pub(super) fn render_with_options(
        &self,
        messages: &[(&str, &str)],
        options: super::ChatRenderOptions,
        max_output_bytes: usize,
    ) -> Result<String, NnError> {
        if messages.is_empty()
            || messages.len() > MAX_MESSAGES
            || max_output_bytes == 0
            || max_output_bytes > MAX_OUTPUT_BYTES
        {
            return Err(error("message count or output budget out of range"));
        }
        let bytes = messages
            .iter()
            .try_fold(self.text.len(), |sum, (role, content)| {
                sum.checked_add(role.len())?.checked_add(content.len())
            });
        if bytes.is_none_or(|bytes| bytes > MAX_INPUT_BYTES) {
            return Err(error("input byte limit exceeded"));
        }
        let request = RenderRequest {
            template: &self.text,
            context: RenderContext {
                messages: messages
                    .iter()
                    .map(|&(role, content)| ChatMessage { role, content })
                    .collect(),
                add_generation_prompt: true,
                enable_thinking: options.enable_thinking(),
                special_tokens: &self.special_tokens,
            },
            max_output_bytes,
        };
        let mut input = BoundedInput::default();
        serde_json::to_writer(&mut input, &request).map_err(error)?;
        Renderer::shared()?.render(&input.0, max_output_bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use super::TEST_SERIAL as SERIAL;

    fn source(text: &str) -> SourceChatTemplate {
        SourceChatTemplate::from_config(json!({"chat_template":text}).as_object().unwrap())
            .unwrap()
            .unwrap()
    }

    #[test]
    fn exact_unicode_whitespace_special_tokens_and_no_recursive_evaluation() {
        let _serial = SERIAL.lock().unwrap();
        let config = json!({"bos_token":{"content":"<bos>"},"chat_template":"{{ bos_token }}{% for m in messages %}{{ m.role }}\n{{ m.content }}{% endfor %}{% if add_generation_prompt %}assistant\n<think>\n{% endif %}"});
        let source = SourceChatTemplate::from_config(config.as_object().unwrap())
            .unwrap()
            .unwrap();
        assert_eq!(
            source
                .render(&[("user", "  Héllo {{ 2 + 2 }} 🌍  ")], 1024)
                .unwrap(),
            "<bos>user\n  Héllo {{ 2 + 2 }} 🌍  assistant\n<think>\n"
        );
    }

    #[test]
    fn thinking_inputs_preserve_undefined_true_false_and_request_isolation() {
        let _serial = SERIAL.lock().unwrap();
        let source = source(
            "{% if enable_thinking is not defined %}default{% elif enable_thinking is false %}disabled{% else %}enabled{% endif %}|{{ messages[0].content }}",
        );
        let message = [("user", "Héllo {{ enable_thinking }} 🌍")];
        let options = super::super::ChatRenderOptions::default();
        for (option, expected) in [
            (options, "default"),
            (options.with_thinking(false), "disabled"),
            (options.with_thinking(true), "enabled"),
            (options, "default"),
        ] {
            assert_eq!(
                source.render_with_options(&message, option, 1024).unwrap(),
                format!("{expected}|Héllo {{{{ enable_thinking }}}} 🌍")
            );
        }
        assert_eq!(
            source.render(&message, 1024).unwrap(),
            source.render_with_options(&message, options, 1024).unwrap()
        );
        assert!(
            source
                .render_with_options(&message, options.with_thinking(false), 1)
                .is_err()
        );
    }

    #[test]
    fn default_selection_and_missing_are_not_error_fallbacks() {
        let _serial = SERIAL.lock().unwrap();
        assert!(
            SourceChatTemplate::from_config(json!({}).as_object().unwrap())
                .unwrap()
                .is_none()
        );
        for invalid in [
            json!(""),
            json!(true),
            json!({"other":"x"}),
            json!([{"name":"default","template":"x"},{"name":"default","template":"y"}]),
            json!({"default":"x","other":false}),
        ] {
            assert!(
                SourceChatTemplate::from_config(
                    json!({"chat_template":invalid}).as_object().unwrap()
                )
                .is_err()
            );
        }
        for valid in [
            json!({"default":"chosen","other":"wrong"}),
            json!([{"name":"other","template":"wrong"},{"name":"default","template":"chosen"}]),
        ] {
            let source = SourceChatTemplate::from_config(
                json!({"chat_template":valid}).as_object().unwrap(),
            )
            .unwrap()
            .unwrap();
            assert_eq!(source.render(&[("user", "Hi")], 1024).unwrap(), "chosen");
        }
        assert!(
            SourceChatTemplate::from_config(json!({"chat_template":"{% if"}).as_object().unwrap())
                .is_err()
        );
    }

    #[test]
    fn python_methods_namespace_reverse_and_hf_json_defaults() {
        let _serial = SERIAL.lock().unwrap();
        assert_eq!(
            self::source("{{ messages[0]|tojson }}")
                .render(&[("user", "Hi")], 1024)
                .unwrap(),
            "{\"role\": \"user\", \"content\": \"Hi\"}"
        );
        let source = source(
            "{% set ns = namespace(v='') %}{% for m in messages[::-1] %}{% set ns.v = ns.v + m.content.rstrip().split(' ')[0] %}{% endfor %}{{ ns.v }}|{{ {'z': '<>&\\\'é', 'a': [1, 2]}|tojson }}",
        );
        assert_eq!(
            source
                .render(&[("user", "first  "), ("assistant", "second  ")], 1024)
                .unwrap(),
            "secondfirst|{\"z\": \"<>&'é\", \"a\": [1, 2]}"
        );
        assert!(
            self::source("{{ {}|tojson(indent=2) }}")
                .render(&[("user", "Hi")], 1024)
                .is_err()
        );
    }

    #[test]
    fn errors_output_overflow_and_host_access_never_return_partial_prompts() {
        let _serial = SERIAL.lock().unwrap();
        for template in [
            "prefix{{ raise_exception('stop') }}",
            "{{ 'x' * 1024 }}",
            "{{ strftime_now('%Y') }}",
            "{% include '/etc/passwd' %}",
            "{{ cycler.__init__.__globals__.os.environ }}",
        ] {
            assert!(
                source(template).render(&[("user", "Hi")], 8).is_err(),
                "{template}"
            );
        }
        assert!(source("x").render(&[], 8).is_err());
        assert!(source("x").render(&[("user", "Hi")], 0).is_err());
        assert_eq!(
            source("healthy").render(&[("user", "Hi")], 8).unwrap(),
            "healthy"
        );
    }

    #[test]
    fn intermediate_heap_exhaustion_is_a_guest_error_and_next_request_survives() {
        let _serial = SERIAL.lock().unwrap();
        for template in [
            "{% set big = 'x' * 40000000 %}tiny",
            "{% set big = ([100000] * 10000000)|tojson %}tiny",
            "{% set big %}{{ 'x' * 40000000 }}{% endset %}tiny",
            "{% set big = range(100000)|list %}{{ big|tojson }}",
            "{% for n in range(100000) %}{% set x = n * n %}{% endfor %}tiny",
        ] {
            // Constant folding can exhaust the guest during compilation. Either
            // compilation or rendering may refuse; neither may escape the guest.
            if let Ok(Some(source)) = SourceChatTemplate::from_config(
                json!({"chat_template":template}).as_object().unwrap(),
            ) {
                assert!(source.render(&[("user", "Hi")], 8).is_err(), "{template}");
            }
            assert_eq!(
                source("alive").render(&[("user", "Hi")], 8).unwrap(),
                "alive"
            );
        }
    }

    #[test]
    fn concurrency_admission_is_nonblocking_and_released_on_drop() {
        let _serial = SERIAL.lock().unwrap();
        let permits: Vec<_> = (0..MAX_CONCURRENT)
            .map(|_| Admission::acquire().unwrap())
            .collect();
        assert!(Admission::acquire().is_err());
        drop(permits);
        assert!(Admission::acquire().is_ok());
    }

    #[test]
    fn work_and_serialized_input_budgets_fail_closed() {
        let _serial = SERIAL.lock().unwrap();
        let failure = source("{% for n in range(100000) %}{% set x = n * n %}{% endfor %}tiny")
            .render(&[("user", "Hi")], 1024)
            .unwrap_err();
        assert!(failure.to_string().contains("fuel"), "{failure}");
        let escaped = "\0".repeat(200_000);
        let failure = source("tiny")
            .render(&[("user", &escaped)], 1024)
            .unwrap_err();
        assert!(
            failure
                .to_string()
                .contains("serialized chat input byte limit")
        );
        assert_eq!(
            source("alive").render(&[("user", "Hi")], 8).unwrap(),
            "alive"
        );
    }
}
