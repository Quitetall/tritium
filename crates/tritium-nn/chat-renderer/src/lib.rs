//! Fixed no-import guest. Model-provided Jinja is parsed only inside this memory.

use std::io::{self, Write};

use minijinja::{Environment, Error, ErrorKind, Value, value::Kwargs};
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct Request {
    template: String,
    context: serde_json::Value,
    max_output_bytes: usize,
    #[serde(default)]
    validate_only: bool,
}

struct BoundedOutput {
    bytes: Vec<u8>,
    limit: usize,
}

impl Write for BoundedOutput {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if bytes.len() > self.limit.saturating_sub(self.bytes.len()) {
            return Err(io::Error::other("chat template output limit exceeded"));
        }
        self.bytes.extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

// Python json.dumps uses these spaces by default, without HTML escaping.
struct PythonJson;

impl serde_json::ser::Formatter for PythonJson {
    fn begin_array_value<W: Write + ?Sized>(
        &mut self,
        writer: &mut W,
        first: bool,
    ) -> io::Result<()> {
        if !first {
            writer.write_all(b", ")?;
        }
        Ok(())
    }

    fn begin_object_key<W: Write + ?Sized>(
        &mut self,
        writer: &mut W,
        first: bool,
    ) -> io::Result<()> {
        if !first {
            writer.write_all(b", ")?;
        }
        Ok(())
    }

    fn begin_object_value<W: Write + ?Sized>(&mut self, writer: &mut W) -> io::Result<()> {
        writer.write_all(b": ")
    }
}

fn tojson(value: Value, kwargs: Kwargs) -> Result<String, Error> {
    // Do not silently accept options with different semantics. Additional HF
    // options need differential fixtures before becoming supported behavior.
    kwargs.assert_all_used()?;
    let mut result = Vec::new();
    value
        .serialize(&mut serde_json::Serializer::with_formatter(
            &mut result,
            PythonJson,
        ))
        .map_err(|error| Error::new(ErrorKind::InvalidOperation, error.to_string()))?;
    String::from_utf8(result)
        .map_err(|error| Error::new(ErrorKind::InvalidOperation, error.to_string()))
}

fn render(input: &[u8]) -> Result<Vec<u8>, String> {
    let request: Request = serde_json::from_slice(input).map_err(|error| error.to_string())?;
    let mut env = Environment::new();
    env.set_fuel(Some(100_000));
    env.set_recursion_limit(64);
    env.set_syntax(
        minijinja::syntax::SyntaxConfig::builder()
            .trim_blocks(true)
            .lstrip_blocks(true)
            .build()
            .map_err(|error| error.to_string())?,
    );
    env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
    env.add_function(
        "raise_exception",
        |message: String| -> Result<Value, Error> {
            Err(Error::new(ErrorKind::InvalidOperation, message))
        },
    );
    env.add_filter("tojson", tojson);
    env.add_template("chat", &request.template)
        .map_err(|error| error.to_string())?;
    if request.validate_only {
        return Ok(Vec::new());
    }
    let mut output = BoundedOutput {
        bytes: Vec::new(),
        limit: request.max_output_bytes,
    };
    env.get_template("chat")
        .map_err(|error| error.to_string())?
        .render_captured_to(minijinja::value::Serde(&request.context), &mut output)
        .map_err(|error| error.to_string())?;
    Ok(output.bytes)
}

/// The host writes exactly `len` bytes into this allocation before `render_chat`.
#[unsafe(no_mangle)]
pub extern "C" fn allocate(len: u32) -> u32 {
    let bytes = vec![0_u8; len as usize].into_boxed_slice();
    Box::into_raw(bytes).cast::<u8>() as u32
}

/// Return `(pointer << 32) | length`, including a one-byte result discriminator.
/// The host always drops this entire guest after reading the result.
///
/// # Safety
/// `ptr,len` must identify the allocation returned by `allocate(len)`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn render_chat(ptr: u32, len: u32) -> u64 {
    // SAFETY: the only caller is the fixed host adapter, which uses allocate(len)
    // and checked Memory::write. No model text controls the guest ABI.
    let input = unsafe { std::slice::from_raw_parts(ptr as *const u8, len as usize) };
    let (status, bytes) = match render(input) {
        Ok(bytes) => (0, bytes),
        Err(message) => (1, message.as_bytes()[..message.len().min(4096)].to_vec()),
    };
    let mut result = Vec::with_capacity(bytes.len() + 1);
    result.push(status);
    result.extend_from_slice(&bytes);
    let len = result.len() as u64;
    let ptr = Box::into_raw(result.into_boxed_slice()).cast::<u8>() as u64;
    (ptr << 32) | len
}
