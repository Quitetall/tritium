//! Typed, deterministic and verifiable evidence logs for Tritium.
//!
//! An evidence event is accepted only when its schema identifier is registered
//! with the run recorder. Each span has an independent sequence and digest
//! chain, so interleaving events from different spans does not affect the run
//! root. The JSONL representation and event digest use RFC 8785 JCS.

#![forbid(unsafe_code)]
#![deny(missing_docs)]

use std::collections::BTreeMap;
use std::fmt;

use serde::Serialize;
use serde_json::Value;
pub use tritium_schema::{UnknownReason, Verdict};

const EVENT_DOMAIN: &[u8] = b"tritium.evidence.event.v1\0";
const ROOT_DOMAIN: &[u8] = b"tritium.evidence.run-root.v1\0";
const ZERO_DIGEST: [u8; 32] = [0; 32];
const MAX_SAFE_JSON_INTEGER: u64 = (1_u64 << 53) - 1;

/// Versioned common evidence envelope with a dynamic JSON event payload.
pub type EvidenceEnvelope = tritium_schema::EvidenceEnvelope<Value>;

#[derive(Serialize)]
struct DigestBody<'a> {
    schema: &'a str,
    v: u16,
    run: &'a str,
    seq: u64,
    span: &'a str,
    parent: &'a Option<String>,
    t: u64,
    payload: &'a Value,
}

/// A typed event that is admissible in an evidence log.
pub trait EvidenceEvent: Serialize {
    /// Canonical event schema id.
    const SCHEMA_ID: &'static str;
    /// Nonzero event schema version.
    const VERSION: u16;
}

/// A registered event schema and its accepted version.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EventRegistration {
    /// Canonical schema id.
    pub schema: &'static str,
    /// Accepted nonzero version.
    pub version: u16,
}

/// A recorder for one run and a fixed registry of typed event schemas.
#[derive(Debug)]
pub struct EvidenceRecorder {
    run: String,
    registered: BTreeMap<String, u16>,
    next_seq: BTreeMap<String, u64>,
    last_digest: BTreeMap<String, [u8; 32]>,
    last_time: BTreeMap<String, u64>,
    span_parent: BTreeMap<String, Option<String>>,
    events: Vec<EvidenceEnvelope>,
}

impl EvidenceRecorder {
    /// Create a recorder, rejecting an empty run id or malformed registry entry.
    pub fn new(
        run: impl Into<String>,
        registered: impl IntoIterator<Item = EventRegistration>,
    ) -> Result<Self, EvidenceError> {
        let run = run.into();
        if run.is_empty() {
            return Err(EvidenceError::InvalidRunId);
        }
        let mut schemas = BTreeMap::new();
        for registration in registered {
            validate_schema_id(registration.schema)?;
            if registration.version == 0 {
                return Err(EvidenceError::InvalidVersion);
            }
            if schemas
                .insert(registration.schema.to_owned(), registration.version)
                .is_some()
            {
                return Err(EvidenceError::DuplicateRegistration(
                    registration.schema.to_owned(),
                ));
            }
        }
        Ok(Self {
            run,
            registered: schemas,
            next_seq: BTreeMap::new(),
            last_digest: BTreeMap::new(),
            last_time: BTreeMap::new(),
            span_parent: BTreeMap::new(),
            events: Vec::new(),
        })
    }

    /// Emit one registered event and append it to this run's in-memory log.
    pub fn emit<E: EvidenceEvent>(
        &mut self,
        span: &str,
        parent: Option<&str>,
        logical_time: u64,
        event: &E,
    ) -> Result<&EvidenceEnvelope, EvidenceError> {
        if self.registered.get(E::SCHEMA_ID) != Some(&E::VERSION) {
            return Err(EvidenceError::UnregisteredSchema(format!(
                "{}@{}",
                E::SCHEMA_ID,
                E::VERSION
            )));
        }
        if span.is_empty() || parent.is_some_and(str::is_empty) {
            return Err(EvidenceError::InvalidSpan);
        }
        if parent == Some(span) {
            return Err(EvidenceError::InvalidParent(span.to_owned()));
        }
        let parent = parent.map(str::to_owned);
        if let Some(previous_parent) = self.span_parent.get(span)
            && previous_parent != &parent
        {
            return Err(EvidenceError::ParentMismatch(span.to_owned()));
        }
        if E::VERSION == 0 {
            return Err(EvidenceError::InvalidVersion);
        }
        if logical_time > MAX_SAFE_JSON_INTEGER {
            return Err(EvidenceError::UnsafeInteger("t"));
        }
        if self
            .last_time
            .get(span)
            .is_some_and(|previous_time| logical_time < *previous_time)
        {
            return Err(EvidenceError::NonMonotonicTime(span.to_owned()));
        }
        let payload = serde_json::to_value(event)
            .map_err(|error| EvidenceError::Serialization(error.to_string()))?;
        validate_json_numbers(&payload)?;
        let seq = *self.next_seq.get(span).unwrap_or(&0);
        if seq > MAX_SAFE_JSON_INTEGER {
            return Err(EvidenceError::UnsafeInteger("seq"));
        }
        let next_seq = seq.checked_add(1).ok_or(EvidenceError::SequenceOverflow)?;
        let body = DigestBody {
            schema: E::SCHEMA_ID,
            v: E::VERSION,
            run: &self.run,
            seq,
            span,
            parent: &parent,
            t: logical_time,
            payload: &payload,
        };
        let previous = *self.last_digest.get(span).unwrap_or(&ZERO_DIGEST);
        let digest = event_digest(previous, &body)?;
        let envelope = EvidenceEnvelope {
            schema: E::SCHEMA_ID.to_owned(),
            v: E::VERSION,
            run: self.run.clone(),
            seq,
            span: span.to_owned(),
            parent: parent.clone(),
            t: logical_time,
            payload,
            digest: encode_hex(&digest),
        };
        self.span_parent.entry(span.to_owned()).or_insert(parent);
        self.next_seq.insert(span.to_owned(), next_seq);
        self.last_digest.insert(span.to_owned(), digest);
        self.last_time.insert(span.to_owned(), logical_time);
        self.events.push(envelope);
        Ok(self.events.last().expect("event was just appended"))
    }

    /// Borrow events in emission order.
    #[must_use]
    pub fn events(&self) -> &[EvidenceEnvelope] {
        &self.events
    }

    /// Serialize the log as canonical JSON Lines.
    pub fn to_jsonl(&self) -> Result<String, EvidenceError> {
        let mut output = String::new();
        for event in &self.events {
            output.push_str(
                &serde_json_canonicalizer::to_string(event)
                    .map_err(|error| EvidenceError::Canonicalization(error.to_string()))?,
            );
            output.push('\n');
        }
        Ok(output)
    }

    /// Compute the deterministic root over sorted span identifiers and final chain digests.
    pub fn root_digest(&self) -> Result<String, EvidenceError> {
        verify_events(&self.events)?;
        root_digest(&self.last_digest)
    }
}

/// Parse and verify a JSONL log, including per-span sequence and digest chains.
pub fn verify_jsonl(input: &str) -> Result<VerifiedLog, EvidenceError> {
    let mut events = Vec::new();
    for (line_index, line) in input.lines().enumerate() {
        if line.trim().is_empty() {
            return Err(EvidenceError::EmptyLine(line_index + 1));
        }
        let event = serde_json::from_str(line)
            .map_err(|error| EvidenceError::InvalidJson(line_index + 1, error.to_string()))?;
        let canonical = serde_json_canonicalizer::to_string(&event)
            .map_err(|error| EvidenceError::Canonicalization(error.to_string()))?;
        if canonical != line {
            return Err(EvidenceError::NonCanonicalLine(line_index + 1));
        }
        events.push(event);
    }
    verify_events(&events)?;
    let final_digests = final_digests(&events)?;
    Ok(VerifiedLog {
        events,
        root_digest: root_digest(&final_digests)?,
    })
}

/// Parse and verify a log against the caller's registered event schema set.
pub fn verify_jsonl_with_registry(
    input: &str,
    registered: impl IntoIterator<Item = EventRegistration>,
) -> Result<VerifiedLog, EvidenceError> {
    let mut registry = BTreeMap::new();
    for registration in registered {
        validate_schema_id(registration.schema)?;
        if registration.version == 0 {
            return Err(EvidenceError::InvalidVersion);
        }
        if registry
            .insert(registration.schema, registration.version)
            .is_some()
        {
            return Err(EvidenceError::DuplicateRegistration(
                registration.schema.to_owned(),
            ));
        }
    }
    let verified = verify_jsonl(input)?;
    for event in &verified.events {
        if registry.get(event.schema.as_str()) != Some(&event.v) {
            return Err(EvidenceError::UnregisteredSchema(format!(
                "{}@{}",
                event.schema, event.v
            )));
        }
    }
    Ok(verified)
}

/// A fully verified event log and its deterministic run root.
#[derive(Clone, Debug, PartialEq)]
pub struct VerifiedLog {
    /// Verified events in their original JSONL order.
    pub events: Vec<EvidenceEnvelope>,
    /// BLAKE3 run root over sorted span roots.
    pub root_digest: String,
}

/// One claim and the review obligation attached to it.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ClaimVerdict {
    /// Whether the claim is required for the aggregate.
    pub required: bool,
    /// Whether its producer is prohibited from independently passing it.
    pub requires_independent: bool,
    /// Whether a separate reviewer/verifier has cleared it.
    pub independently_cleared: bool,
    /// The claim's current verdict.
    pub verdict: Verdict,
}

/// Aggregate required claims without coercing unknown evidence to pass.
#[must_use]
pub fn aggregate_required(claims: &[ClaimVerdict]) -> Verdict {
    let required: Vec<_> = claims.iter().filter(|claim| claim.required).collect();
    if required.is_empty() {
        return Verdict::Unknown {
            reason: UnknownReason::MissingEvidence,
        };
    }
    if required.iter().any(|claim| claim.verdict == Verdict::Fail) {
        return Verdict::Fail;
    }
    for claim in required {
        if claim.requires_independent && !claim.independently_cleared {
            return Verdict::Unknown {
                reason: UnknownReason::AwaitingIndependentReview,
            };
        }
        if let Verdict::Unknown { reason } = &claim.verdict {
            return Verdict::Unknown { reason: *reason };
        }
    }
    Verdict::Pass
}

/// Failure while producing, parsing, or verifying evidence.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum EvidenceError {
    /// Run identifier is empty.
    InvalidRunId,
    /// Event schema identifier is malformed.
    InvalidSchemaId(String),
    /// Event schema was not registered for this run.
    UnregisteredSchema(String),
    /// The same schema id was registered more than once.
    DuplicateRegistration(String),
    /// Span id is empty or parent id is empty.
    InvalidSpan,
    /// A span refers to itself as its parent.
    InvalidParent(String),
    /// Events in one span declare inconsistent parent spans.
    ParentMismatch(String),
    /// An event log refers to a parent span that has no events.
    MissingParent(String),
    /// The span hierarchy contains a cycle.
    ParentCycle(String),
    /// Schema version must be nonzero.
    InvalidVersion,
    /// An integer is outside JCS/I-JSON's exact IEEE-754 integer range.
    UnsafeInteger(&'static str),
    /// Logical time moved backwards within a span.
    NonMonotonicTime(String),
    /// Canonical serialization failed.
    Canonicalization(String),
    /// A typed event could not be converted into its JSON payload.
    Serialization(String),
    /// A JSONL line is empty.
    EmptyLine(usize),
    /// A JSONL line is malformed.
    InvalidJson(usize, String),
    /// A JSONL event is valid JSON but is not encoded in canonical JCS form.
    NonCanonicalLine(usize),
    /// A digest is not 64 lowercase hexadecimal characters.
    InvalidDigest,
    /// An event has an unexpected sequence number for its span.
    InvalidSequence {
        /// Span whose sequence is invalid.
        span: String,
        /// Next sequence number required by the chain.
        expected: u64,
        /// Sequence number found in the event.
        got: u64,
    },
    /// A span sequence reached the end of the u64 domain.
    SequenceOverflow,
    /// An event's digest does not match its content or span predecessor.
    DigestMismatch {
        /// Span whose event digest failed verification.
        span: String,
        /// Sequence number of the invalid event.
        seq: u64,
    },
    /// A root cannot be produced for an empty run.
    EmptyRun,
}

impl fmt::Display for EvidenceError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for EvidenceError {}

fn validate_schema_id(schema: &str) -> Result<(), EvidenceError> {
    let valid = schema.starts_with("tritium.")
        && schema.split('.').count() >= 3
        && schema.split('.').all(|part| {
            !part.is_empty()
                && part
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_' || byte == b'-')
        });
    if valid {
        Ok(())
    } else {
        Err(EvidenceError::InvalidSchemaId(schema.to_owned()))
    }
}

fn validate_json_numbers(value: &Value) -> Result<(), EvidenceError> {
    match value {
        Value::Number(number) => {
            if number
                .as_u64()
                .is_some_and(|integer| integer > MAX_SAFE_JSON_INTEGER)
                || number
                    .as_i64()
                    .is_some_and(|integer| integer.unsigned_abs() > MAX_SAFE_JSON_INTEGER)
            {
                return Err(EvidenceError::UnsafeInteger("payload"));
            }
        }
        Value::Array(values) => {
            for value in values {
                validate_json_numbers(value)?;
            }
        }
        Value::Object(values) => {
            for value in values.values() {
                validate_json_numbers(value)?;
            }
        }
        Value::Null | Value::Bool(_) | Value::String(_) => {}
    }
    Ok(())
}

fn event_digest(previous: [u8; 32], body: &DigestBody<'_>) -> Result<[u8; 32], EvidenceError> {
    let canonical = serde_json_canonicalizer::to_vec(body)
        .map_err(|error| EvidenceError::Canonicalization(error.to_string()))?;
    let mut hasher = blake3::Hasher::new();
    hasher.update(EVENT_DOMAIN);
    hasher.update(&previous);
    hasher.update(&canonical);
    Ok(*hasher.finalize().as_bytes())
}

fn verify_events(events: &[EvidenceEnvelope]) -> Result<(), EvidenceError> {
    let mut next_seq = BTreeMap::<String, u64>::new();
    let mut last_digest = BTreeMap::<String, [u8; 32]>::new();
    let mut last_time = BTreeMap::<String, u64>::new();
    let mut span_parent = BTreeMap::<String, Option<String>>::new();
    let mut run = None;
    for event in events {
        validate_schema_id(&event.schema)?;
        if event.v == 0 {
            return Err(EvidenceError::InvalidVersion);
        }
        if event.seq > MAX_SAFE_JSON_INTEGER {
            return Err(EvidenceError::UnsafeInteger("seq"));
        }
        if event.t > MAX_SAFE_JSON_INTEGER {
            return Err(EvidenceError::UnsafeInteger("t"));
        }
        if last_time
            .get(&event.span)
            .is_some_and(|previous_time| event.t < *previous_time)
        {
            return Err(EvidenceError::NonMonotonicTime(event.span.clone()));
        }
        validate_json_numbers(&event.payload)?;
        if event.span.is_empty() || event.parent.as_ref().is_some_and(String::is_empty) {
            return Err(EvidenceError::InvalidSpan);
        }
        if event.parent.as_deref() == Some(event.span.as_str()) {
            return Err(EvidenceError::InvalidParent(event.span.clone()));
        }
        if let Some(previous_parent) = span_parent.get(&event.span) {
            if previous_parent != &event.parent {
                return Err(EvidenceError::ParentMismatch(event.span.clone()));
            }
        } else {
            span_parent.insert(event.span.clone(), event.parent.clone());
        }
        match &run {
            Some(expected) if expected != &event.run => return Err(EvidenceError::InvalidRunId),
            None if event.run.is_empty() => return Err(EvidenceError::InvalidRunId),
            None => run = Some(event.run.clone()),
            _ => {}
        }
        let expected = *next_seq.get(&event.span).unwrap_or(&0);
        if event.seq != expected {
            return Err(EvidenceError::InvalidSequence {
                span: event.span.clone(),
                expected,
                got: event.seq,
            });
        }
        let previous = *last_digest.get(&event.span).unwrap_or(&ZERO_DIGEST);
        let body = DigestBody {
            schema: &event.schema,
            v: event.v,
            run: &event.run,
            seq: event.seq,
            span: &event.span,
            parent: &event.parent,
            t: event.t,
            payload: &event.payload,
        };
        let digest = event_digest(previous, &body)?;
        if event.digest != encode_hex(&digest) {
            return Err(EvidenceError::DigestMismatch {
                span: event.span.clone(),
                seq: event.seq,
            });
        }
        next_seq.insert(
            event.span.clone(),
            expected
                .checked_add(1)
                .ok_or(EvidenceError::SequenceOverflow)?,
        );
        last_digest.insert(event.span.clone(), digest);
        last_time.insert(event.span.clone(), event.t);
    }
    if events.is_empty() {
        return Err(EvidenceError::EmptyRun);
    }
    for (span, parent) in &span_parent {
        let Some(parent) = parent else { continue };
        if !span_parent.contains_key(parent) {
            return Err(EvidenceError::MissingParent(parent.clone()));
        }
        let mut ancestor = Some(parent.as_str());
        while let Some(current) = ancestor {
            if current == span {
                return Err(EvidenceError::ParentCycle(span.clone()));
            }
            ancestor = span_parent.get(current).and_then(Option::as_deref);
        }
    }
    Ok(())
}

fn final_digests(events: &[EvidenceEnvelope]) -> Result<BTreeMap<String, [u8; 32]>, EvidenceError> {
    let mut roots = BTreeMap::new();
    for event in events {
        roots.insert(event.span.clone(), decode_hex(&event.digest)?);
    }
    Ok(roots)
}

fn root_digest(spans: &BTreeMap<String, [u8; 32]>) -> Result<String, EvidenceError> {
    if spans.is_empty() {
        return Err(EvidenceError::EmptyRun);
    }
    let mut hasher = blake3::Hasher::new();
    hasher.update(ROOT_DOMAIN);
    for (span, digest) in spans {
        hasher.update(&(span.len() as u64).to_le_bytes());
        hasher.update(span.as_bytes());
        hasher.update(digest);
    }
    Ok(encode_hex(hasher.finalize().as_bytes()))
}

fn encode_hex(bytes: &[u8; 32]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(64);
    for byte in bytes {
        output.push(HEX[usize::from(byte >> 4)] as char);
        output.push(HEX[usize::from(byte & 0x0f)] as char);
    }
    output
}

fn decode_hex(value: &str) -> Result<[u8; 32], EvidenceError> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(EvidenceError::InvalidDigest);
    }
    let mut output = [0_u8; 32];
    let (chunks, remainder) = value.as_bytes().as_chunks::<2>();
    if !remainder.is_empty() {
        return Err(EvidenceError::InvalidDigest);
    }
    for (index, chunk) in chunks.iter().enumerate() {
        let high = hex_nibble(chunk[0]).ok_or(EvidenceError::InvalidDigest)?;
        let low = hex_nibble(chunk[1]).ok_or(EvidenceError::InvalidDigest)?;
        output[index] = (high << 4) | low;
    }
    Ok(output)
}

fn hex_nibble(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ClaimVerdict, EventRegistration, EvidenceError, EvidenceRecorder, UnknownReason, Verdict,
        aggregate_required, verify_jsonl, verify_jsonl_with_registry,
    };

    const SCHEMA: &str = "tritium.test.observation";

    #[derive(serde::Serialize)]
    struct Observation {
        ok: bool,
        label: String,
    }

    impl super::EvidenceEvent for Observation {
        const SCHEMA_ID: &'static str = SCHEMA;
        const VERSION: u16 = 1;
    }

    #[derive(serde::Serialize)]
    struct WideIntegerEvent {
        value: u64,
    }

    impl super::EvidenceEvent for WideIntegerEvent {
        const SCHEMA_ID: &'static str = SCHEMA;
        const VERSION: u16 = 1;
    }

    fn recorder() -> EvidenceRecorder {
        EvidenceRecorder::new(
            "run-1",
            [super::EventRegistration {
                schema: SCHEMA,
                version: 1,
            }],
        )
        .expect("valid registry")
    }

    fn observation(ok: bool, label: &str) -> Observation {
        Observation {
            ok,
            label: label.to_owned(),
        }
    }

    #[test]
    fn log_is_canonical_verifiable_and_deterministic() {
        let mut first = recorder();
        first
            .emit("root", None, 0, &observation(true, "stable"))
            .expect("registered event");
        first
            .emit("child", Some("root"), 1, &observation(true, "child"))
            .expect("registered event");
        let jsonl = first.to_jsonl().expect("canonical JSONL");
        assert!(jsonl.starts_with(r#"{"digest":"#));
        let verified = verify_jsonl(&jsonl).expect("valid hash chains");
        assert_eq!(verified.root_digest, first.root_digest().expect("root"));

        let mut second = recorder();
        second
            .emit("root", None, 0, &observation(true, "stable"))
            .expect("registered event");
        second
            .emit("child", Some("root"), 1, &observation(true, "child"))
            .expect("registered event");
        assert_eq!(jsonl, second.to_jsonl().expect("canonical JSONL"));
    }

    #[test]
    fn tampered_payload_is_rejected() {
        let mut recorder = recorder();
        recorder
            .emit("root", None, 0, &observation(true, "stable"))
            .expect("registered event");
        let mut log = recorder.to_jsonl().expect("canonical JSONL");
        log = log.replace("\"ok\":true", "\"ok\":false");
        assert!(matches!(
            verify_jsonl(&log),
            Err(EvidenceError::DigestMismatch { .. })
        ));
    }

    #[test]
    fn verification_checks_the_registered_schema_version() {
        let mut recorder = recorder();
        recorder
            .emit("root", None, 0, &observation(true, "stable"))
            .expect("registered event");
        let log = recorder.to_jsonl().expect("canonical JSONL");
        assert!(matches!(
            verify_jsonl_with_registry(&log, []),
            Err(EvidenceError::UnregisteredSchema(_))
        ));
        assert!(matches!(
            verify_jsonl_with_registry(
                &log,
                [EventRegistration {
                    schema: SCHEMA,
                    version: 2,
                }]
            ),
            Err(EvidenceError::UnregisteredSchema(_))
        ));
    }

    #[test]
    fn rejects_integers_outside_the_exact_jcs_number_domain() {
        let mut recorder = recorder();
        let error = recorder
            .emit(
                "root",
                None,
                0,
                &WideIntegerEvent {
                    value: (1_u64 << 53) + 1,
                },
            )
            .expect_err("large integers must not be silently rounded by JCS");
        assert_eq!(error, EvidenceError::UnsafeInteger("payload"));
        let event = recorder
            .emit("root", None, 0, &observation(true, "still-sequence-zero"))
            .expect("failed emission must not mutate the span state");
        assert_eq!(event.seq, 0);
    }

    #[test]
    fn logical_time_is_monotonic_per_span() {
        let mut recorder = recorder();
        recorder
            .emit("root", None, 4, &observation(true, "first"))
            .expect("first event");
        assert_eq!(
            recorder.emit("root", None, 3, &observation(true, "backwards")),
            Err(EvidenceError::NonMonotonicTime("root".into()))
        );
    }

    #[test]
    fn root_is_independent_of_cross_span_interleaving() {
        let mut first = recorder();
        first
            .emit("root", None, 0, &observation(true, "root-0"))
            .expect("root event");
        first
            .emit("child", Some("root"), 1, &observation(true, "child-0"))
            .expect("child event");
        first
            .emit("root", None, 2, &observation(true, "root-1"))
            .expect("second root event");

        let mut second = recorder();
        second
            .emit("child", Some("root"), 1, &observation(true, "child-0"))
            .expect("child event");
        second
            .emit("root", None, 0, &observation(true, "root-0"))
            .expect("root event");
        second
            .emit("root", None, 2, &observation(true, "root-1"))
            .expect("second root event");
        assert_eq!(
            first.root_digest().expect("first root"),
            second.root_digest().expect("second root")
        );
    }

    #[test]
    fn unknown_and_unreviewed_claims_never_pass() {
        let unreviewed = [ClaimVerdict {
            required: true,
            requires_independent: true,
            independently_cleared: false,
            verdict: Verdict::Pass,
        }];
        assert_eq!(
            aggregate_required(&unreviewed),
            Verdict::Unknown {
                reason: UnknownReason::AwaitingIndependentReview
            }
        );
        let missing = [ClaimVerdict {
            required: true,
            requires_independent: false,
            independently_cleared: false,
            verdict: Verdict::Unknown {
                reason: UnknownReason::MissingCapability,
            },
        }];
        assert_eq!(
            aggregate_required(&missing),
            Verdict::Unknown {
                reason: UnknownReason::MissingCapability
            }
        );
    }
}
