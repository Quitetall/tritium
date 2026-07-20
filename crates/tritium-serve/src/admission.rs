//! Bounded authentication and per-principal admission control.

use std::io;
use std::sync::{Mutex, MutexGuard};
use std::time::Instant;

/// Maximum rotating bearer keys accepted by one server process.
pub const MAX_BEARER_TOKENS: usize = 32;

const MINUTE_NANOS: u128 = 60_000_000_000;

/// Token-bucket parameters for one authenticated principal.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PrincipalRateLimit {
    /// Sustained admitted requests per minute.
    pub requests_per_minute: u32,
    /// Maximum immediately admitted burst after an idle period.
    pub burst: u32,
}

/// Authentication and per-principal request-governance policy.
///
/// Bearer tokens form the complete principal set. Their configured index is
/// the internal principal identity; secrets are never copied into metrics or
/// logs. With an empty list, all loopback callers share one anonymous bucket.
#[derive(Clone, Default, Eq, PartialEq)]
pub struct AdmissionPolicy {
    /// Rotating bearer-token whitelist. At most [`MAX_BEARER_TOKENS`].
    pub bearer_tokens: Vec<String>,
    /// Optional fixed-cardinality rate limit for expensive generation routes.
    pub rate_limit: Option<PrincipalRateLimit>,
}

impl std::fmt::Debug for AdmissionPolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AdmissionPolicy")
            .field("bearer_token_count", &self.bearer_tokens.len())
            .field("rate_limit", &self.rate_limit)
            .finish()
    }
}

#[derive(Debug)]
struct Bucket {
    // Fixed-point credit: one request costs MINUTE_NANOS units and each
    // elapsed nanosecond replenishes `requests_per_minute` units.
    credit: u128,
    last: Instant,
}

#[derive(Debug)]
struct PrincipalLimiter {
    requests_per_minute: u128,
    capacity: u128,
    buckets: Mutex<Vec<Bucket>>,
}

/// Result of checking one principal's bucket.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum AdmissionDecision {
    /// The request consumed one token and may continue.
    Allow,
    /// The request was rejected; retry after this many whole seconds.
    Reject { retry_after_secs: u64 },
}

/// Validated immutable authentication set plus bounded mutable buckets.
#[derive(Debug)]
pub(crate) struct Admission {
    token_digests: Vec<[u8; 32]>,
    limiter: Option<PrincipalLimiter>,
}

impl Admission {
    /// Validate a public policy and merge the legacy single-token config.
    pub(crate) fn new(legacy_token: Option<&str>, mut policy: AdmissionPolicy) -> io::Result<Self> {
        if let Some(token) = legacy_token
            && !policy
                .bearer_tokens
                .iter()
                .any(|candidate| candidate == token)
        {
            policy.bearer_tokens.push(token.to_owned());
        }
        validate_tokens(&policy.bearer_tokens)?;
        let principal_count = policy.bearer_tokens.len().max(1);
        let limiter = policy
            .rate_limit
            .map(|limit| PrincipalLimiter::new(limit, principal_count))
            .transpose()?;
        Ok(Self {
            token_digests: policy
                .bearer_tokens
                .iter()
                .map(|token| *blake3::hash(token.as_bytes()).as_bytes())
                .collect(),
            limiter,
        })
    }

    /// Preserve the original infallible constructor contract.
    pub(crate) fn legacy(token: Option<&str>) -> Self {
        let token_digests = token
            .map(|token| *blake3::hash(token.as_bytes()).as_bytes())
            .into_iter()
            .collect();
        Self {
            token_digests,
            limiter: None,
        }
    }

    /// Authenticate a presented bearer value and return its bounded index.
    /// No configured tokens means one anonymous principal.
    pub(crate) fn authenticate(&self, presented: Option<&str>) -> Option<usize> {
        if self.token_digests.is_empty() {
            return Some(0);
        }
        let presented = presented?;
        if presented.len() > 4096 {
            return None;
        }
        let presented = blake3::hash(presented.as_bytes());
        let mut matched = None;
        // Scan the full configured set so key position does not create an
        // early-return timing signal. Duplicate keys are rejected at startup.
        for (index, expected) in self.token_digests.iter().enumerate() {
            if constant_time_eq(expected, presented.as_bytes()) {
                matched = Some(index);
            }
        }
        matched
    }

    pub(crate) fn admit(&self, principal: usize) -> AdmissionDecision {
        self.limiter
            .as_ref()
            .map_or(AdmissionDecision::Allow, |limiter| limiter.check(principal))
    }
}

impl PrincipalLimiter {
    fn new(limit: PrincipalRateLimit, principal_count: usize) -> io::Result<Self> {
        if limit.requests_per_minute == 0 || limit.burst == 0 {
            return Err(invalid(
                "rate-limit requests_per_minute and burst must both be >= 1",
            ));
        }
        let capacity = u128::from(limit.burst)
            .checked_mul(MINUTE_NANOS)
            .ok_or_else(|| invalid("rate-limit burst is too large"))?;
        let now = Instant::now();
        let buckets = (0..principal_count)
            .map(|_| Bucket {
                credit: capacity,
                last: now,
            })
            .collect();
        Ok(Self {
            requests_per_minute: u128::from(limit.requests_per_minute),
            capacity,
            buckets: Mutex::new(buckets),
        })
    }

    fn buckets(&self) -> MutexGuard<'_, Vec<Bucket>> {
        self.buckets
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn check(&self, principal: usize) -> AdmissionDecision {
        self.check_at(principal, Instant::now())
    }

    fn check_at(&self, principal: usize, now: Instant) -> AdmissionDecision {
        let mut buckets = self.buckets();
        // `principal` only comes from the validated token index or anonymous
        // zero. Fail closed if that invariant is ever broken internally.
        let Some(bucket) = buckets.get_mut(principal) else {
            return AdmissionDecision::Reject {
                retry_after_secs: 60,
            };
        };
        let elapsed = now.checked_duration_since(bucket.last).unwrap_or_default();
        let refill = elapsed.as_nanos().saturating_mul(self.requests_per_minute);
        bucket.credit = bucket.credit.saturating_add(refill).min(self.capacity);
        bucket.last = now;
        if bucket.credit >= MINUTE_NANOS {
            bucket.credit -= MINUTE_NANOS;
            AdmissionDecision::Allow
        } else {
            let missing = MINUTE_NANOS - bucket.credit;
            let wait_nanos = missing.div_ceil(self.requests_per_minute);
            let wait_secs = wait_nanos.div_ceil(1_000_000_000).max(1);
            AdmissionDecision::Reject {
                retry_after_secs: u64::try_from(wait_secs).unwrap_or(u64::MAX),
            }
        }
    }
}

fn validate_tokens(tokens: &[String]) -> io::Result<()> {
    if tokens.len() > MAX_BEARER_TOKENS {
        return Err(invalid(format!(
            "at most {MAX_BEARER_TOKENS} bearer tokens may be configured"
        )));
    }
    for (index, token) in tokens.iter().enumerate() {
        if token.is_empty() || token.len() > 4096 {
            return Err(invalid(format!(
                "bearer token {index} must contain 1..=4096 bytes"
            )));
        }
        if token.bytes().any(|byte| !(0x21..=0x7e).contains(&byte)) {
            return Err(invalid(format!(
                "bearer token {index} must contain visible ASCII without spaces"
            )));
        }
        if tokens[..index].iter().any(|prior| prior == token) {
            return Err(invalid("duplicate bearer tokens are not allowed"));
        }
    }
    Ok(())
}

fn constant_time_eq(left: &[u8], right: &[u8]) -> bool {
    let mut diff = left.len() ^ right.len();
    for index in 0..left.len().max(right.len()) {
        diff |= usize::from(
            left.get(index).copied().unwrap_or(0) ^ right.get(index).copied().unwrap_or(0),
        );
    }
    diff == 0
}

fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn rotating_auth_is_bounded_and_position_independent() {
        let admission = Admission::new(
            Some("old"),
            AdmissionPolicy {
                bearer_tokens: vec!["new-a".into(), "new-b".into()],
                rate_limit: None,
            },
        )
        .unwrap();
        assert_eq!(admission.authenticate(Some("new-a")), Some(0));
        assert_eq!(admission.authenticate(Some("new-b")), Some(1));
        assert_eq!(admission.authenticate(Some("old")), Some(2));
        assert_eq!(admission.authenticate(Some("wrong")), None);
        assert_eq!(admission.authenticate(None), None);
    }

    #[test]
    fn token_bucket_refills_with_integer_arithmetic() {
        let limiter = PrincipalLimiter::new(
            PrincipalRateLimit {
                requests_per_minute: 60,
                burst: 2,
            },
            1,
        )
        .unwrap();
        let start = limiter.buckets()[0].last;
        assert_eq!(limiter.check_at(0, start), AdmissionDecision::Allow);
        assert_eq!(limiter.check_at(0, start), AdmissionDecision::Allow);
        assert_eq!(
            limiter.check_at(0, start),
            AdmissionDecision::Reject {
                retry_after_secs: 1
            }
        );
        assert_eq!(
            limiter.check_at(0, start + Duration::from_secs(1)),
            AdmissionDecision::Allow
        );
    }

    #[test]
    fn invalid_policy_fails_before_serving() {
        let duplicate = AdmissionPolicy {
            bearer_tokens: vec!["same".into(), "same".into()],
            rate_limit: None,
        };
        assert!(Admission::new(None, duplicate).is_err());
        let zero_rate = AdmissionPolicy {
            bearer_tokens: vec![],
            rate_limit: Some(PrincipalRateLimit {
                requests_per_minute: 0,
                burst: 1,
            }),
        };
        assert!(Admission::new(None, zero_rate).is_err());
    }

    #[test]
    fn policy_debug_never_exposes_keys() {
        let policy = AdmissionPolicy {
            bearer_tokens: vec!["highly-secret".into()],
            rate_limit: None,
        };
        let rendered = format!("{policy:?}");
        assert!(rendered.contains("bearer_token_count: 1"));
        assert!(!rendered.contains("highly-secret"));
    }
}
