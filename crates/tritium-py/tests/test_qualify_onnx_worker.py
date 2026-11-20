from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from tritium.torch import qualify_onnx  # noqa: E402


class _Native:
    mtp_verified = True

    @staticmethod
    def reference_language(transactions, max_context, *, include_states=False,
                           max_state_bytes=268435456, state_steps=None):
        assert max_context >= sum(len(value) for value in transactions)
        outputs = []
        committed = []
        for index, tokens in enumerate(transactions):
            committed.extend(tokens)
            selected = include_states and (state_steps is None or index in state_steps)
            outputs.append(SimpleNamespace(
                token_ids=list(tokens),
                last_logits=[0.25, 0.75],
                final_hidden_states=[0.5, -0.5] * len(tokens),
                hidden_size=2,
                state_names=["present_k.0"] if selected else [],
                state_shapes=[[1, len(committed)]] if selected else [],
                states=[list(map(float, committed))] if selected else [],
            ))
        assert not include_states or max_state_bytes >= sum(
            len(output.states[0]) * 4 for output in outputs if output.states
        )
        return outputs

    @staticmethod
    def generate(prompt, max_new_tokens):
        return [1] * max_new_tokens

    @staticmethod
    def reference_mtp(transactions, sampled, max_context):
        assert len(transactions) == len(sampled)
        assert max_context >= sum(map(len, transactions))
        outputs = []
        committed = []
        for tokens, next_token in zip(transactions, sampled, strict=True):
            shifted = list(tokens[1:]) + [next_token]
            committed.extend(shifted)
            outputs.append(SimpleNamespace(
                shifted_input_ids=shifted,
                target_hidden_states=[0.5, -0.5] * len(tokens),
                hidden_size=2,
                last_logits=[0.25, 0.75],
                final_hidden_states=[0.5, -0.5] * len(tokens),
                state_names=["present_k.0", "present_v.0"],
                state_shapes=[[len(committed), 1, 1]] * 2,
                states=[list(map(float, committed)),
                        [float(value) + 0.5 for value in committed]],
            ))
        return outputs


class _Ort:
    @staticmethod
    def __call__(tokens, past_key_values=None):
        state = tokens.to(dtype=torch.float32)
        if past_key_values is not None:
            state = torch.cat((past_key_values[0], state), dim=1)
        return SimpleNamespace(
            logits=torch.tensor([[[0.25, 0.75]]], dtype=torch.float32),
            past_key_values=(state,),
            state_names=("present_k.0",),
        )

    @staticmethod
    def generate(tokens, max_new_tokens):
        suffix = torch.ones((1, max_new_tokens), dtype=torch.int64)
        return torch.cat((tokens, suffix), dim=1)

    @staticmethod
    def draft(shifted, hidden, past_key_values=None):
        state = shifted.to(dtype=torch.float32).reshape(-1, 1, 1)
        value = state + 0.5
        if past_key_values is not None:
            state = torch.cat((past_key_values[0], state), dim=0)
            value = torch.cat((past_key_values[1], value), dim=0)
        return SimpleNamespace(
            logits=torch.tensor([[[0.25, 0.75]]], dtype=torch.float32),
            final_hidden=hidden.unsqueeze(0).clone(),
            past_key_values=(state, value),
            state_names=("present_k.0", "present_v.0"),
        )


def test_language_and_mtp_cases_are_execution_derived():
    language = qualify_onnx._language_cases(_Native(), _Ort())
    mtp = qualify_onnx._mtp_cases(_Native(), _Ort())
    assert [case["kind"] for case in language] == [
        "prompt",
        "cached-decode",
        "generation",
        "prompt",
        "cached-decode",
        "generation",
    ]
    assert [case["kind"] for case in mtp] == ["mtp"] * 4
    assert [case["case_id"] for case in mtp] == [
        "mtp-prefill-0", "mtp-cached-decode-0",
        "mtp-prefill-1", "mtp-cached-decode-1",
    ]
    assert all(case["max_abs_error"] == 0 for case in language + mtp)
    assert all(
        case["token_ids_exact"] and case["states_exact"] and case["output_exact"]
        for case in language + mtp
    )


def test_unpromoted_mtp_blocks_whole_model_evidence():
    native = SimpleNamespace(mtp_verified=False)
    with pytest.raises(qualify_onnx.OnnxQualificationError, match="not promoted"):
        qualify_onnx._require_mtp_oracle(native)


def test_mtp_decode_consumes_the_actual_prefill_cache():
    class RecordingOrt(_Ort):
        def __init__(self):
            self.calls = []

        def draft(self, shifted, hidden, past_key_values=None):
            result = super().draft(shifted, hidden, past_key_values)
            self.calls.append((shifted.clone(), past_key_values, result))
            return result

    runtime = RecordingOrt()
    qualify_onnx._mtp_cases(_Native(), runtime)
    assert len(runtime.calls) == 8  # observed and replay prefill/decode, two prompts
    for offset in (0, 4):
        assert runtime.calls[offset][1] is None
        assert runtime.calls[offset + 1][1] is None
        assert runtime.calls[offset + 2][1] is runtime.calls[offset][2].past_key_values
        assert runtime.calls[offset + 3][1] is runtime.calls[offset + 1][2].past_key_values
        assert runtime.calls[offset + 2][0].shape == (1, 1)


def test_mtp_detects_decode_only_numeric_drift():
    class DriftedOrt(_Ort):
        @staticmethod
        def draft(shifted, hidden, past_key_values=None):
            result = _Ort.draft(shifted, hidden, past_key_values)
            if past_key_values is not None:
                result.logits += 0.5
                result.final_hidden += 0.125
            return result

    cases = qualify_onnx._mtp_cases(_Native(), DriftedOrt())
    assert [case["max_abs_error"] for case in cases] == [0.0, 0.5, 0.0, 0.5]


@pytest.mark.parametrize("field", ["shifted_input_ids", "target_hidden_states"])
def test_mtp_rejects_misaligned_oracle_inputs(field):
    class MisalignedNative(_Native):
        @staticmethod
        def reference_mtp(transactions, sampled, max_context):
            results = _Native.reference_mtp(transactions, sampled, max_context)
            getattr(results[0], field)[0] += 1
            return results

    with pytest.raises(qualify_onnx.OnnxQualificationError, match="alignment"):
        qualify_onnx._mtp_cases(MisalignedNative(), _Ort())


@pytest.mark.parametrize("state", [
    (),
    (torch.tensor([[float("inf")]]),),
    (torch.tensor([[float("nan")]]),),
    (torch.empty((1, 0)),),
    (torch.ones((1, 1), dtype=torch.float64),),
])
def test_mtp_rejects_missing_or_nonfinite_cache(state):
    class InvalidCacheOrt(_Ort):
        @staticmethod
        def draft(shifted, hidden, past_key_values=None):
            result = _Ort.draft(shifted, hidden, past_key_values)
            result.past_key_values = state
            return result

    with pytest.raises(qualify_onnx.OnnxQualificationError, match="cache"):
        qualify_onnx._mtp_cases(_Native(), InvalidCacheOrt())


@pytest.mark.parametrize("field,value", [
    ("hidden_size", True),
    ("hidden_size", 0),
    ("hidden_size", 3),
    ("target_hidden_states", [0.5]),
    ("final_hidden_states", [0.5]),
])
def test_mtp_rejects_malformed_oracle_geometry(field, value):
    class MalformedNative(_Native):
        @staticmethod
        def reference_mtp(transactions, sampled, max_context):
            results = _Native.reference_mtp(transactions, sampled, max_context)
            setattr(results[0], field, value)
            return results

    with pytest.raises(qualify_onnx.OnnxQualificationError, match="alignment"):
        qualify_onnx._mtp_cases(MalformedNative(), _Ort())


def test_mtp_rejects_missing_decode_oracle_transaction():
    class ShortNative(_Native):
        @staticmethod
        def reference_mtp(transactions, sampled, max_context):
            return _Native.reference_mtp(transactions, sampled, max_context)[:1]

    with pytest.raises(qualify_onnx.OnnxQualificationError, match="transaction count"):
        qualify_onnx._mtp_cases(ShortNative(), _Ort())


def test_mtp_detects_replay_only_numeric_drift():
    class ReplayDriftedOrt(_Ort):
        def __init__(self):
            self.calls = 0

        def draft(self, shifted, hidden, past_key_values=None):
            result = super().draft(shifted, hidden, past_key_values)
            self.calls += 1
            if self.calls % 2 == 0:
                result.final_hidden += 0.125
            return result

    cases = qualify_onnx._mtp_cases(_Native(), ReplayDriftedOrt())
    assert all(case["max_abs_error"] == 0.125 for case in cases)


def test_mtp_detects_replay_cache_drift():
    class ReplayDriftedOrt(_Ort):
        def __init__(self):
            self.calls = 0

        def draft(self, shifted, hidden, past_key_values=None):
            result = super().draft(shifted, hidden, past_key_values)
            self.calls += 1
            if self.calls % 2 == 0:
                result.past_key_values = tuple(state + 1 for state in result.past_key_values)
            return result

    cases = qualify_onnx._mtp_cases(_Native(), ReplayDriftedOrt())
    assert all(not case["states_exact"] for case in cases)


@pytest.mark.parametrize("phase", ["prefill", "cached-decode"])
@pytest.mark.parametrize("arm", ["observed", "replay", "both"])
@pytest.mark.parametrize("slot", [0, 1])
def test_mtp_native_cache_error_covers_both_arms_and_both_kv_slots(phase, arm, slot):
    class CacheDriftedOrt(_Ort):
        def __init__(self):
            self.calls = 0

        def draft(self, shifted, hidden, past_key_values=None):
            result = super().draft(shifted, hidden, past_key_values)
            current_arm = "observed" if self.calls % 2 == 0 else "replay"
            self.calls += 1
            current_phase = "prefill" if past_key_values is None else "cached-decode"
            if current_phase == phase and arm in (current_arm, "both"):
                states = list(result.past_key_values)
                states[slot] = states[slot] + 0.125
                result.past_key_values = tuple(states)
            return result

    cases = qualify_onnx._mtp_cases(_Native(), CacheDriftedOrt())
    for case in cases:
        if phase in case["case_id"]:
            assert case["max_abs_error"] == 0.125
            assert case["tolerance"] == 1e-3
            # An identical wrong replay is still disqualified by native values.
            if arm == "both":
                assert case["states_exact"] is True
            assert case["token_ids_exact"] and case["output_exact"]


@pytest.mark.parametrize("field,value", [
    ("state_names", []),
    ("state_names", ["present_v.0", "present_k.0"]),
    ("state_shapes", [[1, 1, 1], [1, 1, 1]]),
    ("states", []),
    ("states", [[float("nan")], [0.0]]),
])
def test_mtp_rejects_missing_or_malformed_native_cache(field, value):
    class MalformedCacheNative(_Native):
        @staticmethod
        def reference_mtp(transactions, sampled, max_context):
            results = _Native.reference_mtp(transactions, sampled, max_context)
            setattr(results[0], field, value)
            return results

    with pytest.raises(qualify_onnx.OnnxQualificationError, match="cache"):
        qualify_onnx._mtp_cases(MalformedCacheNative(), _Ort())


def test_mtp_rejects_reordered_onnx_cache_inventory_even_when_values_match():
    class ReorderedOrt(_Ort):
        @staticmethod
        def draft(shifted, hidden, past_key_values=None):
            result = _Ort.draft(shifted, hidden, past_key_values)
            result.state_names = tuple(reversed(result.state_names))
            return result

    with pytest.raises(qualify_onnx.OnnxQualificationError, match="cache inventory"):
        qualify_onnx._mtp_cases(_Native(), ReorderedOrt())


def test_mtp_rejects_matching_but_incomplete_native_and_onnx_cache_inventories():
    class MissingValueNative(_Native):
        @staticmethod
        def reference_mtp(transactions, sampled, max_context):
            results = _Native.reference_mtp(transactions, sampled, max_context)
            for result in results:
                result.state_names = result.state_names[:1]
                result.state_shapes = result.state_shapes[:1]
                result.states = result.states[:1]
            return results

    class MissingValueOrt(_Ort):
        @staticmethod
        def draft(shifted, hidden, past_key_values=None):
            result = _Ort.draft(shifted, hidden, past_key_values)
            result.state_names = result.state_names[:1]
            result.past_key_values = result.past_key_values[:1]
            return result

    with pytest.raises(qualify_onnx.OnnxQualificationError, match="cache inventory"):
        qualify_onnx._mtp_cases(MissingValueNative(), MissingValueOrt())


@pytest.mark.parametrize("state", [
    (),
    (torch.tensor([[float("inf")]]),),
    (torch.ones((1, 1), dtype=torch.float64),),
])
def test_language_replay_does_not_certify_invalid_cache(state):
    class InvalidCacheOrt(_Ort):
        @staticmethod
        def __call__(tokens, past_key_values=None):
            result = _Ort.__call__(tokens, past_key_values)
            result.past_key_values = state
            return result

    with pytest.raises(qualify_onnx.OnnxQualificationError, match="cache"):
        qualify_onnx._language_cases(_Native(), InvalidCacheOrt())


def test_native_cache_comparison_measures_drift_despite_identical_logits_and_replay():
    reference = _Native.reference_language([[1, 2]], 3, include_states=True)[0]
    observed = _Ort()(torch.tensor([[1, 2]]))
    observed.past_key_values = (observed.past_key_values[0] + 0.125,)
    assert qualify_onnx._native_state_error(reference, observed) == 0.125


@pytest.mark.parametrize("field,value", [
    ("state_names", []),
    ("state_names", ["present_v.0"]),
    ("state_names", [True]),
    ("state_shapes", [[2, 1]]),
    ("state_shapes", [[True, 2]]),
    ("states", []),
    ("states", [[float("nan"), 2.0]]),
    ("states", [[1.0]]),
])
def test_native_cache_comparison_rejects_missing_or_malformed_observations(field, value):
    reference = _Native.reference_language([[1, 2]], 3, include_states=True)[0]
    setattr(reference, field, value)
    observed = _Ort()(torch.tensor([[1, 2]]))
    with pytest.raises(qualify_onnx.OnnxQualificationError, match="cache"):
        qualify_onnx._native_state_error(reference, observed)


def test_language_cases_include_native_decode_cache_error_in_frozen_tolerance():
    class DriftedOrt(_Ort):
        @staticmethod
        def __call__(tokens, past_key_values=None):
            result = _Ort.__call__(tokens, past_key_values)
            if past_key_values is not None:
                result.past_key_values = (result.past_key_values[0] + 0.125,)
            return result

    cases = qualify_onnx._language_cases(_Native(), DriftedOrt())
    assert [case["max_abs_error"] for case in cases] == [0, 0.125, 0.125] * 2
    assert all(case["states_exact"] for case in cases)  # replay is exact, native isn't
    assert all(case["tolerance"] == 1e-3 for case in cases)


def test_language_worker_selects_decode_only_to_fit_the_qwen_snapshot_budget():
    class RecordingNative(_Native):
        calls = []

        @classmethod
        def reference_language(cls, transactions, max_context, **kwargs):
            cls.calls.append((transactions, kwargs))
            return _Native.reference_language(transactions, max_context, **kwargs)

    qualify_onnx._language_cases(RecordingNative(), _Ort())
    assert len(RecordingNative.calls) == 4
    for transactions, kwargs in RecordingNative.calls:
        assert kwargs["include_states"] is True
        if len(transactions) == 2:
            assert kwargs["state_steps"] == [1]
        else:
            assert "state_steps" not in kwargs
