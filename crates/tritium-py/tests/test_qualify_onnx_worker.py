from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from tritium.torch import qualify_onnx  # noqa: E402


class _Native:
    mtp_verified = True

    @staticmethod
    def reference_language(transactions, max_context):
        assert max_context >= sum(len(value) for value in transactions)
        return [
            SimpleNamespace(
                token_ids=list(tokens),
                last_logits=[0.25, 0.75],
                final_hidden_states=[0.5, -0.5] * len(tokens),
                hidden_size=2,
            )
            for tokens in transactions
        ]

    @staticmethod
    def generate(prompt, max_new_tokens):
        return [1] * max_new_tokens

    @staticmethod
    def reference_mtp(transactions, sampled, max_context):
        assert len(transactions) == len(sampled)
        assert max_context >= sum(map(len, transactions))
        return [
            SimpleNamespace(
                shifted_input_ids=list(tokens[1:]) + [next_token],
                target_hidden_states=[0.5, -0.5] * len(tokens),
                hidden_size=2,
                last_logits=[0.25, 0.75],
                final_hidden_states=[0.5, -0.5] * len(tokens),
            )
            for tokens, next_token in zip(transactions, sampled, strict=True)
        ]


class _Ort:
    @staticmethod
    def __call__(tokens, past_key_values=None):
        del past_key_values
        return SimpleNamespace(
            logits=torch.tensor([[[0.25, 0.75]]], dtype=torch.float32),
            past_key_values=(tokens.to(dtype=torch.float32),),
        )

    @staticmethod
    def generate(tokens, max_new_tokens):
        suffix = torch.ones((1, max_new_tokens), dtype=torch.int64)
        return torch.cat((tokens, suffix), dim=1)

    @staticmethod
    def draft(shifted, hidden, past_key_values=None):
        state = shifted.to(dtype=torch.float32)
        if past_key_values is not None:
            state = torch.cat((past_key_values[0], state), dim=1)
        return SimpleNamespace(
            logits=torch.tensor([[[0.25, 0.75]]], dtype=torch.float32),
            final_hidden=hidden.unsqueeze(0).clone(),
            past_key_values=(state,),
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
                result.past_key_values = (result.past_key_values[0] + 1,)
            return result

    cases = qualify_onnx._mtp_cases(_Native(), ReplayDriftedOrt())
    assert all(not case["states_exact"] for case in cases)


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

    cases = qualify_onnx._language_cases(_Native(), InvalidCacheOrt())
    assert all(
        not case["states_exact"]
        for case in cases
        if case["kind"] in {"prompt", "cached-decode"}
    )
