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
        assert len(transactions) == len(sampled) == 1
        assert max_context >= len(transactions[0])
        return [
            SimpleNamespace(
                shifted_input_ids=[sampled[0]],
                target_hidden_states=[0.5, -0.5],
                hidden_size=2,
                last_logits=[0.25, 0.75],
                final_hidden_states=[0.5, -0.5],
            )
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
    def draft(shifted, hidden):
        return SimpleNamespace(
            logits=torch.tensor([[[0.25, 0.75]]], dtype=torch.float32),
            final_hidden=hidden.unsqueeze(0),
            past_key_values=(shifted.to(dtype=torch.float32),),
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
    assert [case["kind"] for case in mtp] == ["mtp", "mtp"]
    assert all(case["max_abs_error"] == 0 for case in language + mtp)
    assert all(
        case["token_ids_exact"] and case["states_exact"] and case["output_exact"]
        for case in language + mtp
    )


def test_unpromoted_mtp_blocks_whole_model_evidence():
    native = SimpleNamespace(mtp_verified=False)
    with pytest.raises(qualify_onnx.OnnxQualificationError, match="not promoted"):
        qualify_onnx._require_mtp_oracle(native)
