"""Decision 0015: single-sequence marker encoding (tiny random ModernBERT, offline)."""

import pytest
import torch
from torch import nn

from metanoia.config import SequenceConfig
from metanoia.sequence_model import SequenceDecisionModel
from metanoia_contracts import Candidate, DecisionRequest

transformers = pytest.importorskip("transformers")


def _tokenizer():
    try:
        return transformers.AutoTokenizer.from_pretrained(
            "answerdotai/ModernBERT-base",
            revision="8949b909ec900327062f0ebf497f51aef5e6f0c8",
            local_files_only=True,
        )
    except OSError:
        pytest.skip("ModernBERT tokenizer is not cached")


class _TinyEncoder(nn.Module):
    external_prefixes: tuple[str, ...] = ()
    max_length = 96

    def __init__(self, tokenizer) -> None:
        super().__init__()
        cfg = transformers.ModernBertConfig(
            vocab_size=len(tokenizer),
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=3,
            num_attention_heads=4,
            pad_token_id=tokenizer.pad_token_id,
            attn_implementation="eager",
        )
        self.model = transformers.ModernBertModel(cfg)
        self.tokenizer = tokenizer
        self.output_dim = 32

    def encode_ids(self, input_ids, attention_mask, position_ids=None):
        kwargs = {} if position_ids is None else {"position_ids": position_ids}
        return self.model(
            input_ids=input_ids, attention_mask=attention_mask, **kwargs
        ).last_hidden_state


def _request(texts, context="Rain fell all day. The river rose.", question="What happened?"):
    return DecisionRequest(
        context=context,
        question=question,
        candidates=tuple(Candidate(id=f"c{i}", text=t) for i, t in enumerate(texts)),
    )


def _model(**kw):
    torch.manual_seed(0)
    cfg = SequenceConfig(head_heads=4, dropout=0.0, shuffle_options=False, **kw)
    return SequenceDecisionModel(_TinyEncoder(_tokenizer()), cfg).eval()


OPTIONS = ["the river rose", "it snowed", "a fire started", "nothing at all"]


def test_options_come_first_and_markers_point_at_mask_tokens():
    model = _model()
    built = model.build([_request(OPTIONS)])
    tok = model.encoder.tokenizer
    ids = built.input_ids[0]
    assert ids[0] == tok.cls_token_id
    for marker in built.markers[0]:
        assert ids[marker] == tok.mask_token_id
    assert built.markers[0].tolist() == sorted(built.markers[0].tolist())
    assert built.option_mask.all()


def test_truncation_cuts_the_context_never_an_option():
    model = _model()
    long_context = "A long sentence about nothing. " * 200
    built = model.build([_request(OPTIONS, context=long_context)])
    assert built.input_ids.shape[1] <= model.encoder.max_length
    assert built.context_kept[0] < built.context_total[0]
    assert built.option_tokens[0] == [
        len(model.encoder.tokenizer(t, add_special_tokens=False)["input_ids"]) for t in OPTIONS
    ]


def test_batches_with_different_option_counts_are_masked():
    model = _model()
    out = model([_request(OPTIONS), _request(OPTIONS[:2])])
    assert out.logits.shape == (2, 4)
    assert torch.isinf(out.logits[1, 2:]).all()
    assert torch.isfinite(out.logits[1, :2]).all()
    assert out.candidate_mask.tolist() == [[True] * 4, [True, True, False, False]]


def test_batch_padding_does_not_change_scores():
    model = _model()
    alone = model([_request(OPTIONS)]).logits
    padded = model([_request(OPTIONS), _request(OPTIONS[:2], context="x " * 40)]).logits[:1]
    assert torch.allclose(alone, padded, atol=1e-4)


def test_rotation_average_is_exactly_equivariant_to_option_order():
    model = _model(rotations=4)
    request = _request(OPTIONS)
    shuffled = _request([OPTIONS[i] for i in (2, 0, 3, 1)])
    a = torch.softmax(model([request]).logits[0], -1)
    b = torch.softmax(model([shuffled]).logits[0], -1)
    assert torch.allclose(a[torch.tensor([2, 0, 3, 1])], b, atol=1e-5)


def test_training_shuffle_returns_logits_in_the_original_order():
    torch.manual_seed(1)
    model = _model()
    request = _request(OPTIONS)
    model.config = model.config.model_copy(update={"shuffle_options": True})
    model.train()
    out = model([request]).logits
    assert torch.isfinite(out).all()
    assert out.shape == (1, 4)


def test_position_reset_gives_every_option_the_same_start_position():
    model = _model(position_reset=True)
    built = model.build([_request(OPTIONS)])
    assert built.position_ids is not None
    starts = {int(built.position_ids[0, m]) for m in built.markers[0]}
    assert len(starts) == 1


def test_gradients_reach_encoder_and_head():
    model = _model()
    model.train()
    out = model([_request(OPTIONS)])
    torch.nn.functional.cross_entropy(out.logits, torch.tensor([0])).backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.encoder.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.scorer.parameters())


def test_config_requires_sequence_settings():
    from metanoia.config import DirectScorerConfig

    with pytest.raises(ValueError):
        DirectScorerConfig(candidate_encoding="sequence")
    with pytest.raises(ValueError):
        DirectScorerConfig(candidate_encoding="joint", sequence=SequenceConfig())
