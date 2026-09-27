"""Workspace initialization: statement nodes from the rendered input (decision 0008).

A statement node is one sentence of the rendered input (the question line, each fact
and each rule), represented by the mean of its tokens' contextual states. Nodes keep
their character span, so a trace can name the sentence behind a node.
"""

import re
from dataclasses import dataclass

import torch

# A sentence ends at . ? ! or a line break; the rendered input puts the question and the
# context on separate lines.
_SENTENCE = re.compile(r"[^.?!\n]+[.?!]?")


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """Character spans of the non-blank sentences of ``text``, in order."""
    spans = []
    for match in _SENTENCE.finditer(text):
        start, end = match.span()
        chunk = match.group()
        stripped = chunk.strip()
        if not stripped:
            continue
        start += len(chunk) - len(chunk.lstrip())
        spans.append((start, start + len(stripped)))
    return spans


@dataclass(frozen=True)
class StatementNodes:
    states: torch.Tensor  # [B, S_max, D] mean token state per sentence
    mask: torch.Tensor  # [B, S_max] bool; False for padding and for fully truncated sentences
    texts: list[list[str]]  # sentence text per example (for traces)


def statement_nodes(
    rendered: list[str], states: torch.Tensor, token_mask: torch.Tensor, offsets: torch.Tensor
) -> StatementNodes:
    """Pool token states over each sentence span. A token belongs to a sentence when its
    character span lies inside the sentence's span; special and padding tokens (empty
    offsets) belong to none. Sentences cut off by truncation get an empty mask entry."""
    spans = [sentence_spans(text) for text in rendered]
    width = max((len(s) for s in spans), default=0) or 1
    batch, _, dim = states.shape
    nodes = states.new_zeros(batch, width, dim)
    mask = torch.zeros(batch, width, dtype=torch.bool, device=states.device)
    starts, ends = offsets[..., 0], offsets[..., 1]
    real = token_mask & (ends > starts)
    for i, example in enumerate(spans):
        if not example:
            continue
        bounds = torch.tensor(example, device=states.device)  # [S, 2]
        member = (
            real[i][None, :]
            & (starts[i][None, :] >= bounds[:, :1])
            & (ends[i][None, :] <= bounds[:, 1:])
        )  # [S, L]
        counts = member.sum(1)
        weights = member.to(states.dtype) / counts.clamp_min(1)[:, None].to(states.dtype)
        nodes[i, : len(example)] = weights @ states[i]
        mask[i, : len(example)] = counts > 0
    texts = [[text[a:b] for a, b in example] for text, example in zip(rendered, spans, strict=True)]
    return StatementNodes(states=nodes, mask=mask, texts=texts)
