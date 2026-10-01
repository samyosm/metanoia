"""Single-sequence marker rendering (decision 0015).

    [CLS] question: Q [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] context [SEP]

The question and options form the *head*, with a token budget; the context fills what is
left, so a long context is truncated and an option never is. Each option's first token is
the tokenizer's ``[MASK]``; its position is the option's *marker*, where the score is read.
"""

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

from metanoia_contracts import DecisionRequest

SEQUENCE_REVISION = "marker-sequence/1"


@dataclass(frozen=True)
class BuiltSequences:
    input_ids: torch.Tensor  # [B, L] long
    attention_mask: torch.Tensor  # [B, L] long
    position_ids: torch.Tensor | None  # [B, L] long; None = default positions
    markers: torch.Tensor  # [B, N_max] long; token index of each option's marker
    option_mask: torch.Tensor  # [B, N_max] bool
    # Per request: tokens of the context kept, and of the full context (to detect truncation).
    context_kept: list[int]
    context_total: list[int]
    option_tokens: list[list[int]]  # tokens kept per option (after the per-option cap)


def context_text(request: DecisionRequest) -> str:
    text = request.context
    if request.observations:
        items = "; ".join(f"{k} = {request.observations[k]}" for k in sorted(request.observations))
        text = f"{text}\nobservations: {items}"
    return text


def build_sequences(
    tokenizer: Any,
    requests: Sequence[DecisionRequest],
    *,
    max_length: int,
    head_max_len: int,
    option_max_tokens: int,
    position_reset: bool,
    position_hash: bool = False,
    position_hash_range: int = 512,
) -> BuiltSequences:
    cls_id, sep_id = tokenizer.cls_token_id, tokenizer.sep_token_id
    mask_id, pad_id = tokenizer.mask_token_id, tokenizer.pad_token_id
    rows: list[list[int]] = []
    positions: list[list[int]] = []
    marker_rows: list[list[int]] = []
    kept, total, opt_tokens = [], [], []

    def ids(text: str) -> list[int]:
        return tokenizer(text, add_special_tokens=False)["input_ids"]

    for request in requests:
        n = len(request.candidates)
        per_option = min(option_max_tokens, max(4, (head_max_len - 16) // n))
        options = [ids(c.text)[:per_option] for c in request.candidates]
        option_cost = sum(len(o) + 1 for o in options)  # a marker per option
        question = ids(f"question: {request.question}")
        question = question[: max(8, head_max_len - option_cost - 3)]

        head = [cls_id, *question, sep_id]
        seq, pos, markers = list(head), list(range(len(head))), []
        start = len(head)
        longest = max(len(o) for o in options) + 1
        for option, candidate in zip(options, request.candidates, strict=True):
            markers.append(len(seq))
            segment = [mask_id, *option]
            seq += segment
            if position_hash:
                offset = int(hashlib.sha256(candidate.text.encode()).hexdigest(), 16)
                first = start + offset % position_hash_range
                pos += list(range(first, first + len(segment)))
            elif position_reset:
                pos += list(range(start, start + len(segment)))
            else:
                pos += list(range(len(seq) - len(segment), len(seq)))
        seq.append(sep_id)
        if position_hash:
            pos.append(start + position_hash_range + option_max_tokens + 1)
        else:
            pos.append(start + longest if position_reset else len(seq) - 1)

        context = ids(context_text(request))
        room = max(0, max_length - len(seq) - 1)
        kept_context = context[:room]
        seq += kept_context + [sep_id]
        base = pos[-1] + 1
        pos += list(range(base, base + len(kept_context) + 1))

        rows.append(seq)
        positions.append(pos)
        marker_rows.append(markers)
        kept.append(len(kept_context))
        total.append(len(context))
        opt_tokens.append([len(o) for o in options])

    width = max(len(r) for r in rows)
    n_max = max(len(m) for m in marker_rows)
    input_ids = torch.full((len(rows), width), pad_id, dtype=torch.long)
    attention = torch.zeros(len(rows), width, dtype=torch.long)
    pos_ids = torch.zeros(len(rows), width, dtype=torch.long)
    markers_t = torch.zeros(len(rows), n_max, dtype=torch.long)
    option_mask = torch.zeros(len(rows), n_max, dtype=torch.bool)
    for b, (row, pos, markers) in enumerate(zip(rows, positions, marker_rows, strict=True)):
        input_ids[b, : len(row)] = torch.tensor(row)
        attention[b, : len(row)] = 1
        pos_ids[b, : len(row)] = torch.tensor(pos)
        markers_t[b, : len(markers)] = torch.tensor(markers)
        option_mask[b, : len(markers)] = True
    return BuiltSequences(
        input_ids=input_ids,
        attention_mask=attention,
        position_ids=pos_ids if (position_reset or position_hash) else None,
        markers=markers_t,
        option_mask=option_mask,
        context_kept=kept,
        context_total=total,
        option_tokens=opt_tokens,
    )
