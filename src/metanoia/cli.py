"""``metanoia decide``: ask a trained predictor bundle for candidate probabilities.

Thin wrapper around :func:`metanoia.api.load_predictor`. Inputs come either from flags
or from a JSON request file (the public :class:`DecisionRequest` schema); the output is
the public :class:`DecisionResult` as JSON, or a readable table.

    metanoia decide --bundle artifacts/predictors/NAME \\
        --context "..." --question "..." --candidate "Paris" --candidate "Rome"
"""

import argparse
import json
import re
import sys
from pathlib import Path

from metanoia_contracts import Candidate, DecisionRequest, TraceMode
from metanoia_contracts.base import load_strict


def _candidate_id(text: str, taken: set[str]) -> str:
    base = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:40] or "candidate"
    cid, k = base, 2
    while cid in taken:
        cid, k = f"{base}_{k}", k + 1
    taken.add(cid)
    return cid


def build_request(args: argparse.Namespace) -> DecisionRequest:
    if args.request:
        raw = sys.stdin.read() if args.request == "-" else Path(args.request).read_text()
        return load_strict(DecisionRequest, json.loads(raw))
    if not args.question or not args.candidate or len(args.candidate) < 2:
        raise ValueError("give --question and at least two --candidate values, or --request")
    context = args.context or ""
    if args.context_file:
        context = Path(args.context_file).read_text()
    taken: set[str] = set()
    return DecisionRequest(
        context=context,
        question=args.question,
        candidates=tuple(Candidate(id=_candidate_id(t, taken), text=t) for t in args.candidate),
        trace=TraceMode.EVENTS if args.trace else TraceMode.OFF,
    )


def decide(args: argparse.Namespace) -> None:
    from metanoia.api import load_predictor

    request = build_request(args)
    try:  # Quiet the pretrained-weights load report; it is not part of the answer.
        from transformers.utils import logging as hf_logging

        hf_logging.set_verbosity_error()
        hf_logging.disable_progress_bar()
    except ImportError:
        pass
    device = args.device
    if device == "auto":
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = load_predictor(
        Path(args.bundle),
        calibration=Path(args.calibration) if args.calibration else None,
        device=device,
        precision=args.precision,
    )
    result = predictor.decide(request)
    if args.json:
        print(json.dumps(result.model_dump(mode="json"), indent=2))
        return
    texts = {c.id: c.text for c in request.candidates}
    rows = sorted(
        zip(result.candidate_ids, result.probabilities, result.logits, strict=True),
        key=lambda row: -row[1],
    )
    width = max(len(texts[cid]) for cid in result.candidate_ids)
    for cid, p, logit in rows:
        bar = "#" * round(40 * p)
        print(f"{texts[cid]:<{width}}  {p:7.2%}  logit {logit:+7.3f}  {bar}")
    print(f"calibration: {result.calibration.status.value}  predictor: {result.predictor_id}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="metanoia", description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("decide", help="candidate probabilities for one request")
    p.add_argument("--bundle", required=True, help="predictor bundle directory")
    p.add_argument("--calibration", help="fitted calibration JSON for this predictor")
    p.add_argument("--context", help="context text")
    p.add_argument("--context-file", help="read the context from a file")
    p.add_argument("--question")
    p.add_argument("--candidate", action="append", help="repeat once per candidate")
    p.add_argument("--request", help="DecisionRequest JSON file, or - for stdin")
    p.add_argument("--device", default="auto", help="cpu, cuda, or auto (default)")
    p.add_argument("--precision", default="fp32", choices=["fp32", "fp16"])
    p.add_argument("--trace", action="store_true", help="include trace events")
    p.add_argument("--json", action="store_true", help="print the DecisionResult as JSON")
    p.set_defaults(func=decide)
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except (ValueError, FileNotFoundError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
