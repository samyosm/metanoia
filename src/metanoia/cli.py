"""``metanoia``: ask a trained predictor bundle for candidate probabilities.

``decide`` answers one request. ``serve`` keeps a bundle loaded and answers requests
as JSON lines on stdin/stdout, for a local front end such as ``../interface``.

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

from metanoia_contracts import Budget, Candidate, DecisionRequest, DecisionResult, TraceMode
from metanoia_contracts.base import load_strict


def _candidate_id(text: str, taken: set[str]) -> str:
    base = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:40] or "candidate"
    cid, k = base, 2
    while cid in taken:
        cid, k = f"{base}_{k}", k + 1
    taken.add(cid)
    return cid


def reasoning_budget(rounds: int) -> Budget:
    """The budget ``reasoning_v1`` revision 2 uses for ``rounds`` workspace rounds."""
    if rounds <= 0:
        return Budget()
    return Budget(max_rounds=rounds, max_nodes=64, max_pairs=rounds * 4096)


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
    rounds = getattr(args, "rounds", 0) or 0
    return DecisionRequest(
        budget=reasoning_budget(rounds),
        context=context,
        question=args.question,
        candidates=tuple(Candidate(id=_candidate_id(t, taken), text=t) for t in args.candidate),
        trace=TraceMode.EVENTS if args.trace else TraceMode.OFF,
    )


def _load(args: argparse.Namespace):  # type: ignore[no-untyped-def]
    from metanoia.api import load_predictor

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
    return load_predictor(
        Path(args.bundle),
        calibration=Path(args.calibration) if args.calibration else None,
        device=device,
        precision=args.precision,
    )


def decide(args: argparse.Namespace) -> None:
    request = build_request(args)
    result = _load(args).decide(request)
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
    if result.usage.rounds or request.budget.max_rounds:
        used, wanted = result.usage.rounds, request.budget.max_rounds
        print(f"rounds: {used}/{wanted}  ({result.termination.value})")


def serve(args: argparse.Namespace) -> None:
    """JSON-lines worker. Each input line is ``{"id": ..., "request": DecisionRequest}``;
    each output line is ``{"id": ..., "result": DecisionResult}`` or
    ``{"id": ..., "error": "..."}``. The first output line announces readiness.
    With ``"stream": true`` the result is preceded by ``{"id": ..., "update":
    DecisionUpdate}`` lines, one per computed stage. Only stdout carries protocol
    lines; diagnostics go to stderr."""
    import time

    out = sys.stdout
    predictor = _load(args)
    ready = {"ready": True, "predictor": predictor.describe()}
    out.write(json.dumps(ready) + "\n")
    out.flush()
    for line in sys.stdin:
        if not line.strip():
            continue
        message_id = None
        try:
            message = json.loads(line)
            message_id = message.get("id")
            request = load_strict(DecisionRequest, message["request"])
            start = time.perf_counter()
            if message.get("stream"):
                for item in predictor.decide_stream(request):
                    if isinstance(item, DecisionResult):
                        result = item
                    else:
                        update = {"id": message_id, "update": item.model_dump(mode="json")}
                        out.write(json.dumps(update) + "\n")
                        out.flush()
            else:
                result = predictor.decide(request)
            reply = {
                "id": message_id,
                "result": result.model_dump(mode="json"),
                "wall_time_ms": (time.perf_counter() - start) * 1000.0,
            }
        except Exception as error:  # noqa: BLE001 - reported to the caller, worker lives on
            reply = {"id": message_id, "error": f"{type(error).__name__}: {error}"}
        out.write(json.dumps(reply) + "\n")
        out.flush()


def _model_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--bundle", required=True, help="predictor bundle directory")
    p.add_argument("--calibration", help="fitted calibration JSON for this predictor")
    p.add_argument("--device", default="auto", help="cpu, cuda, or auto (default)")
    p.add_argument("--precision", default="fp32", choices=["fp32", "fp16"])


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="metanoia", description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("decide", help="candidate probabilities for one request")
    _model_arguments(p)
    p.add_argument("--context", help="context text")
    p.add_argument("--context-file", help="read the context from a file")
    p.add_argument("--question")
    p.add_argument("--candidate", action="append", help="repeat once per candidate")
    p.add_argument("--request", help="DecisionRequest JSON file, or - for stdin")
    p.add_argument(
        "--rounds",
        type=int,
        default=0,
        help="workspace reasoning rounds (workspace models; 0 = direct scorer)",
    )
    p.add_argument("--trace", action="store_true", help="include trace events")
    p.add_argument("--json", action="store_true", help="print the DecisionResult as JSON")
    p.set_defaults(func=decide)
    p = commands.add_parser("serve", help="answer JSON-line requests on stdin/stdout")
    _model_arguments(p)
    p.set_defaults(func=serve)
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except (ValueError, FileNotFoundError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
