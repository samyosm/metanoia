from metanoia.api import Predictor
from metanoia.baselines.lexical import LexicalOverlapPredictor
from metanoia_contracts import Candidate, DecisionRequest


def test_lexical_fixture_is_deterministic_and_implements_protocol() -> None:
    predictor = LexicalOverlapPredictor()
    assert isinstance(predictor, Predictor)
    request = DecisionRequest(
        context="the cat sat on the cat mat",
        question="which?",
        candidates=(Candidate(id="c", text="cat"), Candidate(id="d", text="dog")),
    )
    first, second = predictor.decide(request), predictor.decide_batch([request])[0]
    assert first == second
    assert first.logits == (2.0, 0.0)
