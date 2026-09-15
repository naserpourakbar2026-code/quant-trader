from src.robustness.score import RobustnessScore, RobustnessStatus
from src.robustness.selection import select_top_strategies


def _score(strategy, status, score):
    return RobustnessScore(strategy=strategy, symbol="EURUSD", timeframe="H1", status=status, score=score)


def test_no_robust_strategy_found_when_nothing_passes():
    evaluations = [_score("a", RobustnessStatus.FAIL, None), _score("b", RobustnessStatus.NOT_ROBUST, 40.0)]
    selection = select_top_strategies(evaluations)
    assert selection.found_any is False
    assert selection.medalists == []
    assert "NO ROBUST STRATEGY FOUND" in selection.to_text()


def test_ranks_passing_strategies_by_score_descending():
    evaluations = [
        _score("low", RobustnessStatus.PASS, 65.0),
        _score("high", RobustnessStatus.PASS, 95.0),
        _score("mid", RobustnessStatus.PASS, 80.0),
    ]
    selection = select_top_strategies(evaluations)
    assert [e.strategy for e in selection.medalists] == ["high", "mid", "low"]


def test_caps_at_top_n_even_with_more_passing_candidates():
    evaluations = [_score(f"s{i}", RobustnessStatus.PASS, float(i)) for i in range(5)]
    selection = select_top_strategies(evaluations, top_n=3)
    assert len(selection.medalists) == 3
    assert [e.strategy for e in selection.medalists] == ["s4", "s3", "s2"]


def test_non_passing_statuses_are_excluded_even_with_a_high_score():
    evaluations = [
        _score("fragile", RobustnessStatus.NOT_ROBUST, 99.0),
        _score("real", RobustnessStatus.PASS, 61.0),
    ]
    selection = select_top_strategies(evaluations)
    assert [e.strategy for e in selection.medalists] == ["real"]


def test_to_text_includes_medals_for_each_selected_strategy():
    evaluations = [_score("a", RobustnessStatus.PASS, 90.0), _score("b", RobustnessStatus.PASS, 80.0)]
    text = select_top_strategies(evaluations).to_text()
    assert "🥇" in text
    assert "🥈" in text
    assert "a" in text and "b" in text
