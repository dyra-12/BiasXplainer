import sys
import pathlib
import pytest

# Ensure repository root is on sys.path for imports
ROOT = str(pathlib.Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from main import BiasGuardDashboard, BiasGuardPro


class DummyAnalyzer(BiasGuardPro):
    def __init__(self):
        # Avoid loading heavy models
        self.detector = None
        self.explainer = None
        self.counterfactuals = None

    def analyze_text(self, text: str):
        # Return a lightweight deterministic result
        prob = 0.9 if 'biased' in text.lower() else 0.1
        cls = 'BIASED' if prob > 0.5 else 'NEUTRAL'
        return {
            'text': text,
            'bias_probability': prob,
            'bias_class': cls,
            'confidence': prob if prob>0.5 else 1-prob,
            'top_biased_words': ['word1'] if prob>0.5 else [],
            'shap_scores': [],
            'counterfactuals': [],
            'timestamp': 0
        }


@pytest.fixture
def dashboard(monkeypatch):
    d = BiasGuardDashboard()
    # inject dummy analyzer
    d.analyzer = DummyAnalyzer()
    return d


def test_analyze_batch_and_summary(dashboard):
    texts = ["This is biased content", "Neutral content here", "Some biased phrase"]
    results = dashboard.analyze_batch(texts)
    assert len(results) == 3
    # check structure
    assert all('bias_probability' in r for r in results)

    summary = dashboard.summarize_batch(results)
    assert summary['total'] == 3
    assert 'avg_bias_probability' in summary
    assert 'class_counts' in summary


def test_compare_groups(dashboard):
    # create fake results
    group_a = [dashboard.analyzer.analyze_text('biased 1'), dashboard.analyzer.analyze_text('biased 2')]
    group_b = [dashboard.analyzer.analyze_text('neutral 1')]
    cmp = dashboard.compare_groups(group_a, group_b)
    assert 'avg_bias_delta' in cmp
    assert cmp['group_a']['count'] == 2
    assert cmp['group_b']['count'] == 1