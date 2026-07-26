import lm7


def test_explain_reports_selection():
    explanation = lm7.explain(target="cpu", backend="eager")
    assert "Selected eager for cpu" in explanation
    assert "Candidates:" in explanation
