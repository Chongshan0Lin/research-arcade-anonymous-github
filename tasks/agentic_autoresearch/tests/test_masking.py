from tasks.agentic_autoresearch.build_literature_evidence import (context_tokens, delexicalize,
                                                                  mask_citations, title_aliases)

TEXT = (r"Graph networks~\citep{kipf2017semi,velivckovic2017graph} are widely used. "
        r"We follow the setup of \citet{kipf2017semi} and extend it with attention \cite{vaswani2017}.")


def test_target_marker_is_masked_and_others_anonymized():
    out, spans = mask_citations(TEXT, ["kipf2017semi"])
    assert "[MASKED_CITATION]" in out
    assert "kipf2017semi" not in out           # target key gone
    assert "velivckovic2017graph" not in out   # other keys never leak either
    assert "vaswani2017" not in out
    assert out.count("[MASKED_CITATION]") == 2  # both markers containing the target
    assert sum(s["is_target"] for s in spans) == 2


def test_non_target_markers_become_generic_citation():
    out, _ = mask_citations(TEXT, ["vaswani2017"])
    assert out.count("[MASKED_CITATION]") == 1
    assert out.count("[CITATION]") == 2


def test_no_target_marker_returns_no_target_span():
    out, spans = mask_citations(TEXT, ["not_present_key"])
    assert not any(s["is_target"] for s in spans)
    assert "[MASKED_CITATION]" not in out


def test_delex_removes_title_alias_and_author_year():
    text = "We build on BERT: Pre-training of Deep Bidirectional Transformers, i.e. BERT (Devlin et al., 2019)."
    out, removed = delexicalize(text, "BERT: Pre-training of Deep Bidirectional Transformers",
                                ["BERT"], ["Devlin et al., 2019"])
    assert "BERT" not in out
    assert "Devlin et al., 2019" not in out
    assert len(removed) >= 3


def test_delex_does_not_touch_unrelated_context():
    text = "We evaluate on ImageNet and CIFAR-10 with a ResNet backbone."
    out, removed = delexicalize(text, "Attention Is All You Need", ["Transformer"], [])
    assert out == text and removed == []


def test_context_tokens_ignores_latex_commands():
    assert context_tokens(r"\textbf{A} \alpha") < 3
    assert context_tokens("one two three four five") == 5


def test_title_aliases():
    assert "BERT" in title_aliases("BERT: Pre-training of Deep Bidirectional Transformers")
    assert "GAT" in title_aliases("Graph Attention Networks (GAT)")
    assert title_aliases("a study of things") == []
