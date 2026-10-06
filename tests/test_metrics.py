from eval.metrics import CaseResult

def mk(answer: str, keywords: list[str], should_refuse: bool = False) -> CaseResult:
    return CaseResult(
        id="t", category="test", query="q",
        answer=answer,
        actual_tools=["search_knowledge_real"],
        expected_tools=["search_knowledge_real"],
        expected_keywords=keywords,
        should_refuse=should_refuse,
        latency_ms=100.0
    )

def test_keyword_hit_any_match():
    """电压/电势差/电位差，任意命中即可"""
    r = mk("电压是两点间的电位差", keywords=["电势差", "电压"])
    assert r.keyword_hit

def test_keyword_hit_no_match():
    r = mk("我不知道", keywords=["电压"])
    assert not r.keyword_hit

def test_refusal_correct():
    r = mk("知识库中未找到相关信息。", keywords=[], should_refuse=True)
    assert r.refusal_correct

def test_refusal_miss():
    r = mk("红烧肉的做法是……", keywords=[], should_refuse=True)
    assert not r.refusal_correct

def test_tool_match():
    r = mk("x", keywords=[])
    assert r.tool_match