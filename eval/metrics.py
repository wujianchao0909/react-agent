from dataclasses import dataclass, field

@dataclass
class CaseResult:
    id: str
    category: str
    query: str
    answer: str
    actual_tools: list[str]
    expected_tools: list[str]
    expected_keywords: list[str]
    should_refuse: bool
    latency_ms: float
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def tool_match(self):
        """期望工具集合是否被完整调用（忽略顺序）"""
        return set(self.actual_tools) >= set(self.expected_tools)

    @property
    def keyword_hit(self) -> bool:
        """任意一个期望关键词出现即算命中"""
        if not self.expected_keywords:
            return True
        return any(kw in self.answer for kw in self.expected_keywords)

    @property
    def refusal_correct(self) -> bool:
        refuse_markers = ["未找到", "没有找到", "找不到", "查不到", "无法找到"]
        refused = any(m in self.answer for m in refuse_markers)
        if self.should_refuse:
            return refused
        return not refused

    @property
    def passed(self) -> bool:
        return self.tool_match and self.keyword_hit and self.refusal_correct

@dataclass
class EvalSummary:
    total: int = 0
    passed: int = 0
    tool_match: int = 0
    keyword_hit: int = 0
    refusal_correct: int = 0
    total_latency_ms: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    by_category: dict[str, dict] = field(default_factory=dict)

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total else 0.0

    @property
    def tool_accuracy(self) -> float:
        return self.tool_match / self.total if self.total else 0.0

    @property
    def keyword_accuracy(self) -> float:
        return self.keyword_hit / self.total if self.total else 0.0

    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / self.total if self.total else 0.0

def aggregate(results: list[CaseResult]) -> EvalSummary:
    s = EvalSummary(total=len(results))
    for r in results:
        s.passed += int(r.passed)
        s.tool_match += int(r.tool_match)
        s.keyword_hit += int(r.keyword_hit)
        s.refusal_correct += int(r.refusal_correct)
        s.total_latency_ms += int(r.latency_ms)
        s.total_input_tokens += int(r.input_tokens)
        s.total_output_tokens += int(r.output_tokens)

        cat = s.by_category.setdefault(r.category, {"total": 0, "passed": 0})
        cat["total"] += 1
        cat["passed"] += int(r.passed)

    return s