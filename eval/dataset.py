import json
from pathlib import Path
from typing import TypedDict


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = BASE_DIR / "eval" / "datasets" / "golden.jsonl"

class EvalCase(TypedDict):
    id: str
    category: str
    query: str
    expected_tools: list[str]
    expected_keywords: list[str]
    should_refuse: bool

def load_dataset(path: Path = DEFAULT_DATASET) -> list[EvalCase]:
    cases = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cases.append(json.loads(line))

    return cases