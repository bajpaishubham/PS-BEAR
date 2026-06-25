from __future__ import annotations

import json
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
DESCRIPTIONS_PATH = APP_DIR / "json" / "descriptions.json"


def load_descriptions() -> dict[str, dict[str, str]]:
    """{"topics": {name: desc}, "subtopics": {name: desc}} produced during generation."""
    if not DESCRIPTIONS_PATH.exists():
        return {"topics": {}, "subtopics": {}}
    with DESCRIPTIONS_PATH.open(encoding="utf-8") as handle:
        data = json.load(handle)
    data.setdefault("topics", {})
    data.setdefault("subtopics", {})
    return data
