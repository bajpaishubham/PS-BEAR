from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
TOPICS_CSV = APP_DIR / "csv_tables" / "topics_and_subtopics.csv"


@dataclass(frozen=True)
class SubtopicEntry:
    topic_id: int
    topic: str
    topic_short: str
    subtopic_id: int
    subtopic: str
    local_index: int  # 1-based position of this subtopic within its topic


def load_subtopics() -> list[SubtopicEntry]:
    entries: list[SubtopicEntry] = []
    counts: dict[int, int] = {}
    with TOPICS_CSV.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            topic_id = int(row["Topic_id"])
            counts[topic_id] = counts.get(topic_id, 0) + 1
            entries.append(
                SubtopicEntry(
                    topic_id=topic_id,
                    topic=row["Topic"].strip(),
                    topic_short=row["Topic_short"].strip(),
                    subtopic_id=int(row["Subtopic_id"]),
                    subtopic=row["Subtopic"].strip(),
                    local_index=counts[topic_id],
                )
            )
    return entries


def topic_list() -> list[str]:
    seen: list[str] = []
    for entry in load_subtopics():
        if entry.topic not in seen:
            seen.append(entry.topic)
    return seen
