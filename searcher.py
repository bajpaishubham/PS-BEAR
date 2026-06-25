from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import descriptions
from formulae_table_generator import DEFAULT_TABLE_PATH, load_formula_table

TOP_N = 10


def _rank(query: str, documents: list[str], metadata: list[dict], top_n: int = TOP_N) -> list[dict]:
    """TF-IDF + cosine ranking. Returns the highest-scoring metadata dicts with a
    'score' key added. Returns [] when the corpus has no usable vocabulary."""
    if not query.strip() or not any(doc.strip() for doc in documents):
        return []
    try:
        vectorizer = TfidfVectorizer(stop_words="english")
        document_vectors = vectorizer.fit_transform(documents)
        query_vector = vectorizer.transform([query])
    except ValueError:
        # Empty vocabulary after stop-word removal.
        return []

    scores = cosine_similarity(query_vector, document_vectors).flatten()
    ranked_indices = scores.argsort()[::-1][:top_n]
    results: list[dict] = []
    for index in ranked_indices:
        entry = dict(metadata[index])
        entry["score"] = float(scores[index])
        results.append(entry)
    return results


def search_formulae(query: str, table_path: str = DEFAULT_TABLE_PATH, top_n: int = TOP_N) -> list[dict]:
    rows = load_formula_table(table_path)
    documents: list[str] = []
    metadata: list[dict] = []
    for row in rows:
        document = (
            f"{row.get('description_from_gemini', '')} {row.get('formula_in_plain_english', '')}".strip()
        )
        if not document:
            continue
        documents.append(document)
        metadata.append(
            {
                "label": row.get("name", ""),
                "name": row.get("name", ""),
                "topic": row.get("topic", ""),
                "subtopic": row.get("subtopic", ""),
            }
        )
    return _rank(query, documents, metadata, top_n)


def search_subtopics(query: str, top_n: int = TOP_N) -> list[dict]:
    data = descriptions.load_descriptions()
    documents: list[str] = []
    metadata: list[dict] = []
    for subtopic, description in data.get("subtopics", {}).items():
        if not description.strip():
            continue
        documents.append(description)
        metadata.append({"label": subtopic, "subtopic": subtopic})
    return _rank(query, documents, metadata, top_n)


def search_topics(query: str, top_n: int = TOP_N) -> list[dict]:
    data = descriptions.load_descriptions()
    documents: list[str] = []
    metadata: list[dict] = []
    for topic, description in data.get("topics", {}).items():
        if not description.strip():
            continue
        documents.append(description)
        metadata.append({"label": topic, "topic": topic})
    return _rank(query, documents, metadata, top_n)
