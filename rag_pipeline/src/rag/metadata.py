from typing import Any


def build_chroma_metadata(doc: dict[str, Any], chunk_index: int, has_table: bool, token_estimate: int) -> dict[str, Any]:
    """
    Builds flat, filterable ChromaDB metadata matching the architecture schema:
    {doc_id, section_id, policy_ref, region, grade_scope, expense_category, effective_date}
    """
    grades: list[str] = doc.get("grade_scope", [])
    grade_band = (
        "ALL" if len(grades) >= 8
        else f"{grades[0]}-{grades[-1]}" if grades
        else "UNKNOWN"
    )
    eff_date = doc.get("effective_date", "2026-01-01")
    eff_date_int = int(eff_date.replace("-", ""))  # Enables $gte / $lte date filtering

    meta: dict[str, Any] = {
        "doc_id": doc["doc_id"],
        "section_id": doc["section_id"],
        "policy_ref": doc["policy_ref"],
        "title": doc["title"],
        "region": doc["region"],
        "grade_scope": "|" + "|".join(grades) + "|",  # e.g. "|L1|L2|L3|L4|"
        "grade_band": grade_band,                     # e.g. "L1-L4", "L5-L6", "ALL"
        "expense_category": doc["expense_category"],
        "effective_date": eff_date,
        "effective_date_int": eff_date_int,
        "status": doc.get("status", "ACTIVE"),        # "ACTIVE" or "SUPERSEDED"
        "doc_type": doc.get("doc_type", "policy"),
        "chunk_index": chunk_index,
        "has_table": has_table,
        "token_estimate": token_estimate,
    }
    return meta