from src.rag.hybrid_retriever import HybridPolicyRetriever

IN_DOMAIN = [
    ("Mumbai hotel cap for L3 employee in January", "IN", "hotel", "L3"),
    ("Personal car mileage rate in US", "US", "mileage", "L4"),
    ("Receipt threshold for meals in UK", "UK", "receipt", "L2"),
    ("Domestic flight cabin class for L6 in EU", "EU", "flight", "L6"),
]

OUT_OF_DOMAIN = [
    ("Hotel cap for Thimphu Bhutan for L3 employee", None, "hotel", "L3"),
    ("Daily meal per diem for Antarctica research trip", None, "per_diem", "L4"),
    ("Reimbursement policy for private yacht charter", None, None, "L7"),
    ("SpaceX orbital travel insurance coverage", None, None, "L5"),
]

def calibrate():
    # Set threshold=0.0 temporarily so we can inspect raw top-1 scores
    retriever = HybridPolicyRetriever(confidence_threshold=0.0)

    in_scores, out_scores = [], []

    print("=== IN-DOMAIN QUERIES ===")
    for q, reg, cat, grade in IN_DOMAIN:
        res = retriever.search_policy_corpus(q, region=reg, expense_category=cat, grade=grade, top_k=1)
        score = res["results"][0]["rerank_score"] if res["results"] else 0.0
        in_scores.append(score)
        print(f"  {score:.4f} | {q}")

    print("\n=== OUT-OF-DOMAIN / UNLISTED QUERIES ===")
    for q, reg, cat, grade in OUT_OF_DOMAIN:
        res = retriever.search_policy_corpus(q, region=reg, expense_category=cat, grade=grade, top_k=1)
        score = res["results"][0]["rerank_score"] if res["results"] else 0.0
        out_scores.append(score)
        print(f"  {score:.4f} | {q}")

    suggested = round((min(in_scores) + max(out_scores)) / 2, 3)
    print(f"\nMin In-Domain: {min(in_scores):.4f} | Max Out-of-Domain: {max(out_scores):.4f}")
    print(f"Suggested Calibrated Threshold: {suggested}")

if __name__ == "__main__":
    calibrate()