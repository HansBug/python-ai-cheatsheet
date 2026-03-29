def score_text_quality(text):
    stripped = text.strip()
    if not stripped:
        return 0.0

    unique_ratio = len(set(stripped)) / len(stripped)
    alpha_ratio = sum(ch.isalnum() for ch in stripped) / len(stripped)
    length_score = min(len(stripped) / 32.0, 1.0)
    return 0.4 * unique_ratio + 0.3 * alpha_ratio + 0.3 * length_score


def score_record(record):
    text_score = score_text_quality(record.get("text", ""))
    image_text_score = record.get("image_text_score", 0.5)
    ocr_score = record.get("ocr_score", 0.5)
    return 0.5 * text_score + 0.3 * image_text_score + 0.2 * ocr_score


def filter_records(records, threshold=0.6):
    kept = []
    dropped = []
    for record in records:
        score = score_record(record)
        item = {**record, "quality_score": round(score, 4)}
        if score >= threshold:
            kept.append(item)
        else:
            dropped.append(item)
    return kept, dropped


def bucketize_records(records):
    buckets = {"high": [], "mid": [], "low": []}
    for record in records:
        score = record["quality_score"]
        if score >= 0.8:
            buckets["high"].append(record)
        elif score >= 0.6:
            buckets["mid"].append(record)
        else:
            buckets["low"].append(record)
    return buckets


if __name__ == "__main__":
    records = [
        {"id": 1, "text": "A clear image of a street sign", "image_text_score": 0.9, "ocr_score": 0.8},
        {"id": 2, "text": "!!!!!!", "image_text_score": 0.3, "ocr_score": 0.1},
        {"id": 3, "text": "A receipt photo", "image_text_score": 0.7, "ocr_score": 0.9},
    ]

    kept, dropped = filter_records(records, threshold=0.6)
    print("kept ids:", [item["id"] for item in kept])
    print("dropped ids:", [item["id"] for item in dropped])
    print("bucket sizes:", {k: len(v) for k, v in bucketize_records(kept + dropped).items()})
