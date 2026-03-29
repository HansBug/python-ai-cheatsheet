import random
import re


def reservoir_sample(items, k, seed=0):
    rng = random.Random(seed)
    result = []
    for idx, item in enumerate(items):
        if idx < k:
            result.append(item)
        else:
            j = rng.randint(0, idx)
            if j < k:
                result[j] = item
    return result


def normalize_text(text):
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def shingle(text, n=3):
    text = normalize_text(text)
    if len(text) < n:
        return {text}
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def jaccard(a, b):
    return len(a & b) / max(1, len(a | b))


def deduplicate_records(records, threshold=0.8):
    kept = []
    shingles = []

    for record in records:
        current = shingle(record["text"])
        duplicated = False
        for existing in shingles:
            if jaccard(current, existing) >= threshold:
                duplicated = True
                break

        if not duplicated:
            kept.append(record)
            shingles.append(current)

    return kept


if __name__ == "__main__":
    records = [
        {"id": 1, "text": "A cat sits on the sofa."},
        {"id": 2, "text": "A cat sits on the sofa !"},
        {"id": 3, "text": "A dog runs in the park."},
        {"id": 4, "text": "A person cooks dinner."},
    ]

    sampled = reservoir_sample(records, k=3, seed=42)
    deduped = deduplicate_records(sampled, threshold=0.8)
    print("sampled:", [item["id"] for item in sampled])
    print("deduped:", [item["id"] for item in deduped])
