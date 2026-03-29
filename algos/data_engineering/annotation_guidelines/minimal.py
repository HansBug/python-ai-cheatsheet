from dataclasses import dataclass


@dataclass
class AnnotationTask:
    name: str
    labels: set[str]
    multi_label: bool = False


def validate_annotation(task, annotation):
    labels = annotation["labels"]
    if not labels:
        return False, "empty labels"
    if not task.multi_label and len(labels) > 1:
        return False, "multi-label not allowed"
    if not set(labels).issubset(task.labels):
        return False, "unknown label"
    return True, "ok"


def majority_vote(label_groups):
    counts = {}
    for label in label_groups:
        counts[label] = counts.get(label, 0) + 1
    return max(counts, key=counts.get)


def agreement_rate(annotation_groups):
    matched = sum(len(set(group)) == 1 for group in annotation_groups)
    return matched / max(1, len(annotation_groups))


if __name__ == "__main__":
    task = AnnotationTask(
        name="image_safety",
        labels={"safe", "unsafe"},
        multi_label=False,
    )

    annotations = [
        {"sample_id": 1, "labels": ["safe"]},
        {"sample_id": 2, "labels": ["unsafe"]},
    ]

    for annotation in annotations:
        print(annotation["sample_id"], validate_annotation(task, annotation))

    agreement = agreement_rate(
        [
            ["safe", "safe", "safe"],
            ["unsafe", "unsafe", "safe"],
            ["safe", "safe", "safe"],
        ]
    )
    print("agreement:", agreement)
    print("majority vote:", majority_vote(["unsafe", "safe", "unsafe"]))
