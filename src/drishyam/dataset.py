"""Case dataset: dataset rows -> (state, Question list, gold vectors) per splits/split_v1.json."""

import json
from pathlib import Path

from datasets import load_dataset

from drishyam.data_inspect import REPO, to_gold, to_questions


class CaseDataset:
    """rows selected by split indices; case_id matches the stress-suite convention wf/split/i."""

    def __init__(self, workflow: str, split: str, indices: list[int],
                 split_file: str = "splits/split_v1.json"):
        Path(split_file).read_text()  # fail fast if splits missing
        self.rows = load_dataset(REPO, workflow, split=split).select(indices)
        self.workflow = workflow
        self.split = split

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        row = self.rows[i]
        qs = to_questions(row)
        gold = to_gold(row)
        return {
            "case_id": f"{self.workflow}/{self.split}/{i}",
            "workflow": self.workflow,
            "state": row["state"],
            "questions": qs,
            "gold": [gold[q.qid] for q in qs],
        }


def load_seen_splits(split_file="splits/split_v1.json"):
    from drishyam.splits import SEEN

    spec = json.loads(Path(split_file).read_text())
    # dataset exposes only train/test: train and val are disjoint INDEX lists into split="train"
    train = {wf: CaseDataset(wf, "train", spec["train"][wf], split_file) for wf in SEEN}
    val = {wf: CaseDataset(wf, "train", spec["val"][wf], split_file) for wf in SEEN}
    return train, val
