"""Pydantic schemas — every config, artifact and results row validates here (user law)."""

from enum import StrEnum

from pydantic import BaseModel


class Primitive(StrEnum):
    NOUL = "noul"
    CHOICE = "choice"
    SCORE = "score"


class Question(BaseModel):
    qid: str
    primitive: Primitive
    text: str
    options: list[str]  # noul -> ["No","Yes"]; score -> ordered rubric levels


class SanityStep(BaseModel):
    step: int
    loss: float
    grad_norm: float
    finite: bool


class SanityReport(BaseModel):
    model_id: str
    optimizer: str
    device: str
    torch_version: str
    steps: list[SanityStep]
    loss_decreased: bool
    all_finite: bool

    @property
    def passed(self) -> bool:
        return self.all_finite and self.loss_decreased


class SplitSpec(BaseModel):
    version: str
    seed: int
    parquet_sha256: dict[str, str]
    train: dict[str, list[int]]
    val: dict[str, list[int]]
    test_indices: list[int]
    unseen_workflow: str
    unseen_test_indices: list[int]


class DecisionRow(BaseModel):
    case_id: str
    workflow: str
    qid: str
    primitive: Primitive
    split: str
    suite: str  # indomain|unseen|jitter|multiq|ablation|speedup|baseline
    probs: list[float]
    gold: list[float]
    temp_applied: float | None = None


class GateInputs(BaseModel):
    indomain: dict
    unseen: dict
    jitter_flip_rate: float
    multiq_max_delta_raw: float
    multiq_max_delta_4dp: float
    multiq_bitwise_frac: float
    ablation_flip_rate: float
    ablation_acc_drop: float
    speedup_ratio: float
    speedup_p50_ms: dict
