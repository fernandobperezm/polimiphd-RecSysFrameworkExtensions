from enum import Enum


class EvaluationStrategy(Enum):
    TIMESTAMP = "TIMESTAMP"
    LEAVE_LAST_K_OUT = "LEAVE_LAST_K_OUT"
