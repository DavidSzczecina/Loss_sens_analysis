from enum import Enum


class MetricType(Enum):
    ACCURACY = "accuracy"
    TRAINING_LOSS = "training_loss"
    TEST_LOSS = "test_loss"
    AVG_SENS = "avg_sens"
    AVG_SPEC = "avg_spec"