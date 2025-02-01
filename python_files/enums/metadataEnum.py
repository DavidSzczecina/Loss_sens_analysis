from enum import Enum


class MetadataType(Enum):
    NUM_EPOCHS = "num_epochs"
    SEED_VALUE = "seed_value"
    MIN_SIZE = "min_size"
    CORRUPTION_RATES = "corruption_rates"
    NUM_CLASSES = "num_classes"
    DATASET = "dataset"
    MODEL_ARCHITECTURE = "model_architecture"
    VERSION = "version"