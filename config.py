from pathlib import Path

TRAIN_DATASET_PATH = Path("dataset/train.csv")
TEST_DATASET_PATH = Path("dataset/test.csv")
TRAIN_DATASET_TENSOR_PATH = Path("dataset/train-large.pt")
TEST_DATASET_TENSOR_PATH = Path("dataset/test-large.pt")
MODEL_PATH = Path("models_base_lr1e4")
pretrained_model = "t5-large"
summary_model = "t5-large"
max_input_length = 256
max_target_length = 64
