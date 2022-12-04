import gc

import torch
from transformers import (
    T5ForConditionalGeneration,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
import wandb
from config import (
    pretrained_model,
    TRAIN_DATASET_TENSOR_PATH,
    TEST_DATASET_TENSOR_PATH,
    MODEL_PATH,
)
from dataset_utils import T2TDataCollator


def train() -> None:
    """Training pipeline to fine-tune the pre-trained model"""
    model = T5ForConditionalGeneration.from_pretrained(pretrained_model)
    wandb.init(project="question-generation")

    train_dataset = torch.load(
        TRAIN_DATASET_TENSOR_PATH, map_location=torch.device("cuda")
    )
    test_dataset = torch.load(
        TEST_DATASET_TENSOR_PATH, map_location=torch.device("cuda")
    )

    training_args = TrainingArguments(
        output_dir=str(MODEL_PATH),
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,
        num_train_epochs=4,
        learning_rate=1e-4,
        logging_steps=300,
        evaluation_strategy="steps",
        optim="adafactor",
        save_strategy="steps",
        save_steps=1200,
        load_best_model_at_end=True,
        report_to="wandb",
        metric_for_best_model="eval_loss",
    )
    torch.cuda.empty_cache()
    gc.collect()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=T2TDataCollator(),
    )
    trainer.train()
    wandb.finish()


if __name__ == "__main__":
    train()
