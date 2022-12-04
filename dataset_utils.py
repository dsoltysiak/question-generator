from dataclasses import dataclass
from typing import List, Dict

import torch
from datasets import load_dataset
from pandas import DataFrame
from sklearn.utils import shuffle
from torch.utils.data import Dataset
from tqdm import tqdm
from tqdm.notebook import tqdm
from transformers import T5Tokenizer, BatchEncoding

from config import (
    TEST_DATASET_PATH,
    TRAIN_DATASET_PATH,
    TRAIN_DATASET_TENSOR_PATH,
    TEST_DATASET_TENSOR_PATH,
    max_input_length,
    max_target_length,
    pretrained_model,
)


@dataclass
class T2TDataCollator:
    def __call__(self, batch: List) -> Dict[str, torch.Tensor]:
        """
        Take a list of samples from a dataset and collate them into a batch. Returns dictionary of tensors.
        """

        input_ids = torch.stack([example["input_ids"] for example in batch])
        lm_labels = torch.stack([example["decoder_input_ids"] for example in batch])
        lm_labels[lm_labels[:, :] == 0] = -100
        attention_mask = torch.stack([example["attention_mask"] for example in batch])
        decoder_attention_mask = torch.stack(
            [example["decoder_attention_mask"] for example in batch]
        )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": lm_labels,
            "decoder_attention_mask": decoder_attention_mask,
        }


def prepare_dataset() -> None:
    train_dataset = load_dataset("squad", split="train")
    test_dataset = load_dataset("squad", split="validation")

    df_train = get_dataframe(train_dataset)
    df_test = get_dataframe(test_dataset)

    save_dataset(df_train, df_test)

    dataset = load_dataset(
        "csv",
        data_files={"train": str(TRAIN_DATASET_PATH), "test": str(TEST_DATASET_PATH)},
    )
    dataset = dataset.filter(drop_long_inputs, batched=True, load_from_cache_file=False)
    # dataset = dataset.filter(drop_samples, with_indices=True, batched=True, load_from_cache_file=False)
    dataset = dataset.map(convert_to_features, batched=True, load_from_cache_file=False)

    dataset = dataset.remove_columns(["context", "answer", "question"])

    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    columns = [
        "input_ids",
        "decoder_input_ids",
        "attention_mask",
        "decoder_attention_mask",
    ]
    train_dataset.set_format(type="torch", columns=columns)
    test_dataset.set_format(type="torch", columns=columns)

    torch.save(train_dataset, TRAIN_DATASET_TENSOR_PATH)
    torch.save(test_dataset, TEST_DATASET_TENSOR_PATH)


def get_input_encodings(
    batch: Dict, tokenizer: T5Tokenizer, padding: bool = True
) -> BatchEncoding:
    contexts = [f"ask question: {context} " for context in batch["context"]]
    answers = [f"answer: {answer} </s>" for answer in batch["answer"]]
    input_batch = [context + answer for context, answer in zip(contexts, answers)]
    input_encodings = tokenizer.batch_encode_plus(
        input_batch,
        max_length=max_input_length,
        add_special_tokens=True,
        truncation="do_not_truncate",
        pad_to_max_length=padding,
    )
    return input_encodings


def convert_to_features(batch: Dict) -> Dict:
    tokenizer = T5Tokenizer.from_pretrained(pretrained_model)
    input_encodings = get_input_encodings(batch, tokenizer)
    output_batch = [f"question: {question} </s>" for question in batch["question"]]
    target_encodings = tokenizer.batch_encode_plus(
        output_batch,
        max_length=max_target_length,
        add_special_tokens=True,
        truncation=True,
        pad_to_max_length=True,
    )
    encodings = {
        "input_ids": input_encodings["input_ids"],
        "attention_mask": input_encodings["attention_mask"],
        "decoder_input_ids": target_encodings["input_ids"],
        "decoder_attention_mask": target_encodings["attention_mask"],
    }
    return encodings


def drop_long_inputs(batch: Dict) -> List[bool]:
    tokenizer = T5Tokenizer.from_pretrained(pretrained_model)
    input_encodings = get_input_encodings(batch, tokenizer, padding=False)
    return [
        True if len(input_id) <= max_input_length else False
        for input_id in input_encodings.data["input_ids"]
    ]


def drop_samples(batch: Dict, indices: List[int]) -> List[bool]:
    return [True if not idx % 4 else False for idx in range(len(indices))]


def save_dataset(df_train: DataFrame, df_test: DataFrame) -> None:
    train_path = TRAIN_DATASET_PATH
    test_path = TEST_DATASET_PATH
    df_train.to_csv(train_path, index=False)
    df_test.to_csv(test_path, index=False)


def get_dataframe(dataset: Dataset) -> DataFrame:
    df_columns = ["context", "answer", "question"]
    dataframe = DataFrame(columns=df_columns)

    for idx, text in enumerate(tqdm(dataset)):
        context = text["context"]
        answer = text["answers"]["text"][0]
        question = text["question"]
        num_of_words = len(answer.split())
        if num_of_words >= 7:
            continue
        else:
            dataframe.loc[idx] = [context] + [answer] + [question]
    return shuffle(dataframe)


if __name__ == "__main__":
    prepare_dataset()
