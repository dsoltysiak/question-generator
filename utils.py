from dataclasses import dataclass
from typing import Dict, List, Any

import torch
from keybert import KeyBERT
from nltk import sent_tokenize
from transformers import T5ForConditionalGeneration, T5Tokenizer

from config import max_input_length, summary_model


def get_best_keywords(text) -> List[str]:
    keywords = get_keywords(text)
    summarized_text = summarize_text(text)
    keywords_summarized = get_keywords(summarized_text)
    important_keywords = [
        keyword for keyword in keywords if keyword in keywords_summarized
    ]
    print(summarized_text)
    return important_keywords[:2]


def get_keywords(text: str) -> tuple[Any]:
    keywords_model = KeyBERT()
    keywords = keywords_model.extract_keywords(text)
    return list(zip(*keywords))[0]


def summarize_text(text: str) -> str:
    model = T5ForConditionalGeneration.from_pretrained(summary_model).to("cuda")
    tokenizer = T5Tokenizer.from_pretrained(summary_model)
    text = text.strip().replace("\n", " ")
    text = "summarize: " + text
    encoding = tokenizer.encode_plus(
        text,
        max_length=max_input_length,
        pad_to_max_length=False,
        truncation=True,
        return_tensors="pt",
    ).to("cuda")

    input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]

    outs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        early_stopping=True,
        num_beams=4,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        min_length=32,
        max_length=256,
    ).to("cuda")

    summary = tokenizer.decode(outs[0], skip_special_tokens=True)
    summary = postprocess_text(summary)
    return summary.strip()


def postprocess_text(text: str) -> str:
    final_text = ""
    for sent in sent_tokenize(text):
        sent = sent.capitalize()
        final_text += " " + sent
    return final_text


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
