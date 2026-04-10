"""Provided support code for Assignment 3. You do not need to modify this file."""

import json
import os
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


DEFAULT_MODEL_NAME = "gpt2"
DEFAULT_MAX_LENGTH = 512
DEFAULT_ANALYSIS_PROMPTS = [
    (
        "Below is an instruction that describes a task. Write a response that appropriately completes "
        "the request.\n\n### Instruction:\nExplain the main idea behind dropout in neural networks."
        "\n\n### Response:\n"
    ),
    (
        "Below is an instruction that describes a task. Write a response that appropriately completes "
        "the request.\n\n### Instruction:\nGive me two practical study tips for a deep learning course."
        "\n\n### Response:\n"
    ),
    (
        "Below is an instruction that describes a task. Write a response that appropriately completes "
        "the request.\n\n### Instruction:\nI am confused about the difference between KL divergence and "
        "cross-entropy.\n\n### Response:\n"
    ),
    (
        "Below is an instruction that describes a task. Write a response that appropriately completes "
        "the request.\n\n### Instruction:\nSummarize why reinforcement learning from human feedback is useful."
        "\n\n### Response:\n"
    ),
    (
        "Below is an instruction that describes a task. Write a response that appropriately completes "
        "the request.\n\n### Instruction:\nWrite a polite refusal to a request for unsafe advice."
        "\n\n### Response:\n"
    ),
]
RESPONSE_MARKERS = ("\n\n### Response:\n", "### Response:", "\n\nAssistant:", "Assistant:")


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



def find_project_root(marker: str = "q3_utils.py") -> Path:
    cwd = Path.cwd().resolve()
    candidates = [
        Path("/content/drive/MyDrive/assignment3_release-new"),
        Path("/content/drive/MyDrive/assignment3_release"),
        Path("/content/assignment3_release-new"),
        Path("/content/assignment3_release"),
        cwd,
    ]

    for parent in [cwd, *cwd.parents]:
        candidates.append(parent / "assignment3_release-new")
        candidates.append(parent / "assignment3_release")

    seen = set()
    for candidate in candidates:
        candidate = candidate.expanduser().resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if (candidate / marker).exists():
            return candidate

    raise FileNotFoundError(
        f"Could not locate {marker}. Unzip the release folder on your computer first, upload the "
        "extracted assignment3_release-new folder to Google Drive, do not upload the zip file "
        "itself, and open the notebook from that folder."
    )


def get_q3_artifact_paths(project_root: Path) -> Dict[str, Path]:
    artifact_root = project_root / "artifacts" / "q3_rlhf"
    return {
        "artifact_root": artifact_root,
        "preference_dataset": artifact_root / "instruction-preference.json",
        "sft_model_dir": artifact_root / "sft_gpt2",
    }


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)


def load_preference_records(path: Path) -> List[Dict[str, str]]:
    return [normalize_preference_record(record) for record in load_json(path)]


def _normalize_space(text: str) -> str:
    return " ".join(text.split())


def build_instruction_prompt(instruction: str, input_text: str = "") -> str:
    instruction = str(instruction).strip()
    input_text = str(input_text).strip()
    prompt = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{instruction}"
    )
    if input_text:
        prompt += f"\n\n### Input:\n{input_text}"
    prompt += "\n\n### Response:\n"
    return prompt


def _find_last_response_index(text: str) -> Tuple[int, str]:
    for marker in RESPONSE_MARKERS:
        idx = text.rfind(marker)
        if idx != -1:
            return idx, marker
    raise ValueError("Could not find the final response marker in the transcript.")


def split_prompt_and_response(transcript: str) -> Tuple[str, str]:
    text = transcript.strip()
    idx, marker = _find_last_response_index(text)
    prompt = text[: idx + len(marker)].strip()
    response = text[idx + len(marker) :].strip()
    if not prompt or not response:
        raise ValueError("Transcript does not contain a valid prompt/response pair.")
    return prompt, response


def normalize_preference_record(example: Mapping[str, str]) -> Dict[str, str]:
    if {"prompt", "chosen_response", "rejected_response"}.issubset(example.keys()):
        prompt = str(example["prompt"]).strip()
        chosen_response = str(example["chosen_response"]).strip()
        rejected_response = str(example["rejected_response"]).strip()
        if not prompt or not chosen_response or not rejected_response:
            raise ValueError("Normalized preference example contains an empty field.")
        return {
            "prompt": prompt,
            "chosen_response": chosen_response,
            "rejected_response": rejected_response,
        }

    if {"instruction", "chosen", "rejected"}.issubset(example.keys()):
        prompt = build_instruction_prompt(example["instruction"], example.get("input", ""))
        chosen_response = str(example["chosen"]).strip()
        rejected_response = str(example["rejected"]).strip()
        if not chosen_response or not rejected_response:
            raise ValueError("Instruction preference example contains an empty chosen or rejected response.")
        return {
            "prompt": prompt,
            "chosen_response": chosen_response,
            "rejected_response": rejected_response,
        }

    if "chosen" not in example or "rejected" not in example:
        raise KeyError("Expected either normalized fields or raw 'chosen'/'rejected' transcript fields.")

    chosen_text = str(example["chosen"]).strip()
    rejected_text = str(example["rejected"]).strip()

    chosen_prompt, chosen_response = split_prompt_and_response(chosen_text)
    rejected_prompt, rejected_response = split_prompt_and_response(rejected_text)

    if _normalize_space(chosen_prompt) == _normalize_space(rejected_prompt):
        prompt = chosen_prompt
    else:
        common_prefix = os.path.commonprefix([chosen_text, rejected_text]).rstrip()
        idx, marker = _find_last_response_index(common_prefix)
        prompt = common_prefix[: idx + len(marker)].strip()
        chosen_response = chosen_text[idx + len(marker) :].strip()
        rejected_response = rejected_text[idx + len(marker) :].strip()

    if not prompt or not chosen_response or not rejected_response:
        raise ValueError("Failed to extract a valid shared prompt from the preference pair.")

    return {
        "prompt": prompt,
        "chosen_response": chosen_response,
        "rejected_response": rejected_response,
    }


def response_suffix(prompt: str, response: str) -> str:
    if not response:
        raise ValueError("Response cannot be empty.")
    if prompt.endswith((" ", "\n")) or response.startswith((" ", "\n")):
        return response
    return f" {response}"


def format_prompt_response(prompt: str, response: str) -> str:
    return f"{prompt}{response_suffix(prompt, response)}"


def encode_prompt_response(
    prompt: str,
    response: str,
    tokenizer,
    max_length: int = DEFAULT_MAX_LENGTH,
) -> Dict[str, List[int]]:
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    response_ids = tokenizer.encode(response_suffix(prompt, response), add_special_tokens=False)

    if tokenizer.eos_token_id is not None:
        response_ids = response_ids + [tokenizer.eos_token_id]

    input_ids = prompt_ids + response_ids
    if len(input_ids) > max_length:
        raise ValueError(f"Sequence length {len(input_ids)} exceeds max_length={max_length}.")

    response_mask = [0] * len(prompt_ids) + [1] * len(response_ids)
    attention_mask = [1] * len(input_ids)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "response_mask": response_mask,
    }


def preference_record_fits(
    example: Mapping[str, str],
    tokenizer,
    max_length: int = DEFAULT_MAX_LENGTH,
) -> bool:
    record = normalize_preference_record(example)
    try:
        encode_prompt_response(record["prompt"], record["chosen_response"], tokenizer, max_length=max_length)
        encode_prompt_response(record["prompt"], record["rejected_response"], tokenizer, max_length=max_length)
    except ValueError:
        return False
    return True


def build_filtered_subset(
    raw_examples: Iterable[Mapping[str, str]],
    tokenizer,
    limit: int,
    max_length: int = DEFAULT_MAX_LENGTH,
) -> Tuple[List[Dict[str, str]], int]:
    filtered = []
    skipped = 0
    for example in raw_examples:
        try:
            record = normalize_preference_record(example)
        except Exception:
            skipped += 1
            continue

        if preference_record_fits(record, tokenizer, max_length=max_length):
            filtered.append(record)
        else:
            skipped += 1

        if len(filtered) >= limit:
            break
    return filtered, skipped


class PreferenceDataset(Dataset):
    def __init__(self, records: Sequence[Mapping[str, str]]):
        self.records = [normalize_preference_record(record) for record in records]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, str]:
        return dict(self.records[index])


class SFTDataset(Dataset):
    def __init__(self, records: Sequence[Mapping[str, str]]):
        self.records = [normalize_preference_record(record) for record in records]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, str]:
        record = self.records[index]
        return {"prompt": record["prompt"], "response": record["chosen_response"]}


def _pad_batch_field(sequences: Sequence[Sequence[int]], pad_value: int) -> torch.Tensor:
    max_length = max(len(sequence) for sequence in sequences)
    batch = torch.full((len(sequences), max_length), pad_value, dtype=torch.long)
    for row, sequence in enumerate(sequences):
        batch[row, : len(sequence)] = torch.tensor(sequence, dtype=torch.long)
    return batch


class PreferenceCollator:
    def __init__(self, tokenizer, max_length: int = DEFAULT_MAX_LENGTH):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_token_id = tokenizer.pad_token_id
        if self.pad_token_id is None:
            if tokenizer.eos_token_id is None:
                raise ValueError("Tokenizer must define either pad_token_id or eos_token_id.")
            self.pad_token_id = tokenizer.eos_token_id

    def __call__(self, batch: Sequence[Mapping[str, str]]) -> Dict[str, Any]:
        chosen_encoded = []
        rejected_encoded = []

        for example in batch:
            record = normalize_preference_record(example)
            chosen_encoded.append(
                encode_prompt_response(
                    record["prompt"],
                    record["chosen_response"],
                    self.tokenizer,
                    max_length=self.max_length,
                )
            )
            rejected_encoded.append(
                encode_prompt_response(
                    record["prompt"],
                    record["rejected_response"],
                    self.tokenizer,
                    max_length=self.max_length,
                )
            )

        combined_lengths = [
            len(item["input_ids"]) for item in chosen_encoded + rejected_encoded
        ]
        common_max_length = max(combined_lengths)

        def pad_encoded(encoded_batch: Sequence[Dict[str, List[int]]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            input_ids = []
            attention_mask = []
            response_mask = []
            for item in encoded_batch:
                pad_size = common_max_length - len(item["input_ids"])
                input_ids.append(item["input_ids"] + [self.pad_token_id] * pad_size)
                attention_mask.append(item["attention_mask"] + [0] * pad_size)
                response_mask.append(item["response_mask"] + [0] * pad_size)
            return (
                torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(attention_mask, dtype=torch.long),
                torch.tensor(response_mask, dtype=torch.long),
            )

        chosen_input_ids, chosen_attention_mask, chosen_response_mask = pad_encoded(chosen_encoded)
        rejected_input_ids, rejected_attention_mask, rejected_response_mask = pad_encoded(rejected_encoded)

        return {
            "prompt": [normalize_preference_record(item)["prompt"] for item in batch],
            "chosen_text": [
                format_prompt_response(
                    normalize_preference_record(item)["prompt"],
                    normalize_preference_record(item)["chosen_response"],
                )
                for item in batch
            ],
            "rejected_text": [
                format_prompt_response(
                    normalize_preference_record(item)["prompt"],
                    normalize_preference_record(item)["rejected_response"],
                )
                for item in batch
            ],
            "chosen_input_ids": chosen_input_ids,
            "chosen_attention_mask": chosen_attention_mask,
            "chosen_response_mask": chosen_response_mask,
            "rejected_input_ids": rejected_input_ids,
            "rejected_attention_mask": rejected_attention_mask,
            "rejected_response_mask": rejected_response_mask,
        }


class SFTCollator:
    def __init__(self, tokenizer, max_length: int = DEFAULT_MAX_LENGTH):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_token_id = tokenizer.pad_token_id
        if self.pad_token_id is None:
            if tokenizer.eos_token_id is None:
                raise ValueError("Tokenizer must define either pad_token_id or eos_token_id.")
            self.pad_token_id = tokenizer.eos_token_id

    def __call__(self, batch: Sequence[Mapping[str, str]]) -> Dict[str, torch.Tensor]:
        encoded_batch = []
        prompt_lengths = []
        for example in batch:
            prompt = str(example["prompt"]).strip()
            response = str(example["response"]).strip()
            encoded = encode_prompt_response(prompt, response, self.tokenizer, max_length=self.max_length)
            prompt_lengths.append(len(self.tokenizer.encode(prompt, add_special_tokens=False)))
            encoded_batch.append(encoded)

        max_length = max(len(item["input_ids"]) for item in encoded_batch)
        input_ids = []
        attention_mask = []
        labels = []

        for item, prompt_length in zip(encoded_batch, prompt_lengths):
            pad_size = max_length - len(item["input_ids"])
            padded_ids = item["input_ids"] + [self.pad_token_id] * pad_size
            padded_attention_mask = item["attention_mask"] + [0] * pad_size

            label_row = list(item["input_ids"])
            for idx in range(prompt_length):
                label_row[idx] = -100
            label_row += [-100] * pad_size

            input_ids.append(padded_ids)
            attention_mask.append(padded_attention_mask)
            labels.append(label_row)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def move_batch_to_device(batch: Mapping[str, Any], device: torch.device) -> Dict[str, Any]:
    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def summarize_metrics(metric_history: Sequence[Mapping[str, float]]) -> Dict[str, float]:
    if not metric_history:
        return {}

    keys = metric_history[0].keys()
    summary = {}
    for key in keys:
        summary[key] = float(np.mean([entry[key] for entry in metric_history]))
    return summary


def mean_reward_by_n(reward_table: Mapping[int, Sequence[float]]) -> Dict[int, float]:
    return {int(n): float(np.mean(values)) for n, values in reward_table.items()}
