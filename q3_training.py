"""Provided support code for Assignment 3. You do not need to modify this file."""

from contextlib import nullcontext
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader

from q3_bon import best_of_n_sample
from q3_utils import DEFAULT_MAX_LENGTH, PreferenceCollator, PreferenceDataset, summarize_metrics


def slice_records(records: Sequence[Mapping[str, str]], limit: Optional[int] = None) -> List[Mapping[str, str]]:
    if limit is None:
        return list(records)
    return list(records[:limit])


def build_preference_loader(
    records: Sequence[Mapping[str, str]],
    tokenizer,
    batch_size: int = 4,
    shuffle: bool = False,
    max_length: int = DEFAULT_MAX_LENGTH,
) -> DataLoader:
    collator = PreferenceCollator(tokenizer, max_length=max_length)
    return DataLoader(
        PreferenceDataset(records),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collator,
    )


def _is_cuda_device(device) -> bool:
    return str(device).startswith("cuda")


def _count_batch_tokens(batch: Mapping[str, torch.Tensor]) -> int:
    token_count = 0
    for key in ("chosen_attention_mask", "rejected_attention_mask", "attention_mask"):
        value = batch.get(key)
        if value is not None and torch.is_tensor(value):
            token_count += int(value.sum().item())
    return token_count


def train_reward_model(
    trainer,
    dataloader: DataLoader,
    epochs: int = 1,
    grad_accum_steps: int = 1,
    log_every: int = 50,
    eval_loader: Optional[DataLoader] = None,
    eval_every: Optional[int] = None,
    return_history: bool = False,
):
    if trainer.optimizer is None:
        raise ValueError("Reward-model training requires trainer.optimizer to be set.")

    use_amp = _is_cuda_device(trainer.device)
    scaler = torch.amp.GradScaler(device="cuda", enabled=use_amp)
    history = []
    eval_history = []

    global_step = 0
    tokens_seen = 0

    for epoch in range(epochs):
        trainer.model.train()
        trainer.optimizer.zero_grad()
        for step, batch in enumerate(dataloader, start=1):
            global_step += 1
            tokens_seen += _count_batch_tokens(batch)
            amp_context = torch.autocast(device_type="cuda", enabled=use_amp) if use_amp else nullcontext()
            with amp_context:
                metrics = trainer.train_step(batch)
                loss = metrics["loss"] / grad_accum_steps

            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            should_step = (step % grad_accum_steps == 0) or (step == len(dataloader))
            if should_step:
                if use_amp:
                    scaler.step(trainer.optimizer)
                    scaler.update()
                else:
                    trainer.optimizer.step()
                trainer.optimizer.zero_grad()

            record = {
                "epoch": float(epoch + 1),
                "step": float(step),
                "global_step": float(global_step),
                "tokens_seen": float(tokens_seen),
                "loss": float(metrics["loss"].detach().cpu().item()),
                "accuracy": float(metrics["accuracy"].detach().cpu().item()),
            }
            history.append(record)

            if log_every and (step % log_every == 0 or step == len(dataloader)):
                print(
                    f"RM epoch {epoch + 1}/{epochs} step {step}/{len(dataloader)} "
                    f"loss={record['loss']:.4f} accuracy={record['accuracy']:.4f}"
                )

            should_eval = eval_loader is not None and eval_every and (
                step % eval_every == 0 or step == len(dataloader)
            )
            if should_eval:
                eval_metrics = trainer.evaluate_loader(eval_loader)
                eval_record = {
                    "epoch": float(epoch + 1),
                    "step": float(step),
                    "global_step": float(global_step),
                    "tokens_seen": float(tokens_seen),
                }
                eval_record.update({key: float(value) for key, value in eval_metrics.items()})
                eval_history.append(eval_record)
                print(
                    f"RM validation @ step {step}/{len(dataloader)} "
                    f"loss={eval_record['loss']:.4f} accuracy={eval_record['accuracy']:.4f}"
                )

    summary = summarize_metrics(history)
    if return_history:
        return {"summary": summary, "history": history, "eval_history": eval_history}
    return summary


def train_dpo(
    trainer,
    dataloader: DataLoader,
    epochs: int = 1,
    grad_accum_steps: int = 1,
    log_every: int = 50,
    eval_loader: Optional[DataLoader] = None,
    eval_every: Optional[int] = None,
    return_history: bool = False,
):
    if trainer.optimizer is None:
        raise ValueError("DPO training requires trainer.optimizer to be set.")

    use_amp = _is_cuda_device(trainer.device)
    scaler = torch.amp.GradScaler(device="cuda", enabled=use_amp)
    history = []
    eval_history = []

    global_step = 0
    tokens_seen = 0

    for epoch in range(epochs):
        trainer.policy_model.train()
        trainer.optimizer.zero_grad()
        for step, batch in enumerate(dataloader, start=1):
            global_step += 1
            tokens_seen += _count_batch_tokens(batch)
            amp_context = torch.autocast(device_type="cuda", enabled=use_amp) if use_amp else nullcontext()
            with amp_context:
                loss, metrics = trainer.compute_loss(batch)
                scaled_loss = loss / grad_accum_steps

            if use_amp:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            should_step = (step % grad_accum_steps == 0) or (step == len(dataloader))
            if should_step:
                if use_amp:
                    scaler.step(trainer.optimizer)
                    scaler.update()
                else:
                    trainer.optimizer.step()
                trainer.optimizer.zero_grad()

            record = {
                "epoch": float(epoch + 1),
                "step": float(step),
                "global_step": float(global_step),
                "tokens_seen": float(tokens_seen),
                "loss": float(loss.detach().cpu().item()),
                "reward_margin": float(metrics["reward_margin"].detach().cpu().item()),
                "accuracy": float(metrics["accuracy"].detach().cpu().item()),
            }
            history.append(record)

            if log_every and (step % log_every == 0 or step == len(dataloader)):
                print(
                    f"DPO epoch {epoch + 1}/{epochs} step {step}/{len(dataloader)} "
                    f"loss={record['loss']:.4f} reward_margin={record['reward_margin']:.4f} "
                    f"accuracy={record['accuracy']:.4f}"
                )

            should_eval = eval_loader is not None and eval_every and (
                step % eval_every == 0 or step == len(dataloader)
            )
            if should_eval:
                eval_metrics = trainer.evaluate_loader(eval_loader)
                eval_record = {
                    "epoch": float(epoch + 1),
                    "step": float(step),
                    "global_step": float(global_step),
                    "tokens_seen": float(tokens_seen),
                }
                eval_record.update({key: float(value) for key, value in eval_metrics.items()})
                eval_history.append(eval_record)
                print(
                    f"DPO validation @ step {step}/{len(dataloader)} "
                    f"loss={eval_record['loss']:.4f} reward_margin={eval_record['reward_margin']:.4f} "
                    f"accuracy={eval_record['accuracy']:.4f}"
                )

    summary = summarize_metrics(history)
    if return_history:
        return {"summary": summary, "history": history, "eval_history": eval_history}
    return summary


def evaluate_best_of_n_curve(
    policy_model,
    reward_model,
    tokenizer,
    prompts: Sequence[str],
    n_values: Sequence[int],
    max_new_tokens: int = 48,
    temperature: float = 0.8,
) -> Tuple[Dict[int, List[float]], Dict[int, str]]:
    reward_table = {int(n): [] for n in n_values}
    exemplar_texts = {}

    for prompt_idx, prompt in enumerate(prompts):
        for n in n_values:
            text, reward, _ = best_of_n_sample(
                policy_model,
                reward_model,
                tokenizer,
                prompt=prompt,
                n=int(n),
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
            reward_table[int(n)].append(float(reward))
            if prompt_idx == 0:
                exemplar_texts[int(n)] = text

    return reward_table, exemplar_texts
