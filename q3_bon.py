from typing import List, Tuple

import torch
from torch import nn


def _module_device(module: nn.Module) -> torch.device:
    try:
        return next(module.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def select_best_of_n(
    responses: List[torch.Tensor],
    rewards: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if len(responses) == 0:
        raise ValueError("responses must contain at least one candidate.")
    if rewards.numel() != len(responses):
        raise ValueError("rewards must have the same number of elements as responses.")

    # Helper provided for best_of_n_sample.
    best_index = int(torch.argmax(rewards).item())
    best_response = responses[best_index]
    best_reward = rewards[best_index]

    return best_response, best_reward


def best_of_n_sample(
    policy_model: nn.Module,
    reward_model: nn.Module,
    tokenizer,
    prompt: str,
    n: int = 16,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
) -> Tuple[str, float, torch.Tensor]:
    if n <= 0:
        raise ValueError("n must be positive.")

    policy_device = _module_device(policy_model)
    reward_device = _module_device(reward_model)

    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    prompt_inputs = tokenizer(prompt, return_tensors="pt")
    prompt_input_ids = prompt_inputs["input_ids"].to(policy_device)
    prompt_attention_mask = prompt_inputs["attention_mask"].to(policy_device)
    prompt_length = prompt_input_ids.shape[1]

    response_candidates = []
    full_sequences = []

    for _ in range(n):
        generated = policy_model.generate(
            input_ids=prompt_input_ids,
            attention_mask=prompt_attention_mask,
            do_sample=True,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
        )
        full_sequence = generated[0].detach()
        response_candidates.append(full_sequence[prompt_length:].detach().cpu())
        full_sequences.append(full_sequence.to(reward_device))

    max_length = max(sequence.shape[0] for sequence in full_sequences)
    padded_sequences = torch.full(
        (n, max_length),
        tokenizer.pad_token_id,
        dtype=torch.long,
        device=reward_device,
    )
    attention_mask = torch.zeros((n, max_length), dtype=torch.long, device=reward_device)
    for row, sequence in enumerate(full_sequences):
        padded_sequences[row, : sequence.shape[0]] = sequence
        attention_mask[row, : sequence.shape[0]] = 1

    with torch.no_grad():
        rewards = reward_model(
            input_ids=padded_sequences,
            attention_mask=attention_mask,
        ).detach().cpu()

    # Best-of-N response selection.
    # rewards shape: (n,)
    # response_candidates contains the generated response tokens only.
    # return:
    #   best_text: selected response as a string
    #   best_reward: scalar float
    #   rewards: all reward scores
    # ==========================

    best_response, best_reward = select_best_of_n(response_candidates, rewards)
    best_text = tokenizer.decode(best_response, skip_special_tokens=True)
    
    return best_text, float(best_reward.item()), rewards
