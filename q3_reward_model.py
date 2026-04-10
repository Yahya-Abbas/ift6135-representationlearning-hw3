from typing import Dict, Iterable, Optional

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel

from q3_utils import move_batch_to_device, summarize_metrics


class RewardModel(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        # Reward model on top of a pretrained GPT-2 backbone.
        # model_name: local path or HF model identifier
        # Save the transformer backbone and a scalar reward head on self.
        # ==========================
        self.backbone = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.backbone.config.hidden_size
        self.reward_head = nn.Linear(self.hidden_size, 1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # Score each prompt-response sequence with a scalar reward.
        # input_ids shape: (batch_size, sequence_length)
        # attention_mask shape: (batch_size, sequence_length)
        # return shape: (batch_size,)
        # ==========================
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_states = outputs.last_hidden_state # shape (batch_size, seq_length, hidden_size)
        last_np_token_idx = attention_mask.sum(dim=1) - 1 # shape (batch_size,)
        index = last_np_token_idx.view(-1, 1, 1).expand(-1, 1, self.hidden_size) # shape (batch_size, 1, hidden_size)
        last_hidden_state = torch.gather(last_hidden_states, 1, index) # shape (batch_size, 1, hidden_size)
        last_hidden_state = last_hidden_state.squeeze(1) # shape (batch_size, hidden_size)
        rewards = self.reward_head(last_hidden_state) # shape (batch_size, 1)
        rewards = rewards.squeeze(1) # shape (batch_size,)

        return rewards


def compute_preference_loss(
    rewards_chosen: torch.Tensor,
    rewards_rejected: torch.Tensor,
) -> torch.Tensor:
    # Bradley-Terry preference loss.
    # rewards_chosen shape: (batch_size,)
    # rewards_rejected shape: (batch_size,)
    # return: scalar loss tensor
    # ==========================
    loss = -F.logsigmoid(rewards_chosen - rewards_rejected).mean()
    return loss


def compute_reward_accuracy(
    rewards_chosen: torch.Tensor,
    rewards_rejected: torch.Tensor,
) -> torch.Tensor:
    # Preference accuracy.
    # rewards_chosen shape: (batch_size,)
    # rewards_rejected shape: (batch_size,)
    # return: scalar accuracy tensor
    # ==========================
    accuracy = (rewards_chosen > rewards_rejected).float().mean()
    return accuracy


class RewardModelTrainer:
    def __init__(
        self,
        model: RewardModel,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.device = torch.device(device) if device is not None else self._infer_device(model)
        self.model.to(self.device)
        self.optimizer = optimizer

    @staticmethod
    def _infer_device(model: nn.Module) -> torch.device:
        try:
            return next(model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        batch = move_batch_to_device(batch, self.device)
        # One reward-model forward pass on a batch of chosen/rejected sequences.
        # batch contains chosen/rejected input ids and attention masks.
        # return keys: `loss` and `accuracy`
        # ==========================
        # the keys in the batch are:
        # dict_keys(['prompt', 'chosen_text', 'rejected_text', 'chosen_input_ids',
        # 'chosen_attention_mask', 'chosen_response_mask', 'rejected_input_ids',
        # 'rejected_attention_mask', 'rejected_response_mask'])
        
        chosen_rewards = self.model(batch["chosen_input_ids"], batch["chosen_attention_mask"])
        rejected_rewards = self.model(batch["rejected_input_ids"], batch["rejected_attention_mask"])

        loss = compute_preference_loss(chosen_rewards, rejected_rewards)
        accuracy = compute_reward_accuracy(chosen_rewards, rejected_rewards)

        return {"loss": loss, "accuracy": accuracy}

    def optimizer_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        if self.optimizer is None:
            raise ValueError("RewardModelTrainer.optimizer_step requires an optimizer.")
        self.model.train()
        metrics = self.train_step(batch)
        self.optimizer.zero_grad()
        metrics["loss"].backward()
        self.optimizer.step()
        return {key: float(value.detach().cpu().item()) for key, value in metrics.items()}

    def evaluate_loader(self, dataloader: Iterable[Dict[str, torch.Tensor]]) -> Dict[str, float]:
        self.model.eval()
        metric_history = []
        with torch.no_grad():
            for batch in dataloader:
                metrics = self.train_step(batch)
                metric_history.append(
                    {key: float(value.detach().cpu().item()) for key, value in metrics.items()}
                )
        return summarize_metrics(metric_history)
