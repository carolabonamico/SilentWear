"""
Defines task-specific strategies for computing loss and making predictions, such as CrossEntropy and CTC.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from abc import ABC, abstractmethod


class TaskStrategy(ABC):

    @abstractmethod
    def compute_loss(self, outputs, targets, device: torch.device) -> torch.Tensor:
        pass

    def backward(self, loss: torch.Tensor) -> None:
        """Run one backward pass for the current strategy."""
        loss.backward()

    @staticmethod
    def _extract_logits(outputs) -> torch.Tensor:
        """Extracts logits from model outputs, handling both tensors and tuple/list outputs."""
        return outputs[0] if isinstance(outputs, (tuple, list)) else outputs

    @staticmethod
    def greedy_decode(logits: torch.Tensor) -> torch.Tensor:
        """Greedy decode: pick the maximum-probability class at each position."""
        return torch.argmax(logits, dim=-1)

    @abstractmethod
    def predict_labels(self, outputs) -> np.ndarray:
        """Return predicted class labels as a numpy array."""
        pass


class CrossEntropyStrategy(TaskStrategy):
    """Standard cross-entropy strategy for classification tasks."""

    def __init__(self):
        self.criterion = nn.CrossEntropyLoss()

    def _prepare_logits(self, outputs) -> torch.Tensor:
        logits = self._extract_logits(outputs)
        if logits.ndim == 3:
            return logits.mean(dim=1)
        if logits.ndim != 2:
            raise ValueError(f"Unsupported logits shape for CE: {tuple(logits.shape)}")
        return logits

    def compute_loss(self, outputs, targets, device: torch.device) -> torch.Tensor:
        return self.criterion(self._prepare_logits(outputs), targets.long())

    def predict_labels(self, outputs) -> np.ndarray:
        return self.greedy_decode(self._prepare_logits(outputs)).detach().cpu().numpy()


class CTCStrategy(TaskStrategy):
    """CTC strategy for sequence-to-sequence tasks."""

    def __init__(self, text_mapper, allow_nearest_word_match: bool = True):
        self.text_mapper = text_mapper
        self.criterion = nn.CTCLoss(blank=self.text_mapper.blank_id, zero_infinity=True)
        self.allow_nearest_word_match = allow_nearest_word_match

    def compute_loss(self, outputs, targets, device: torch.device) -> torch.Tensor:
        """Expects logits of shape (B, T, C)."""
        logits = self._extract_logits(outputs)
        bsz, time_steps, _ = logits.shape
        pred = F.log_softmax(logits, dim=-1).transpose(0, 1)
        input_lengths = torch.full((bsz,), fill_value=time_steps, dtype=torch.long, device=device)
        target_tokens, target_lengths = self.text_mapper.ctc_targets_from_label_int(targets, device)
        return self.criterion(pred, target_tokens, input_lengths, target_lengths)

    def predict_labels(self, outputs) -> np.ndarray:
        logits = self._extract_logits(outputs)
        words = self.text_mapper.token_int_to_words(self.greedy_decode(logits))
        preds, _ = self.text_mapper.words_to_label_int(words, allow_nearest=self.allow_nearest_word_match)
        return np.asarray(preds, dtype=np.int64)

    def backward(self, loss: torch.Tensor) -> None:
        """Temporarily disable deterministic algorithms on CUDA for CTC backward."""
        deterministic = loss.device.type == "cuda" and torch.are_deterministic_algorithms_enabled()
        if deterministic:
            torch.use_deterministic_algorithms(False)
        loss.backward()
        if deterministic:
            torch.use_deterministic_algorithms(True)