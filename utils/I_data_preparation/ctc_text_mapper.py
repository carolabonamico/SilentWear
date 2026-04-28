"""
Text mapper for CTC labels and token IDs.
"""

from __future__ import annotations

import os
from typing import Dict, List, Tuple

import editdistance
import torch

from utils.I_data_preparation.text_transform import CTCTextTransform

DEFAULT_BLANK_ID = 0


class CTCTextMapper(CTCTextTransform):
    """Map class labels to CTC token targets and decode token IDs back to words."""

    def __init__(
        self,
        lexicon_path: str | None = None,
        lexicon_words: List[str] | None = None,
        train_label_map: Dict[int, str] | None = None,
        blank_id: int = DEFAULT_BLANK_ID,
    ):
        self.blank_id = int(blank_id)
        self.train_label_map = train_label_map or {}

        self.label_to_word_map = {
            int(k): self.clean_text(v) for k, v in self.train_label_map.items()
        }
        self.word_to_label_map = {
            word: label for label, word in self.label_to_word_map.items()
        }

        self.lexicon_words = []
        if lexicon_path:
            if not os.path.exists(lexicon_path):
                raise FileNotFoundError(f"CTC lexicon_path does not exist: {lexicon_path}.")
            with open(lexicon_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        self.lexicon_words.append(self.clean_text(line.strip().split()[0]))
        elif lexicon_words:
            self.lexicon_words = [self.clean_text(w) for w in lexicon_words]
        else:
            raise ValueError("CTCTextMapper requires either lexicon_path or lexicon_words.")

        self.lexicon_words = list(dict.fromkeys(self.lexicon_words))

        # If training labels are provided, keep only lexicon words that belong to the active label set (e.g., drop "rest" when include_rest=False).
        if self.word_to_label_map:
            active_words = set(self.word_to_label_map.keys())
            self.lexicon_words = [word for word in self.lexicon_words if word in active_words]

            missing_active_words = sorted(active_words.difference(self.lexicon_words))
            if missing_active_words:
                raise ValueError(f"CTC lexicon is missing active label words: {missing_active_words}.")

        if not self.lexicon_words:
            raise ValueError("CTC lexicon is empty after loading.")

        super().__init__(vocab_words=self.lexicon_words, blank_id=self.blank_id)

    def label_int_to_words(self, labels: torch.Tensor) -> List[str]:
        """Convert label IDs to words using label_to_word_map."""
        return [
            self.label_to_word_map.get(int(label), str(int(label)))
            for label in labels.detach().cpu().tolist()
        ]

    def ctc_targets_from_label_int(
        self, targets: torch.Tensor, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode target labels into CTC character token stream and per-sample lengths."""
        target_tokens, target_lengths = [], []
        for word in self.label_int_to_words(targets):
            encoded = self.text_to_int(word)
            if not encoded:
                raise ValueError(f"CTC target word '{word}' produced an empty token sequence.")
            target_tokens.extend(encoded)
            target_lengths.append(len(encoded))

        return (
            torch.tensor(target_tokens, dtype=torch.long, device=device),
            torch.tensor(target_lengths, dtype=torch.long, device=device),
        )

    def token_int_to_words(self, pred_ids: torch.Tensor) -> List[str]:
        """Decode predicted token IDs to words with CTC collapse (exact decode)."""
        if pred_ids.ndim == 1:
            pred_ids = pred_ids.unsqueeze(0)

        words = []
        for seq in pred_ids.detach().cpu().tolist():
            collapsed, prev = [], None
            for token_id in seq:
                if token_id == self.blank_id:
                    prev = token_id
                    continue
                if token_id != prev:
                    collapsed.append(token_id)
                prev = token_id

            words.append(self.clean_text(self.int_to_text(collapsed)))

        return words

    def _closest_known_word(self, word: str) -> str | None:
        """Find nearest train label word by edit distance for lexicon-constrained decoding."""
        if not word or not self.word_to_label_map:
            return None
        return min(self.word_to_label_map, key=lambda candidate: editdistance.eval(word, candidate))

    def words_to_label_int(
        self, words: List[str], allow_nearest: bool = True
    ) -> Tuple[List[int], float]:
        """Map decoded words to class IDs, with optional nearest-word fallback."""
        if not words:
            return [], 0.0

        preds = []
        unknown = 0

        for raw_word in words:
            word = self.clean_text(raw_word)

            if word in self.word_to_label_map:
                preds.append(int(self.word_to_label_map[word]))
                continue

            if word == "":
                preds.append(-1)
                unknown += 1
                continue

            if allow_nearest:
                nearest = self._closest_known_word(word)
                if nearest is not None:
                    preds.append(int(self.word_to_label_map[nearest]))
                    continue

            preds.append(-1)
            unknown += 1

        return preds, unknown / float(len(words))