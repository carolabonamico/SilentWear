#!/bin/bash
#
# Copyright ETH Zurich 2026
# Licensed under Apache v2.0 see LICENSE for details.
#
# SPDX-License-Identifier: Apache-2.0
#
# Gate: baseline on the published corpus
# ======================================
#
# Reproduces: Table 4.2
#
# Settles no axis. It establishes that the pipeline reimplemented here behaves
# as the published one on the data the published one was built from, which is
# what licenses every sentence-level figure that follows.
#
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
start_log "01_gate_baseline_published"

SUBJECTS_RUN="$SUBJECTS_4"
ROOT="$ARTIFACTS_BASE/01_gate_baseline_published"

# The published corpus keeps its own directory names (data_raw_and_filt /
# wins_and_feats_final), which the dedicated base configuration pins.
BASE_PUB="config_thesis/thesis_base_words_published_w1400_rest.yaml"

# Cross-entropy: the published operating point, rows 2 of Table 4.2.
train_and_analyse "$BASE_PUB" \
                  config_thesis/models_configs/speechnet_baseline_words_ce.yaml \
                  "$DATA_WORDS_PUBLISHED" "$ROOT/ce" speechnet 1.4 w1400ms

# CTC on the un-resized backbone: row 3 of Table 4.2, the frame-budget result.
train_and_analyse "$BASE_PUB" \
                  config_thesis/models_configs/speechnet_baseline_words_ctc.yaml \
                  "$DATA_WORDS_PUBLISHED" "$ROOT/ctc" speechnet 1.4 w1400ms

# The STFT backbone with a recurrent stage: row 4 of Table 4.2.
train_and_analyse "$BASE_PUB" \
                  config_thesis/models_configs/words_stft_ce_bilstm.yaml \
                  "$DATA_WORDS_PUBLISHED" "$ROOT/stft_bilstm" speechnet 1.4 w1400ms

done_msg "01_gate_baseline_published" "$ROOT"
