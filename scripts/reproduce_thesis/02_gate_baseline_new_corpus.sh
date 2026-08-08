#!/bin/bash
#
# Copyright ETH Zurich 2026
# Licensed under Apache v2.0 see LICENSE for details.
#
# SPDX-License-Identifier: Apache-2.0
#
# Gate: baseline on the corpus of this thesis
# ===========================================
#
# Reproduces: Table 4.3
#
# The same reference model on the word subset recorded for this thesis. Together
# with 01 it separates a difference due to the reimplementation from one due to
# the recordings.
#
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
start_log "02_gate_baseline_new_corpus"

SUBJECTS_RUN="$SUBJECTS_3"
ROOT="$ARTIFACTS_BASE/02_gate_baseline_new_corpus"

train_and_analyse config_thesis/thesis_base_words_w1400_rest.yaml \
                  config_thesis/models_configs/speechnet_baseline_words_ce.yaml \
                  "$DATA_WORDS" "$ROOT" speechnet 1.4 w1400ms

done_msg "02_gate_baseline_new_corpus" "$ROOT" "Table 4.3"
