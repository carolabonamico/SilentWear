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

train_and_analyse config_thesis/thesis_base_words_w1400_rest.yaml \
                  config_thesis/models_configs/speechnet_baseline_words_ce.yaml \
                  "$DATA_WORDS_PUBLISHED" "$ROOT" speechnet 1.4 w1400ms

done_msg "01_gate_baseline_published" "$ROOT" "Table 4.2"
