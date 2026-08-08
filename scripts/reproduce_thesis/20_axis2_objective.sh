#!/bin/bash
#
# Copyright ETH Zurich 2026
# Licensed under Apache v2.0 see LICENSE for details.
#
# SPDX-License-Identifier: Apache-2.0
#
# Axis 2: the training objective
# ==============================
#
# Reproduces: Table 4.4, CTC rows
#
# Cross-entropy against CTC on the front-end axis 1 selected. Cross-entropy
# leaves the study here, and it leaves for a reason stronger than its accuracy:
# it emits no per-frame posterior and therefore cannot produce a character
# string at all, which is what the whole of Chapter 6 requires.
#
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
start_log "20_axis2_objective"

SUBJECTS_RUN="$SUBJECTS"

for CORPUS in words sentences; do
    if [ "$CORPUS" = "words" ]; then
        BASE=config_thesis/thesis_base_words_w1400_rest.yaml
        DATA="$DATA_WORDS"; WIN=1.4; WID=w1400ms
    else
        BASE=config_thesis/thesis_base_sentences_w2000_rest.yaml
        DATA="$DATA_SENTENCES"; WIN=2.0; WID=w2000ms
    fi
    MODEL="config_thesis/models_configs/${CORPUS}_stft_ctc_bilstm_classification_greedy.yaml"
    ROOT="$ARTIFACTS_BASE/20_axis2_objective/${CORPUS}_stft_ctc_bilstm"
    train_and_analyse "$BASE" "$MODEL" "$DATA" "$ROOT" speechnet "$WIN" "$WID"
done

done_msg "20_axis2_objective" "$ARTIFACTS_BASE/20_axis2_objective" "Table 4.4, CTC rows"
