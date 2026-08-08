#!/bin/bash
#
# Copyright ETH Zurich 2026
# Licensed under Apache v2.0 see LICENSE for details.
#
# SPDX-License-Identifier: Apache-2.0
#
# Axis 5: the window anchor
# =========================
#
# Reproduces: Table 4.10 and Table 6.4
#
# Cue-anchored windows against trigger-free ones, run on the single
# configuration the chain has left carrying the argument, namely the Transformer
# on the STFT map under greedy decoding. The model is held fixed because the
# comparison is between two datasets; the beam sweep is not repeated because
# axis 4 has already shown the closed-set rule absorbs almost all of it.
#
# All three rows carry 20 references and no rest reference, since the
# trigger-free extraction emits no rest window, so the cue-anchored row is a
# rest-free run rather than the 21-class one of axis 3.
#
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
start_log "50_axis5_trigger_free"

SUBJECTS_RUN="$SUBJECTS"

for TASK in classification recognition; do
    MODEL="config_thesis/models_configs/sentences_stft_ctc_transformer_${TASK}_greedy.yaml"

    train_and_analyse config_thesis/thesis_base_sentences_w2000_norest.yaml "$MODEL" \
        "$DATA_SENTENCES" "$ARTIFACTS_BASE/50_axis5_trigger_free/cue_w2000ms_${TASK}" \
        speechnet_transformer 2.0 w2000ms

    train_and_analyse config_thesis/thesis_base_sentences_onset_w2000_norest.yaml "$MODEL" \
        "$DATA_SENTENCES" "$ARTIFACTS_BASE/50_axis5_trigger_free/onset_w2000ms_${TASK}" \
        speechnet_transformer 2.0 w2000ms

    train_and_analyse config_thesis/thesis_base_sentences_onset_w2400_norest.yaml "$MODEL" \
        "$DATA_SENTENCES" "$ARTIFACTS_BASE/50_axis5_trigger_free/onset_w2400ms_${TASK}" \
        speechnet_transformer 2.4 w2400ms
done

# Yield of the detector: the fraction of cue boxes that receive a window.
$PYTHON utils/I_data_preparation/onset_window_yield.py --data_dir "$DATA_SENTENCES" \
    || echo "  [warn] yield report skipped"

done_msg "50_axis5_trigger_free" "$ARTIFACTS_BASE/50_axis5_trigger_free" "Tables 4.10 and 6.4"
