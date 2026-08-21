#!/bin/bash
#
# Copyright ETH Zurich 2026
# Licensed under Apache v2.0 see LICENSE for details.
#
# SPDX-License-Identifier: Apache-2.0
#
# Axis 1, completion: the CTC cells of the input-domain matrix
# ============================================================
#
# Fills: Table 4.3, panel (a), the two rows currently marked \nodata
#        (Time + CTC + BiLSTM, MFCC 64/40 + CTC + BiLSTM).
#
# The chain of Table 4.1 settles the objective on the surviving front-end only,
# so these two cells are outside it by design. They are produced here because
# the table reads better complete: without them the CTC block of panel (a) has
# two rows out of four and the reader cannot see that the domain ordering holds
# under CTC as it does under cross-entropy.
#
# Sentences only. The word panel is left as it is: the words carry the two
# ablations of Section 4.6 and nothing else.
#
# Runtime: 2 configurations x 7 subjects x 2 conditions x 2 protocols.
#
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
start_log "11_axis1_ctc_domains"

ROOT="$ARTIFACTS_BASE/11_axis1_ctc_domains"
BASE="config_thesis/thesis_base_sentences_w2000_rest.yaml"

for CFG in sentences_time_ctc_bilstm_classification_greedy \
           sentences_mfcc_b64_q40_ctc_bilstm_classification_greedy; do
    train_and_analyse "$BASE" \
                      "config_thesis/models_configs/${CFG}.yaml" \
                      "$DATA_SENTENCES" "$ROOT/$CFG" speechnet 2.0 w2000ms
done

done_msg "11_axis1_ctc_domains" "$ROOT"
