#!/bin/bash
#
# Copyright ETH Zurich 2026
# Licensed under Apache v2.0 see LICENSE for details.
#
# SPDX-License-Identifier: Apache-2.0
#
# Every experiment reported in the thesis, in the order of the decision chain.
# ===========================================================================
#
# Each script holds every earlier axis at the value its predecessor selected, so
# the order is not a convenience: running one out of turn reproduces a
# configuration the thesis does not report. Set ONLY to a space-separated list
# of leading numbers to run a subset, for example:
#
#   ONLY="30 40" ./scripts/reproduce_thesis/run_all.sh
#
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ONLY="${ONLY:-}"

for s in "$HERE"/[0-9][0-9]_*.sh; do
    n="$(basename "$s" | cut -c1-2)"
    if [ -n "$ONLY" ] && ! echo " $ONLY " | grep -q " $n "; then continue; fi
    echo ""
    echo "#####################################################################"
    echo "# $(basename "$s")"
    echo "#####################################################################"
    bash "$s"
done
