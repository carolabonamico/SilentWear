# Copyright ETH Zurich 2026
# Licensed under Apache v2.0 see LICENSE for details.
#
# SPDX-License-Identifier: Apache-2.0
#

"""
This file includes constants used during the data collection.
"""

FS = 500

# ----------------------------
# Label mappings
# -----------------------------

ORIGINAL_LABELS_WORDS = {
    0: "rest",
    1: "up",
    2: "down",
    3: "left",
    4: "right",
    5: "forward",
    6: "backward",
    7: "start",
    8: "stop",
    9: "advance",
    10: "reverse",
    11: "rotate",
    12: "halt",
    13: "begin",
    14: "grab",
    15: "place",
}

ORIGINAL_LABELS_SENTENCES = {
    0: "rest",
    1: "move forward",
    2: "turn left",
    3: "turn right",
    4: "stop the mission",
    5: "go to the center",
    6: "move inside",
    7: "go to the wall",
    8: "move backward",
    9: "proceed to the door",
    10: "back to initial position",
    11: "enter the room",
    12: "proceed two meters",
    13: "pick up the object",
    14: "put the object down",
    15: "turn on the light",
    16: "switch off the light",
    17: "turn off the volume",
    18: "stop right there",
    19: "start the task",
    20: "start the mission"
}

def get_active_labels(mode: str = "word") -> dict:
    """Returns the active labels based on the mode."""
    if mode == "sentence":
        return ORIGINAL_LABELS_SENTENCES.copy()
    return ORIGINAL_LABELS_WORDS.copy()