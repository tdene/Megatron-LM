# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

from enum import Enum


class Headers(Enum):
    """
    Enum representing headers used for communication with the inference-coordinator.
    """

    ENGINE_CONNECT = 0
    CLIENT_CONNECT = 1
    ACK = 2
    MICROBATCH_SYNC = 3
    SUBMIT_REQUEST = 4
    ENGINE_REPLY = 5
    PAUSE = 6
    UNPAUSE = 7
    STOP = 8
