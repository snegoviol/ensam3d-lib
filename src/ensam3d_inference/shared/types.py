# ==================================================================================================
# 
# Copyright (c) 2026 Ilya Snegov
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
# http://www.apache.org/licenses/LICENSE-2.0
# 
# ==================================================================================================
# src/ensam3d_inference/shared/types.py
# ==================================================================================================

from enum import StrEnum

# ==================================================================================================

class DeviceType(StrEnum):
    """
    Computation device target for model inference and tensor operations.

    Attributes
    ----------
    CPU : DeviceType
        Executes computations on the central processing unit.
    CUDA : DeviceType
        Executes computations on an NVIDIA GPU using the CUDA runtime.
    """

    CPU = "cpu"
    CUDA = "cuda"

# ==================================================================================================
