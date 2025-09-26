############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# ##########################################################################

"""
FINN: A Framework for Fast, Scalable Quantized Neural Network Inference

This package provides tools for building and deploying quantized neural networks
on FPGAs and other accelerators.
"""

import os
import warnings
from pathlib import Path
import importlib.util


def _validate_env_vars():
    """Validate and set up required environment variables using importlib."""
    # Check FINN_ROOT
    finn_root = os.environ.get("FINN_ROOT")
    if not finn_root:
        # Use importlib to find the finn package location
        try:
            finn_spec = importlib.util.find_spec("finn")
            if finn_spec and finn_spec.origin:
                # Get the path to the finn package
                finn_init_path = Path(finn_spec.origin).resolve()
                # Navigate from src/finn/__init__.py to project root
                finn_root = finn_init_path.parent.parent.parent
                os.environ["FINN_ROOT"] = str(finn_root)
            else:
                raise RuntimeError("Could not find finn module spec")
        except Exception as e:
            warnings.warn(
                f"FINN_ROOT environment variable is not set and could not be inferred: {e}\n"
                "This may cause issues with certain FINN operations. "
                "Please set FINN_ROOT to the root directory of your FINN installation."
            )
            return

    # Check FINN_DEPS_DIR
    finn_deps_dir = os.environ.get("FINN_DEPS_DIR")
    if not finn_deps_dir and finn_root:
        # Always set default location under FINN_ROOT
        default_deps_dir = Path(finn_root) / "deps"
        os.environ["FINN_DEPS_DIR"] = str(default_deps_dir)
        if not default_deps_dir.exists():
            warnings.warn(
                f"FINN_DEPS_DIR set to {default_deps_dir}, but directory does not exist yet. "
                "Dependencies will need to be fetched before some operations can work. "
                "Run ./fetch-repos.sh or use the Docker container for full functionality."
            )


# Run validation on import
_validate_env_vars()

# Version information
try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"