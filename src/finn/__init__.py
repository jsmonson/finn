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


def _setup_ld_library_path():
    """Set up LD_LIBRARY_PATH for Vivado and Vitis libraries."""
    ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")
    paths_to_add = []

    # Add Vivado library path if XILINX_VIVADO is set
    vivado_path = os.environ.get("XILINX_VIVADO")
    if vivado_path:
        vivado_lib = Path(vivado_path) / "lib" / "lnx64.o"
        if vivado_lib.exists():
            paths_to_add.append(str(vivado_lib))

        # Also add standard system lib path
        system_lib = Path("/lib/x86_64-linux-gnu")
        if system_lib.exists():
            paths_to_add.append(str(system_lib))

    # Add Vitis FPO library path if VITIS_PATH is set
    vitis_path = os.environ.get("VITIS_PATH")
    if vitis_path:
        vitis_fpo = Path(vitis_path) / "lnx64" / "tools" / "fpo_v7_1"
        if vitis_fpo.exists():
            paths_to_add.append(str(vitis_fpo))

    # Update LD_LIBRARY_PATH if we have paths to add
    if paths_to_add:
        existing_paths = ld_library_path.split(":") if ld_library_path else []
        # Only add paths that aren't already present
        for path in paths_to_add:
            if path not in existing_paths:
                existing_paths.append(path)
        os.environ["LD_LIBRARY_PATH"] = ":".join(existing_paths)


# Run validation on import
_validate_env_vars()
_setup_ld_library_path()

# Version information
try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"