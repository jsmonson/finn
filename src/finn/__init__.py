# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of FINN nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

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