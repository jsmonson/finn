############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# ##########################################################################
"""FINN XSI (Xilinx Simulation Interface) support module

This module provides utilities for RTL simulation support via finn_xsi.
The finn_xsi extension must be built separately using the setup command.

Usage:
    # Check if XSI support is available
    from finn import xsi
    if xsi.is_available():
        import finn_xsi.adapter

    # Or require XSI support (raises error if not available)
    from finn.xsi import require
    require()
    import finn_xsi.adapter
"""

import os
import sys
from pathlib import Path
from typing import Any, Optional


def is_available() -> bool:
    """Check if XSI (RTL simulation) support is available.

    Returns:
        bool: True if finn_xsi can be imported, False otherwise
    """
    # Check if xsi.so exists
    xsi_path = Path(os.environ["FINN_ROOT"]) / "finn_xsi"
    xsi_so = xsi_path / "xsi.so"
    if not xsi_so.exists():
        return False

    # Try loading the modules (this will cache them if successful)
    return _load_modules()


def require() -> None:
    """Ensure XSI support is available, raise helpful error if not.

    Raises:
        ImportError: If finn_xsi is not available with setup instructions
    """
    if not is_available():
        raise ImportError(
            "FINN XSI (RTL simulation) support not available.\n"
            "\n"
            "To set up XSI support:\n"
            "  1. Ensure Xilinx tools are available in your environment\n"
            "  2. Run: python -m finn.xsi.setup\n"
            "\n"
            "For detailed instructions, see:\n"
            "  https://finn.readthedocs.io/en/latest/rtl_simulation.html"
        )


def get_adapter() -> Any:
    """Get the finn_xsi adapter module if available.

    Returns:
        module: The finn_xsi.adapter module

    Raises:
        ImportError: If finn_xsi is not available
    """
    require()
    import finn_xsi.adapter

    return finn_xsi.adapter


# Optional: Provide status information
def status() -> None:
    """Print XSI support status information."""
    # Show expected path
    expected_path = Path(os.environ["FINN_ROOT"]) / "finn_xsi"
    print(f"Expected finn_xsi location: {expected_path}")

    # Check if module exists
    if expected_path.exists():
        print("✓ finn_xsi directory exists")
        xsi_so = expected_path / "xsi.so"
        if xsi_so.exists():
            print("✓ xsi.so compiled module found")
        else:
            print("✗ xsi.so compiled module NOT found (need to build)")
    else:
        print("✗ finn_xsi directory not found")

    # Check if available through finn.xsi
    if is_available():
        print("✓ XSI support is available through finn.xsi")
        print("  No manual PYTHONPATH configuration needed")

        # Show available functions
        available_funcs = [
            "compile_sim_obj",
            "get_simkernel_so",
            "load_sim_obj",
            "reset_rtlsim",
            "close_rtlsim",
            "rtlsim_multi_io",
            "SimEngine",
        ]
        print("  Available functions:")
        for func in available_funcs:
            if hasattr(sys.modules[__name__], func):
                print(f"    - xsi.{func}")
    else:
        print("✗ XSI support not available")
        print("  Run 'python -m finn.xsi.setup' to build")

    # Check if Xilinx tools are available
    import shutil

    if shutil.which("vivado"):
        print("✓ Vivado found in PATH")
    else:
        print("✗ Vivado not found in PATH")


# Cache for loaded modules
_adapter_module: Optional[Any] = None
_sim_engine_module: Optional[Any] = None
_xsi_module: Optional[Any] = None


def _load_modules() -> bool:
    """Load finn_xsi modules if available."""
    global _adapter_module, _sim_engine_module, _xsi_module

    if _adapter_module is not None:
        return True

    xsi_path = Path(os.environ["FINN_ROOT"]) / "finn_xsi"
    xsi_so = xsi_path / "xsi.so"

    if not xsi_so.exists():
        return False

    # Temporarily add to path for import
    path_added = str(xsi_path) not in sys.path
    if path_added:
        sys.path.insert(0, str(xsi_path))

    try:
        import finn_xsi.adapter
        import finn_xsi.sim_engine
        import xsi

        _xsi_module = xsi
        _adapter_module = finn_xsi.adapter
        _sim_engine_module = finn_xsi.sim_engine

        return True
    except ImportError as e:
        # Log the specific import error for debugging
        import logging

        logging.debug(f"Failed to import finn_xsi modules: {e}")
        return False
    except Exception as e:
        # Catch any unexpected errors during module loading
        import logging

        logging.warning(f"Unexpected error loading finn_xsi: {type(e).__name__}: {e}")
        return False
    finally:
        # Remove from path if we added it
        if path_added and str(xsi_path) in sys.path:
            try:
                sys.path.remove(str(xsi_path))
            except ValueError:
                pass  # Path was already removed somehow


# List of functions to wrap from finn_xsi.adapter
_ADAPTER_FUNCTIONS = [
    "locate_glbl",
    "compile_sim_obj",
    "get_simkernel_so",
    "load_sim_obj",
    "reset_rtlsim",
    "close_rtlsim",
    "rtlsim_multi_io",
]


def __getattr__(name: str) -> Any:
    """Dynamically wrap finn_xsi.adapter functions."""
    if name in _ADAPTER_FUNCTIONS:

        def wrapper(*args, **kwargs):
            if not _load_modules():
                raise ImportError("finn_xsi not available. Run: python -m finn.xsi.setup")
            return getattr(_adapter_module, name)(*args, **kwargs)

        wrapper.__name__ = name
        wrapper.__doc__ = f"Wrapper for finn_xsi.adapter.{name}"
        return wrapper
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


# SimEngine class wrapper
class SimEngine:
    """Wrapper for finn_xsi.sim_engine.SimEngine."""

    def __init__(self, *args, **kwargs):
        if not _load_modules():
            raise ImportError("finn_xsi not available. Run: python -m finn.xsi.setup")
        self._engine = _sim_engine_module.SimEngine(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._engine, name)
