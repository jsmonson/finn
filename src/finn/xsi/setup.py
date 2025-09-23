"""Setup script for FINN XSI (RTL simulation) support.

This script builds and configures the finn_xsi C++ extension module
required for RTL simulation in FINN.

Usage:
    python -m finn.xsi.setup [options]

Options:
    --force    Force rebuild even if already built
    --clean    Clean build artifacts
    --check    Only check if build is needed
"""

import sys
import os
import subprocess
import shutil
import argparse
from pathlib import Path
from typing import List, Optional




def check_prerequisites() -> List[str]:
    """Check if required tools are available."""
    errors = []
    
    # Check for make
    if not shutil.which("make"):
        errors.append("'make' command not found. Please install build-essential or equivalent.")
    
    # Check for Xilinx tools
    if not shutil.which("vivado"):
        errors.append("'vivado' not found. Ensure Xilinx tools are in PATH.")
    
    # Check for C++ compiler
    if not shutil.which("g++") and not shutil.which("clang++"):
        errors.append("No C++ compiler found. Please install g++ or clang++.")
    
    return errors


def build_xsi(force: bool = False, verbose: bool = True) -> bool:
    """Build the finn_xsi extension module.
    
    Args:
        force: Force rebuild even if already built
        verbose: Print build output
        
    Returns:
        bool: True if build successful
    """
    finn_root = Path(os.environ["FINN_ROOT"])
    xsi_path = finn_root / "finn_xsi"
    
    if not xsi_path.exists():
        print(f"Error: finn_xsi source not found at {xsi_path}")
        return False
    
    # Check if already built
    if not force:
        xsi_so = xsi_path / "xsi.so"
        if xsi_so.exists():
            # Try importing to see if it works
            sys.path.insert(0, str(xsi_path))
            try:
                import xsi
                sys.path.pop(0)
                print("xsi.so is already built and working.")
                return True
            except ImportError:
                sys.path.pop(0)
                print("xsi.so exists but failed to import, rebuilding...")
        # else: Need to build
    
    print(f"Building finn_xsi in {xsi_path}...")
    
    # Run make
    env = os.environ.copy()
    result = subprocess.run(
        ["make"],
        cwd=xsi_path,
        env=env,
        capture_output=not verbose,
        text=True
    )
    
    if result.returncode != 0:
        print("Build failed!")
        if not verbose and result.stderr:
            print("Error output:", result.stderr)
        if not verbose and result.stdout:
            print("Build output:", result.stdout)
        print("\nCommon issues:")
        print("  - Ensure Xilinx Vivado is properly sourced")
        print("  - Check that pybind11 is available")
        print("  - Verify C++ compiler is installed")
        return False
    
    print("Build completed successfully.")
    return True


def verify_installation() -> bool:
    """Verify that finn_xsi can be imported and works."""
    finn_root = Path(os.environ["FINN_ROOT"])
    xsi_path = finn_root / "finn_xsi"
    
    # Check if xsi.so exists
    xsi_so = xsi_path / "xsi.so"
    if not xsi_so.exists():
        print(f"\n✗ Compiled extension xsi.so not found at {xsi_so}")
        return False
    
    # Temporarily add to path
    sys.path.insert(0, str(xsi_path))
    
    try:
        # Import the compiled C++ extension
        import xsi
        print("\n✓ xsi C++ extension module imports successfully")
        
        # Import the Python package
        import finn_xsi.adapter
        print("✓ finn_xsi.adapter imports successfully")
        
        # Check for basic functionality
        if hasattr(finn_xsi.adapter, "rtlsim_multi_io"):
            print("✓ RTL simulation functions available")
        
        return True
        
    except ImportError as e:
        print(f"\n✗ Failed to import modules: {e}")
        return False
    finally:
        sys.path.pop(0)


def clean_build() -> bool:
    """Clean build artifacts."""
    finn_root = Path(os.environ["FINN_ROOT"])
    xsi_path = finn_root / "finn_xsi"
    
    print(f"Cleaning build artifacts in {xsi_path}...")
    result = subprocess.run(["make", "clean"], cwd=xsi_path)
    return result.returncode == 0


def main() -> int:
    """Main setup function."""
    parser = argparse.ArgumentParser(
        description="Setup FINN XSI (RTL simulation) support"
    )
    parser.add_argument("--force", action="store_true",
                        help="Force rebuild even if already built")
    parser.add_argument("--clean", action="store_true",
                        help="Clean build artifacts and exit")
    parser.add_argument("--check", action="store_true",
                        help="Only check prerequisites")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress build output")
    
    args = parser.parse_args()
    
    # Clean and exit if requested
    if args.clean:
        if clean_build():
            print("Clean completed successfully.")
            return 0
        else:
            print("Clean failed.")
            return 1
    
    # Check prerequisites
    print("Checking prerequisites...")
    errors = check_prerequisites()
    
    if errors:
        print("Prerequisite check failed:")
        for error in errors:
            print(f"  ✗ {error}")
        print("Please resolve these issues and try again.")
        return 1
    
    print("✓ All prerequisites satisfied")
    
    if args.check:
        return 0
    
    # Build finn_xsi
    print("Building finn_xsi extension...")
    if not build_xsi(force=args.force, verbose=not args.quiet):
        print("Build failed. Please check the error messages above.")
        return 1
    
    # Verify installation
    if verify_installation():
        print("\nFINN XSI setup completed successfully!")
        return 0
    else:
        print("\nSetup completed but verification failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())