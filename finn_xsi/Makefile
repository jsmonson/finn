#############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# @brief	Build FINN XSI Python bindings
# @author	Thomas B. Preußer <thomas.preusser@amd.com>
# @author	Yaman Umuroglu <yaman.umuroglu@amd.com>
#############################################################################
#
# NOTE: This Makefile is maintained for backwards compatibility only.
# For new installations, please use the Python-based build system:
#   python -m finn.xsi.setup
#
# The Python approach provides better integration with virtual environments
# and automatically detects the correct Python and pybind11 include paths.
#############################################################################

.PHONY: clean

#############################################################################
# Reproduction Build

default: xsi.so

# Python Binding
xsi.so: xsi_bind.cpp xsi_finn.cpp
	g++ --std=c++17 -Wall -O3 -shared -fPIC \
	$$(python3-config --includes) -I$(XILINX_VIVADO)/data/xsim/include \
	-o$@ $^ -ldl -lrt


#############################################################################
# Cleanup Working Directory
clean:
	rm -rf xsi.so
