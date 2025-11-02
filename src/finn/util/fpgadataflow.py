# Copyright (c) 2020 Xilinx, Inc.
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
# * Neither the name of Xilinx nor the names of its
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

from qonnx.util.basic import get_by_name, is_finn_op


def is_fpgadataflow_node(node):
    """Returns True if given node is fpgadataflow node. Otherwise False.

    Recognizes nodes with backend attribute set to "fpgadataflow" (generic),
    "hls" (HLS-specialized), or "rtl" (RTL-specialized).
    """
    is_node = False
    if node is not None:
        if is_finn_op(node.domain):
            n_backend = get_by_name(node.attribute, "backend")
            if n_backend is not None:
                backend_value = n_backend.s.decode("UTF-8")
                # Recognize all fpgadataflow nodes: generic, HLS, and RTL
                if backend_value in ("fpgadataflow", "hls", "rtl"):
                    is_node = True

    return is_node


def is_hls_node(node):
    """Returns True if given node is hls node. Otherwise False.

    Checks the backend attribute first (modern approach), then falls back
    to domain-based detection (legacy approach) for backwards compatibility.
    """
    is_node = False
    if node is not None:
        n_backend = get_by_name(node.attribute, "backend")
        if n_backend is not None:
            backend_value = n_backend.s.decode("UTF-8")
            # Modern approach: backend attribute indicates implementation style
            if backend_value == "hls":
                is_node = True
            # Legacy approach: domain indicates implementation style
            elif backend_value == "fpgadataflow":
                if node.domain.endswith(".custom_op.fpgadataflow.hls") or (
                    node.domain.startswith("brainsmith.kernels") and node.domain.endswith(".hls")
                ):
                    is_node = True

    return is_node


def is_rtl_node(node):
    """Returns True if given node is rtl node. Otherwise False.

    Checks the backend attribute first (modern approach), then falls back
    to domain-based detection (legacy approach) for backwards compatibility.
    """
    is_node = False
    if node is not None:
        n_backend = get_by_name(node.attribute, "backend")
        if n_backend is not None:
            backend_value = n_backend.s.decode("UTF-8")
            # Modern approach: backend attribute indicates implementation style
            if backend_value == "rtl":
                is_node = True
            # Legacy approach: domain indicates implementation style
            elif backend_value == "fpgadataflow":
                if node.domain.endswith(".custom_op.fpgadataflow.rtl") or (
                    node.domain.startswith("brainsmith.kernels") and node.domain.endswith(".rtl")
                ):
                    is_node = True

    return is_node
