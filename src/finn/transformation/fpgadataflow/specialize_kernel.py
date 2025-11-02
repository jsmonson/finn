############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
#
############################################################################

import warnings
from onnx import helper
from qonnx.custom_op.registry import hasCustomOp
from qonnx.transformation.base import Transformation

from finn.custom_op.fpgadataflow.hlsbackend import HLSBackend
from finn.custom_op.fpgadataflow.rtlbackend import RTLBackend
from finn.util.basic import getHWCustomOp


class SpecializeKernel(Transformation):
    """Specialize a specific kernel to one of multiple backend variant classes.

    This transformation attempts to specialize a kernel to backend variants in priority order.
    It modifies:
    1. The op_type (adding _hls or _rtl suffix)
    2. The backend attribute (from "fpgadataflow" to "hls" or "rtl")
    3. The domain (to the variant's registered domain, e.g., finn.custom_op.fpgadataflow.hls)

    This supports both standard FINN domains and custom domains (e.g., brainsmith.kernels.*)
    by using the variant class's registered domain.
    """

    def __init__(self, kernel_class, backend_variants, fpgapart):
        """Initialize the SpecializeKernel transformation.

        Args:
            kernel_class: The base HWCustomOp class to specialize (e.g., MVAU)
            backend_variants: List of specialized variant classes in priority order
                            (e.g., [MVAU_rtl, MVAU_hls])
            fpgapart: FPGA part string for constraint checking
        """
        super().__init__()
        self.kernel_class = kernel_class
        self.backend_variants = backend_variants
        self.fpgapart = fpgapart

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        graph_modified = False

        # Get base kernel name for fast string matching
        kernel_name = self.kernel_class.__name__

        for node in graph.node:
            # Fast string check: only process nodes matching the kernel class name
            if node.op_type != kernel_name:
                node_ind += 1
                continue

            # Check if this is a generic fpgadataflow node (only instantiate after string match)
            node_inst = getHWCustomOp(node, model)
            if not hasattr(node_inst, 'get_nodeattr'):
                node_ind += 1
                continue

            try:
                backend_value = node_inst.get_nodeattr("backend")
            except Exception:
                node_ind += 1
                continue

            # Only specialize nodes that are still generic (backend="fpgadataflow")
            if backend_value != "fpgadataflow":
                node_ind += 1
                continue

            # Try each backend variant class in priority order
            selected_variant = None
            for variant_class in self.backend_variants:
                # Extract metadata from variant class
                variant_optype = variant_class.__name__
                variant_module = variant_class.__module__

                # Extract backend style - check suffix first, then inheritance
                if variant_optype.endswith("_rtl"):
                    backend_style = "rtl"
                elif variant_optype.endswith("_hls"):
                    backend_style = "hls"
                elif issubclass(variant_class, RTLBackend):
                    backend_style = "rtl"
                elif issubclass(variant_class, HLSBackend):
                    backend_style = "hls"
                else:
                    warnings.warn(
                        f"Cannot determine backend style for {variant_optype}: "
                        f"doesn't end with _hls/_rtl and doesn't inherit from HLSBackend/RTLBackend"
                    )
                    continue

                # Verify variant exists in registry
                # Module path like "finn.custom_op.fpgadataflow.hls.matrixvectoractivation_hls"
                # Extract domain by removing the final component (Python module name)
                variant_domain = variant_module.rsplit(".", 1)[0]

                if hasCustomOp(variant_domain, variant_optype):
                    # Check if constraints are met for this backend variant
                    if self._check_backend_constraints(node, backend_style, model):
                        selected_variant = (variant_class, variant_optype, backend_style, variant_domain)
                        break
                else:
                    # Warn if backend doesn't exist
                    warnings.warn(
                        f"Backend variant {variant_optype} not found in domain {variant_domain}"
                    )

            # If no backend could be selected, warn and skip
            if selected_variant is None:
                variant_names = [v.__name__ for v in self.backend_variants]
                warnings.warn(
                    f"Could not specialize kernel {node.name} (op_type={node.op_type}). "
                    f"None of the backend variants {variant_names} met constraints or were available."
                )
                node_ind += 1
                continue

            # Unpack selected variant
            variant_class, new_optype, backend_style, variant_domain = selected_variant

            # Create the specialized node with the variant's domain
            new_node = helper.make_node(
                new_optype,
                node.input,
                node.output,
                domain=variant_domain,  # Use variant's registered domain
                name=node.name,
            )

            # Copy all attributes except backend
            for attribute in node.attribute:
                if attribute.name != "backend":
                    new_node.attribute.append(attribute)

            # Set the new backend attribute value
            new_node.attribute.append(
                helper.make_attribute("backend", backend_style)
            )

            # Replace the node
            graph.node.insert(node_ind, new_node)
            graph.node.remove(node)
            graph_modified = True

            node_ind += 1

        return (model, graph_modified)

    def _check_backend_constraints(self, node, backend, model):
        """Check if the given backend meets all constraints for this node.

        Args:
            node: ONNX node to check
            backend: Backend name ("hls" or "rtl")
            model: ModelWrapper

        Returns:
            bool: True if constraints are met, False otherwise
        """
        # Import constraint checking functions from specialize_layers
        from finn.transformation.fpgadataflow.specialize_layers import (
            _dwc_determine_impl_style,
            _mvu_rtl_possible,
            _vvu_rtl_possible,
        )

        optype = node.op_type
        node_inst = getHWCustomOp(node, model)

        # For RTL backend, check specific constraints
        if backend == "rtl":
            if optype == "StreamingDataWidthConverter":
                return _dwc_determine_impl_style(node, model) == "rtl"
            elif optype == "MVAU":
                return _mvu_rtl_possible(node, self.fpgapart, model)
            elif optype == "VVAU":
                return _vvu_rtl_possible(node, self.fpgapart, model)
            # For other ops, RTL is always acceptable if it exists
            return True

        elif backend == "hls":
            # HLS is generally more flexible, but check for specific constraints
            if optype == "MVAU":
                # HLS MVAU doesn't support some features that RTL requires
                idt = node_inst.get_input_datatype(0)
                wdt = node_inst.get_input_datatype(1)
                # HLS supports smaller bitwidths that RTL doesn't
                if idt.bitwidth() < 2 or wdt.bitwidth() < 2:
                    return True  # Only HLS can handle this
            return True

        # Unknown backend
        return False
