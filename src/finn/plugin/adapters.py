"""
FINN Plugin Adapters

Adapter decorators that register FINN plugins with both QONNX and FINN registries.
"""

import logging
from typing import Type, Optional

logger = logging.getLogger(__name__)

# Define valid stages matching BrainSmith
VALID_STAGES = ["cleanup", "topology_opt", "kernel_opt", "dataflow_opt"]


def transform(
    name: str,
    stage: Optional[str] = None,
    description: Optional[str] = None,
    author: Optional[str] = None,
    version: Optional[str] = None
):
    """
    Decorator for registering FINN transforms with both QONNX and FINN registries.
    
    This adapter:
    1. Stores FINN metadata on the class
    2. Registers with QONNX's transformation registry
    3. Registers with FINN's plugin registry
    
    Args:
        name: Name of the transform (required)
        stage: Compilation stage - one of: cleanup, topology_opt, kernel_opt, dataflow_opt
        description: Human-readable description
        author: Author name or organization
        version: Version string (e.g., "1.0.0")
    
    Example:
        @transform(
            name="AbsorbSignBiasIntoMultiThreshold",
            stage="topology_optimization",
            description="Absorb scalar bias into MultiThreshold"
        )
        class AbsorbSignBiasIntoMultiThreshold(Transformation):
            ...
    """
    def decorator(cls: Type) -> Type:
        # No strict validation - just log warnings
        try:
            from qonnx.transformation.base import Transformation
            if not issubclass(cls, Transformation):
                logger.debug(f"Transform '{name}' does not inherit from Transformation base class")
        except ImportError:
            logger.debug("QONNX not available for validation")
        
        # Warn about non-standard stages but allow them
        if stage and stage not in VALID_STAGES:
            logger.debug(
                f"Transform '{name}' uses non-standard stage '{stage}'. "
                f"Standard stages are: {', '.join(VALID_STAGES)}"
            )
        
        # Store FINN metadata on class
        cls._plugin_metadata = {
            "type": "transform",
            "name": name,
            "stage": stage,
            "description": description,
            "author": author,
            "version": version,
            "framework": "finn"
        }
        
        # Register with QONNX transformation registry
        try:
            from qonnx.transformation.registry import register_transformation
            # Pass all metadata except finn-specific ones
            qonnx_kwargs = {
                "description": description,
                "author": author,
                "version": version
            }
            cls = register_transformation(name, **qonnx_kwargs)(cls)
            logger.debug(f"Registered transform with QONNX: {name}")
        except Exception as e:
            logger.debug(f"QONNX registration skipped: {e}")
        
        # Register with FINN plugin registry using new method
        try:
            from .registry import get_finn_registry
            registry = get_finn_registry()
            registry.register("transform", name, cls, 
                            stage=stage,
                            description=description,
                            author=author,
                            version=version)
            logger.debug(f"Registered transform with FINN: {name} (stage: {stage})")
        except Exception as e:
            logger.debug(f"FINN registration skipped: {e}")
        
        return cls
    
    return decorator


def kernel(
    name: str,
    op_type: str,
    description: Optional[str] = None,
    domain: str = "finn.custom_op.fpgadataflow",
    author: Optional[str] = None,
    version: Optional[str] = None
):
    """
    Decorator for registering FINN kernels (HWCustomOp subclasses).
    
    This adapter:
    1. Stores FINN metadata on the class
    2. Registers with QONNX's custom op registry (if available)
    3. Registers with FINN's plugin registry
    
    Args:
        name: Name of the kernel (required)
        op_type: ONNX operation type this kernel implements (required)
        description: Human-readable description
        domain: ONNX domain for custom op registration
        author: Author name or organization
        version: Version string (e.g., "1.0.0")
    
    Example:
        @kernel(
            name="MatrixVectorActivation",
            op_type="MatrixVectorActivation",
            description="Matrix-vector activation layer"
        )
        class MatrixVectorActivation(HWCustomOp):
            ...
    """
    def decorator(cls: Type) -> Type:
        # Store FINN metadata on class
        cls._plugin_metadata = {
            "type": "kernel",
            "name": name,
            "op_type": op_type,
            "domain": domain,
            "description": description,
            "author": author,
            "version": version,
            "framework": "finn"
        }
        
        # Register with QONNX custom op registry
        try:
            from qonnx.custom_op.registry import register_op
            cls = register_op(domain, op_type)(cls)
            logger.debug(f"Registered kernel with QONNX: {op_type} in domain {domain}")
        except Exception as e:
            logger.debug(f"QONNX registration skipped: {e}")
        
        # Register with FINN plugin registry using new method
        try:
            from .registry import get_finn_registry
            registry = get_finn_registry()
            registry.register("kernel", name, cls,
                            op_type=op_type,
                            domain=domain,
                            description=description,
                            author=author,
                            version=version)
            logger.debug(f"Registered kernel with FINN: {name}")
        except Exception as e:
            logger.debug(f"FINN registration skipped: {e}")
        
        return cls
    
    return decorator


def backend(
    name: str,
    kernel: str,
    backend_type: str,
    description: Optional[str] = None,
    author: Optional[str] = None,
    version: Optional[str] = None
):
    """
    Decorator for registering FINN backend implementations.
    
    Backends declare which kernel they implement, allowing dynamic discovery
    of available backends for each kernel.
    
    Args:
        name: Name of the backend (required)
        kernel: Name of the kernel this backend implements (required)
        backend_type: Type of backend - "hls" or "rtl" (required)
        description: Human-readable description
        author: Author name or organization
        version: Version string (e.g., "1.0.0")
    
    Example:
        @backend(
            name="MatrixVectorActivationHLS",
            kernel="MatrixVectorActivation",
            backend_type="hls",
            description="HLS backend for matrix-vector activation"
        )
        class MatrixVectorActivation_hls(MatrixVectorActivation, HLSBackend):
            ...
    """
    def decorator(cls: Type) -> Type:
        # Warn about non-standard backend types but allow them
        if backend_type not in ["hls", "rtl"]:
            logger.debug(f"Backend '{name}' uses non-standard type '{backend_type}'. Standard types are: hls, rtl")
        
        # Store FINN metadata on class
        cls._plugin_metadata = {
            "type": "backend",
            "name": name,
            "kernel": kernel,
            "backend_type": backend_type,
            "description": description,
            "author": author,
            "version": version,
            "framework": "finn"
        }
        
        # Register with FINN plugin registry using new method
        try:
            from .registry import get_finn_registry
            registry = get_finn_registry()
            registry.register("backend", name, cls,
                            kernel=kernel,
                            backend_type=backend_type,
                            description=description,
                            author=author,
                            version=version)
            logger.debug(f"Registered backend with FINN: {name} for kernel {kernel} (type: {backend_type})")
        except Exception as e:
            logger.debug(f"FINN registration skipped: {e}")
        
        return cls
    
    return decorator