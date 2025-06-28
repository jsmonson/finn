"""
FINN Plugin Adapters

Adapter decorators that register FINN plugins with both QONNX and FINN registries.
"""

import logging
from typing import Type, Optional

logger = logging.getLogger(__name__)


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
        stage: Compilation stage where transform applies (topology_optimization, graph_optimization, etc.)
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
        # Validate it's a Transformation
        try:
            from qonnx.transformation.base import Transformation
            if not issubclass(cls, Transformation):
                raise ValueError(
                    f"Transform '{name}' must inherit from qonnx.transformation.base.Transformation"
                )
        except ImportError:
            logger.warning("QONNX not available, skipping transform validation")
        
        # Use stage directly
        effective_stage = stage
        
        # Store FINN metadata on class
        cls._plugin_metadata = {
            "type": "transform",
            "name": name,
            "stage": effective_stage,
            "description": description,
            "author": author,
            "version": version
        }
        
        # Register with QONNX transformation registry
        try:
            from qonnx.transformation.registry import register_transformation
            cls = register_transformation(name)(cls)
            logger.info(f"Registered transform with QONNX: {name}")
        except ImportError:
            logger.warning(f"QONNX registry not available, skipping QONNX registration for {name}")
        except Exception as e:
            logger.error(f"Failed to register {name} with QONNX: {e}")
        
        # Register with FINN plugin registry
        try:
            from .registry import FinnPluginRegistry
            FinnPluginRegistry.register(cls)
            logger.info(f"Registered transform with FINN: {name} (stage: {effective_stage})")
        except Exception as e:
            logger.error(f"Failed to register {name} with FINN: {e}")
        
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
            "version": version
        }
        
        # Register with QONNX custom op registry if available
        try:
            from qonnx.custom_op.registry import register_op
            cls = register_op(domain, op_type)(cls)
            logger.info(f"Registered kernel with QONNX: {op_type} in domain {domain}")
        except ImportError:
            logger.warning(f"QONNX not available, skipping custom op registration for {name}")
        except Exception as e:
            logger.error(f"Failed to register {name} with QONNX: {e}")
        
        # Register with FINN plugin registry
        try:
            from .registry import FinnPluginRegistry
            FinnPluginRegistry.register(cls)
            logger.info(f"Registered kernel with FINN: {name}")
        except Exception as e:
            logger.error(f"Failed to register kernel {name} with FINN: {e}")
        
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
        # Validate backend type
        if backend_type not in ["hls", "rtl"]:
            raise ValueError(f"Invalid backend_type '{backend_type}'. Must be 'hls' or 'rtl'")
        
        # Store FINN metadata on class
        cls._plugin_metadata = {
            "type": "backend",
            "name": name,
            "kernel": kernel,
            "backend_type": backend_type,
            "description": description,
            "author": author,
            "version": version
        }
        
        # Register with FINN plugin registry
        try:
            from .registry import FinnPluginRegistry
            FinnPluginRegistry.register(cls)
            logger.info(f"Registered backend with FINN: {name} for kernel {kernel} (type: {backend_type})")
        except Exception as e:
            logger.error(f"Failed to register backend {name} with FINN: {e}")
        
        return cls
    
    return decorator