# FINN Plugin System

The FINN plugin system provides a unified interface for registering and discovering FINN components - transformations, kernels (HWCustomOp), and backends (HLS/RTL implementations) - making them accessible to both QONNX and external tools like BrainSmith.

## Overview

The plugin system supports three types of components:

1. **Transforms** - Graph transformations that optimize or modify ONNX models
2. **Kernels** - Hardware custom operations (HWCustomOp) that define FPGA-accelerated layers
3. **Backends** - HLS or RTL implementations of kernels

### Architecture

```
finn/plugin/
├── __init__.py      # Public API exports
├── adapters.py      # Decorators for transforms, kernels, backends
├── registry.py      # Plugin registry with dynamic discovery
└── README.md        # This file
```

## Transforms

### Registering a Transform

Use the `@transform` decorator to register FINN transformations:

```python
from finn.plugin import transform
from qonnx.transformation.base import Transformation

@transform(
    name="AbsorbAddIntoMultiThreshold",
    stage="topology_optimization",
    description="Absorb preceding Add ops into MultiThreshold"
)
class AbsorbAddIntoMultiThreshold(Transformation):
    def apply(self, model):
        # Transform implementation
        return (model, modified)
```

### Transform Parameters

- `name` (required): Transform name used for discovery
- `stage` (required): Compilation stage where transform applies
- `description` (optional): Human-readable description
- `author` (optional): Author name or organization
- `version` (optional): Version string (e.g., "1.0.0")

### Standard Compilation Stages

- `graph_cleanup` - Initial graph preparation and cleanup
- `topology_optimization` - Topology-level optimizations (streamlining)
- `kernel_mapping` - Mapping operations to hardware kernels
- `kernel_optimization` - Kernel-level optimizations
- `graph_optimization` - Graph-level and dataflow optimizations

## Kernels

### Registering a Kernel

Use the `@kernel` decorator to register HWCustomOp implementations:

```python
from finn.plugin import kernel
from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp

@kernel(
    name="MatrixVectorActivation",
    op_type="MatrixVectorActivation",
    description="Matrix-vector activation layer for neural networks"
)
class MatrixVectorActivation(HWCustomOp):
    def get_nodeattr_types(self):
        # Define node attributes
        return {
            "MW": int,
            "MH": int,
            "SIMD": int,
            "PE": int,
            # ... other attributes
        }
```

### Kernel Parameters

- `name` (required): Kernel name for plugin registry
- `op_type` (required): ONNX operation type this kernel implements
- `description` (optional): Human-readable description
- `domain` (optional): ONNX domain (default: "finn.custom_op.fpgadataflow")
- `author` (optional): Author name or organization
- `version` (optional): Version string

## Backends

### Registering a Backend

Use the `@backend` decorator to register backend implementations:

```python
from finn.plugin import backend
from finn.custom_op.fpgadataflow.hlsbackend import HLSBackend

@backend(
    name="MatrixVectorActivation_hls",
    kernel="MatrixVectorActivation",  # Associates with kernel
    backend_type="hls",              # "hls" or "rtl"
    description="HLS implementation of matrix-vector activation"
)
class MatrixVectorActivation_hls(MatrixVectorActivation, HLSBackend):
    def generate_params(self):
        # HLS-specific parameter generation
        pass
```

### Backend Parameters

- `name` (required): Backend name for registry
- `kernel` (required): Name of the kernel this backend implements
- `backend_type` (required): Type of backend - must be "hls" or "rtl"
- `description` (optional): Human-readable description
- `author` (optional): Author name or organization
- `version` (optional): Version string

### Backend Association Pattern

The plugin system uses a flexible association pattern:

1. **Kernels** are registered independently without listing backends
2. **Backends** declare which kernel they implement via the `kernel` parameter
3. **Registry** dynamically discovers available backends for each kernel

Benefits:
- Add new backends without modifying kernel definitions
- Support multiple backend types per kernel
- Clean separation of concerns

## Discovery and Usage

### Transform Discovery

```python
from finn.plugin import get_finn_registry

registry = get_finn_registry()

# Get specific transform
transform_cls = registry.get_transform("AbsorbAddIntoMultiThreshold")

# List all transforms
for name, cls in registry.list_transforms():
    metadata = cls._plugin_metadata
    print(f"{name}: stage={metadata['stage']}")

# Filter transforms by stage
topology_transforms = [
    (name, cls) for name, cls in registry.list_transforms()
    if cls._plugin_metadata.get('stage') == 'topology_optimization'
]
```

### Kernel and Backend Discovery

```python
# Get a specific kernel
mvau_kernel = registry.get_kernel("MatrixVectorActivation")

# Discover available backends for a kernel
backends = registry.get_backends_for_kernel("MatrixVectorActivation")
# Returns: {"hls": MatrixVectorActivation_hls, "rtl": MatrixVectorActivation_rtl}

# Get a specific backend
hls_backend = registry.get_backend("MatrixVectorActivation", "hls")

# List all kernels
for name, kernel_cls in registry.list_kernels():
    available_backends = registry.get_backends_for_kernel(name)
    print(f"Kernel: {name}, Backends: {list(available_backends.keys())}")

# Find kernels by ONNX op type
mvau_kernels = registry.get_kernels_by_op_type("MatrixVectorActivation")
```

## Integration with External Systems

### QONNX Integration

Transforms and kernels automatically register with QONNX:

```python
# Transforms register with QONNX transformation registry
from qonnx.transformation.registry import get_transformation
transform = get_transformation("AbsorbAddIntoMultiThreshold")

# Kernels register with QONNX custom op registry
from qonnx.custom_op.registry import getCustomOp
op_inst = getCustomOp(onnx_node)  # Works if node has correct domain/op_type
```

### BrainSmith Integration

BrainSmith can discover FINN components using the `finn:` prefix:

```python
# In BrainSmith steps
@finn_step(
    name="streamlining",
    transforms=[
        "finn:AbsorbSignBiasIntoMultiThreshold",
        "finn:MoveScalarMulPastMatMul",
        "finn:CollapseRepeatedMul"
    ]
)
def streamlining_step(model, cfg, transforms):
    # Transforms are automatically resolved and injected
    pass

# In BrainSmith DSE
from brainsmith.plugin.query import get_unified_query
query = get_unified_query()

# Resolve FINN kernel
kernel_cls = query.resolve_kernel_name("finn:MatrixVectorActivation")
```

## Complete Example

Here's a complete example showing all three component types:

```python
from finn.plugin import transform, kernel, backend
from qonnx.transformation.base import Transformation
from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp
from finn.custom_op.fpgadataflow.hlsbackend import HLSBackend
from finn.custom_op.fpgadataflow.rtlbackend import RTLBackend

# 1. Register a transform
@transform(
    name="OptimizeMatrixOps",
    stage="kernel_optimization",
    description="Optimize matrix operations for hardware"
)
class OptimizeMatrixOps(Transformation):
    def apply(self, model):
        # Optimization logic
        return (model, False)

# 2. Register a kernel
@kernel(
    name="OptimizedMatMul",
    op_type="MatMul",
    description="Optimized matrix multiplication kernel"
)
class OptimizedMatMul(HWCustomOp):
    def get_nodeattr_types(self):
        return {"rows": int, "cols": int}

# 3. Register HLS backend
@backend(
    name="OptimizedMatMul_hls",
    kernel="OptimizedMatMul",
    backend_type="hls",
    description="HLS implementation"
)
class OptimizedMatMul_hls(OptimizedMatMul, HLSBackend):
    def generate_params(self):
        # HLS parameter generation
        pass

# 4. Register RTL backend
@backend(
    name="OptimizedMatMul_rtl",
    kernel="OptimizedMatMul",
    backend_type="rtl",
    description="RTL implementation"
)
class OptimizedMatMul_rtl(OptimizedMatMul, RTLBackend):
    def generate_hdl(self):
        # RTL generation
        pass

# Usage
from finn.plugin import get_finn_registry
registry = get_finn_registry()

# Get transform
transform = registry.get_transform("OptimizeMatrixOps")

# Get kernel and its backends
kernel = registry.get_kernel("OptimizedMatMul")
backends = registry.get_backends_for_kernel("OptimizedMatMul")
print(f"Available backends: {list(backends.keys())}")  # ['hls', 'rtl']
```

## Best Practices

1. **Consistent Naming** - Use clear, descriptive names for all components
2. **Stage Assignment** - Always specify appropriate stages for transforms
3. **Backend Declaration** - Backends should always declare their kernel
4. **Documentation** - Include descriptions and docstrings
5. **Version Management** - Use version strings for tracking changes
6. **Domain Specification** - Use appropriate ONNX domains for kernels

## Summary

The FINN plugin system provides:
- Unified registration for transforms, kernels, and backends
- Dynamic discovery of available implementations
- Seamless integration with QONNX infrastructure
- External tool accessibility (BrainSmith, etc.)
- Flexible backend association pattern

This creates a maintainable and extensible system for managing FINN's compilation pipeline and hardware implementations.