# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FINN (Fast, Scalable Quantized Neural Network Inference) is a framework for deploying quantized neural networks on FPGAs. It transforms ONNX models into synthesized FPGA implementations through a series of graph transformations.

**Key Principle**: FINN uses Docker exclusively for execution. All development, testing, and builds must run inside the Docker container.

## Essential Commands

### Docker Container Management

```bash
# Build and enter interactive container
./run-docker.sh

# Run full test suite (inside Docker)
./run-docker.sh test

# Run quick tests (non-Vivado, non-slow tests)
./run-docker.sh quicktest

# Run Jupyter notebooks
./run-docker.sh notebook

# Build a dataflow design
./run-docker.sh build_dataflow /path/to/build/config
```

### Testing (Inside Docker Container)

```bash
# Run all tests
pytest

# Run tests with markers (exclude slow/Vivado tests)
pytest -m "not slow and not vivado"

# Run specific test file
pytest tests/fpgadataflow/test_convert_to_hw_layers.py

# Run single test
pytest tests/util/test_basic.py::test_pynq_part_map -v

# Common marker combinations
pytest -m "streamline"           # Streamlining transformations
pytest -m "fpgadataflow"         # FPGA dataflow tests
pytest -m "transform"            # Transformation tests
pytest -m "not vivado"           # Skip Vivado-dependent tests
```

### Python Development (Inside Docker)

```bash
# Install package in development mode (done by entrypoint)
pip install -e .

# Run the build_dataflow CLI
build_dataflow /path/to/config

# Format code (pre-commit hooks)
pre-commit run --all-files
```

## Architecture Overview

FINN transforms ONNX models to FPGA bitfiles through a **transformation-based pipeline**. Understanding this architecture is critical for effective development.

### Core Abstractions

**ModelWrapper** (from QONNX dependency):
- Wraps ONNX models with metadata storage
- All FINN operations work with `ModelWrapper`
- Stores synthesis parameters, estimates, and measured values

**Transformations**:
- Composable graph modifications: `model = model.transform(TransformClass())`
- Each transformation implements `apply(model) -> (model, modified_flag)`
- Located in `src/finn/transformation/`

**HWCustomOp** (Hardware Custom Operators):
- Base class for hardware-synthesizable operations
- Each operator has HLS and RTL variants (e.g., `MVAU_hls`, `MVAU_rtl`)
- Located in `src/finn/custom_op/fpgadataflow/`

### Build Pipeline Flow (19 Steps)

The standard pipeline (`default_build_dataflow_steps`) progresses through 5 stages:

**Stage 1: Model Preparation**
1. `step_qonnx_to_finn` - Convert QONNX quantization nodes to FINN
2. `step_tidy_up` - Shape inference, constant folding, cleanup
3. `step_streamline` - Graph optimization (constant propagation, operation fusion)

**Stage 2: Hardware Mapping**
4. `step_convert_to_hw` - Convert ops to HWCustomOp (MatMul → MVAU, MultiThreshold → Thresholding)
5. `step_create_dataflow_partition` - Separate accelerator from CPU ops into StreamingDataflowPartition

**Stage 3: Specialization**
6. `step_specialize_layers` - Choose HLS vs RTL per operator
7. `step_target_fps_parallelization` - Auto-determine PE/SIMD for target throughput
8. `step_apply_folding_config` - Apply parallelization config

**Stage 4: Code Generation**
9. `step_minimize_bit_width` - Optimize bit-widths
10. `step_generate_estimate_reports` - Performance/resource estimates
11. `step_hw_codegen` - Generate HLS C++ and Verilog
12. `step_hw_ipgen` - Run HLS synthesis to create IP blocks

**Stage 5: Integration**
13. `step_set_fifo_depths` - Insert and size FIFOs
14. `step_create_stitched_ip` - Integrate all IP blocks
15. `step_measure_rtlsim_performance` - RTL simulation verification
16. `step_out_of_context_synthesis` - Synthesis timing analysis
17. `step_synthesize_bitfile` - Full place & route
18. `step_make_driver` - Generate software driver
19. `step_deployment_package` - Package for deployment

### Key Hardware Operators

**Computational**:
- `MVAU` - Matrix-Vector Activation Unit (main compute kernel)
- `Thresholding` - Multi-threshold quantized activation
- `ConvolutionInputGenerator` - Sliding window to matrix conversion
- `Pool` - Pooling operations

**Data Movement**:
- `StreamingFIFO` - Pipeline buffering
- `StreamingDataWidthConverter` - AXI width alignment
- `DuplicateStreams` - Fan-out
- `AddStreams` - Element-wise addition

## Directory Structure

```
src/finn/
├── builder/              # Build orchestration
│   ├── build_dataflow.py              # Main build entry point
│   ├── build_dataflow_config.py       # DataflowBuildConfig class
│   └── build_dataflow_steps.py        # 19 pipeline step implementations
├── core/                 # Execution backends
│   ├── onnx_exec.py                   # NumPy CPU execution
│   ├── rtlsim_exec.py                 # RTL simulation
│   └── throughput_test.py             # Performance measurement
├── transformation/       # Graph transformations
│   ├── fpgadataflow/                  # FPGA-specific transforms
│   │   ├── convert_to_hw_layers.py    # Infer hardware operators
│   │   ├── create_dataflow_partition.py # CPU/FPGA partitioning
│   │   ├── specialize_layers.py       # HLS vs RTL selection
│   │   ├── set_folding.py             # Parallelization config
│   │   ├── prepare_ip.py, hlssynth_ip.py # Code gen and synthesis
│   │   ├── insert_fifo.py, set_fifo_depths.py # Pipeline insertion
│   │   └── create_stitched_ip.py      # IP integration
│   ├── streamline/                    # Graph optimization
│   └── qonnx/                         # QONNX integration
├── custom_op/fpgadataflow/ # Hardware operators
│   ├── hwcustomop.py                  # Base class for HW ops
│   ├── hls/                           # HLS implementations
│   │   ├── matrixvectoractivation_hls.py
│   │   ├── thresholding_hls.py
│   │   └── ...
│   └── rtl/                           # RTL implementations
│       ├── matrixvectoractivation_rtl.py
│       └── ...
├── analysis/fpgadataflow/ # Analysis passes
│   ├── dataflow_performance.py        # Critical path analysis
│   ├── exp_cycles_per_layer.py        # Cycle estimation
│   └── res_estimation.py              # Resource estimation
└── util/                 # Utilities
    ├── basic.py                       # Environment setup
    └── fpgadataflow.py                # FPGA helpers

tests/                    # Test suite
├── brevitas/            # Brevitas export tests
├── end2end/             # Full build tests
├── fpgadataflow/        # Hardware operator tests
├── transformation/      # Transformation tests
└── util/                # Utility tests
```

## Development Workflow

### Adding a New Transformation

1. Create subclass of `Transformation` in `src/finn/transformation/fpgadataflow/`
2. Implement `apply(model) -> (model, modified)` method
3. Add to build pipeline in `build_dataflow_steps.py` if needed
4. Add unit test in `tests/transformation/`

### Adding a New Hardware Operator

1. Create HLS variant: `src/finn/custom_op/fpgadataflow/hls/myop_hls.py`
2. Create RTL variant: `src/finn/custom_op/fpgadataflow/rtl/myop_rtl.py`
3. Both inherit from `HWCustomOp` and implement required methods
4. Create inference transformation to recognize and insert operator
5. Add tests in `tests/fpgadataflow/`

### Debugging Build Pipeline

Intermediate models are saved when `save_intermediate_models=True` in `DataflowBuildConfig`:
```
output_dir/intermediate_models/
├── tidy_up.onnx
├── streamlined.onnx
├── convert_to_hw.onnx
├── apply_folding_config.onnx
└── ... (one per pipeline step)
```

Load and inspect with:
```python
from finn.core.modelwrapper import ModelWrapper
model = ModelWrapper("path/to/intermediate.onnx")
```

## Important Patterns

### FINN vs QONNX Relationship

- **QONNX** provides: ModelWrapper, base transformations, quantization nodes
- **FINN** extends: Hardware operators, FPGA transformations, synthesis

Always import `ModelWrapper` from QONNX:
```python
from qonnx.core.modelwrapper import ModelWrapper
```

### Dual-Backend Strategy

Every computational operator has both HLS and RTL implementations:
- **HLS**: Flexible, easier to modify, less predictable resource usage
- **RTL**: Optimized, precise timing, requires more effort to change

Set via `specialize_layers` step or manually via node attribute `preferred_impl_style`.

### Streaming Dataflow Model

Operations consume/produce AXI streams. Key implications:
- FIFOs decouple pipeline stages
- Throughput determined by slowest stage (critical path)
- Parallelization (PE/SIMD) is per-layer

### Configuration Files

Generated during build:
- `template_specialize_layers_config.json` - HLS/RTL choices template
- `auto_folding_config.json` - Auto-generated PE/SIMD settings
- `final_hw_config.json` - Final architecture parameters
- `estimate_*.json` - Performance/resource estimates

## Environment Variables

Key variables set by `run-docker.sh`:
- `FINN_ROOT` - Repository root
- `FINN_BUILD_DIR` - Build output directory (default: `/tmp/finn_dev_<user>`)
- `VIVADO_PATH`, `VITIS_PATH`, `HLS_PATH` - Xilinx tool paths
- `NUM_DEFAULT_WORKERS` - Parallel worker count

## Common Issues

**Issue**: "No module named 'finn'"
**Fix**: Ensure you're inside Docker container and `pip install -e .` ran successfully

**Issue**: Vivado/HLS synthesis hangs
**Fix**: Docker launched with `--init` flag (already in `run-docker.sh`)

**Issue**: Tests fail with "Xilinx tools not found"
**Fix**: Set `FINN_XILINX_PATH` and `FINN_XILINX_VERSION` before running `./run-docker.sh`

**Issue**: "Permission denied" for build outputs
**Fix**: Check `FINN_HOST_BUILD_DIR` permissions match Docker user

## Testing Strategy

Tests use pytest markers to categorize by requirements:
- `slow` - Long-running tests
- `vivado` - Requires Vivado
- `vitis` - Requires Vitis
- `board` - Requires PYNQ board
- `fpgadataflow` - FPGA operator tests
- `transform` - Transformation tests
- `streamline` - Streamlining tests

Use markers to run subset: `pytest -m "fpgadataflow and not vivado"`

## Key Dependencies

- **QONNX**: Quantized ONNX foundation (graph wrapper, base transformations)
- **ONNX**: Model format
- **Brevitas**: Quantization-aware training (optional, for model import)
- **Vivado/Vitis/HLS**: Xilinx synthesis tools (mounted from host)
- **PyTorch**: For model import and testing

## Additional Resources

- Full documentation: https://finn.readthedocs.io
- Tutorial notebooks: `notebooks/` (run via `./run-docker.sh notebook`)
- Example networks: https://github.com/Xilinx/finn-examples
- Discussions: https://github.com/Xilinx/finn/discussions
