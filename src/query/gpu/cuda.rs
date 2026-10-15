//! CUDA / `ROCm` scalar reductions via NVRTC + block tree reduce (`tetration-gpu` or `tetration-rocm`).

use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::sync::{Arc, Mutex, OnceLock};

use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::compile_ptx;

use super::GPU_VRAM_BUFFER_FRACTION;
use crate::query::fold::reduction::{ReductionKind, ScalarReductionResult};

const BLOCK_THREADS: u32 = 256;
const PTX_SRC: &str = r#"
extern "C" __global__ void block_reduce_sum(const float* in, float* out, int n) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    float acc = 0.f;
    for (int i = blockIdx.x * blockDim.x + tid; i < n; i += blockDim.x * gridDim.x) {
        acc += in[i];
    }
    sdata[tid] = acc;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            sdata[tid] += sdata[tid + offset];
        }
        __syncthreads();
    }
    if (tid == 0) {
        out[blockIdx.x] = sdata[0];
    }
}

extern "C" __global__ void block_reduce_min(const float* in, float* out, int n) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    float acc = __int_as_float(0x7f800000);
    for (int i = blockIdx.x * blockDim.x + tid; i < n; i += blockDim.x * gridDim.x) {
        acc = fminf(acc, in[i]);
    }
    sdata[tid] = acc;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            sdata[tid] = fminf(sdata[tid], sdata[tid + offset]);
        }
        __syncthreads();
    }
    if (tid == 0) {
        out[blockIdx.x] = sdata[0];
    }
}

extern "C" __global__ void block_reduce_max(const float* in, float* out, int n) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    float acc = -__int_as_float(0x7f800000);
    for (int i = blockIdx.x * blockDim.x + tid; i < n; i += blockDim.x * gridDim.x) {
        acc = fmaxf(acc, in[i]);
    }
    sdata[tid] = acc;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + offset]);
        }
        __syncthreads();
    }
    if (tid == 0) {
        out[blockIdx.x] = sdata[0];
    }
}
"#;

struct GpuKernels {
    sum: CudaFunction,
    min: CudaFunction,
    max: CudaFunction,
}

struct GpuState {
    ctx: Arc<CudaContext>,
    kernels: GpuKernels,
}

static GPU_STATES: OnceLock<Mutex<HashMap<usize, GpuState>>> = OnceLock::new();

static VISIBLE_DEVICES: OnceLock<usize> = OnceLock::new();

/// Number of visible accelerator devices (cached probe).
#[must_use]
pub(crate) fn visible_device_count() -> usize {
    *VISIBLE_DEVICES.get_or_init(|| {
        let n = (0..16).filter(|i| CudaContext::new(*i).is_ok()).count();
        n.max(1)
    })
}

fn with_gpu_state<R>(
    device_index: usize,
    f: impl FnOnce(&GpuState) -> Result<R, String>,
) -> Result<R, String> {
    let map = GPU_STATES.get_or_init(|| Mutex::new(HashMap::new()));
    let mut guard = map.lock().map_err(|_| "gpu_runtime_error".to_string())?;
    if let Entry::Vacant(slot) = guard.entry(device_index) {
        slot.insert(init_gpu(device_index)?);
    }
    let state = guard
        .get(&device_index)
        .ok_or_else(|| "gpu_runtime_error".to_string())?;
    f(state)
}

fn init_gpu(device_index: usize) -> Result<GpuState, String> {
    let ctx = CudaContext::new(device_index).map_err(|e| format!("{e}"))?;
    let ptx = compile_ptx(PTX_SRC).map_err(|e| format!("{e}"))?;
    let module = ctx.load_module(ptx).map_err(|e| format!("{e}"))?;
    let kernels = GpuKernels {
        sum: module
            .load_function("block_reduce_sum")
            .map_err(|e| format!("{e}"))?,
        min: module
            .load_function("block_reduce_min")
            .map_err(|e| format!("{e}"))?,
        max: module
            .load_function("block_reduce_max")
            .map_err(|e| format!("{e}"))?,
    };
    Ok(GpuState { ctx, kernels })
}

/// Best-effort device memory check before allocating the host→device buffer.
pub(crate) fn vram_check(device_index: usize, logical_bytes: usize) -> Result<(), &'static str> {
    with_gpu_state(device_index, |state| {
        let (_total, free) = state.ctx.mem_get_info().map_err(|e| format!("{e}"))?;
        let limit = ((free as f64) * GPU_VRAM_BUFFER_FRACTION) as u64;
        if u64::try_from(logical_bytes).is_ok_and(|b| b > limit) {
            return Err("gpu_vram_exceeded".into());
        }
        Ok(())
    })
    .map_err(|e| {
        if e == "gpu_vram_exceeded" {
            "gpu_vram_exceeded"
        } else {
            "gpu_runtime_error"
        }
    })
}

pub(crate) fn reduce_f32_scalar(
    host: &[f32],
    kind: ReductionKind,
    device_index: usize,
) -> Result<ScalarReductionResult, String> {
    let n = host.len();
    if n == 0 {
        return Ok(ScalarReductionResult {
            element_count: 0,
            ..ScalarReductionResult::default_fields(0)
        });
    }
    if matches!(kind, ReductionKind::Var | ReductionKind::Std) {
        let var = super::host_f32_population_variance(host);
        return Ok(ScalarReductionResult {
            element_count: n,
            var_scalar: Some(var),
            std_scalar: if matches!(kind, ReductionKind::Std) {
                Some(var.sqrt())
            } else {
                None
            },
            ..ScalarReductionResult::default_fields(n)
        });
    }
    with_gpu_state(device_index, |state| {
        let stream = state.ctx.default_stream();
        let device_buf = stream.clone_htod(host).map_err(|e| format!("{e}"))?;

        match kind {
            ReductionKind::Count => Ok(ScalarReductionResult {
                element_count: n,
                ..ScalarReductionResult::default_fields(n)
            }),
            ReductionKind::Sum | ReductionKind::Mean => {
                let sum = block_tree_reduce_f32(&stream, &state.kernels.sum, &device_buf, n)?;
                let sum_f64 = f64::from(sum);
                Ok(ScalarReductionResult {
                    element_count: n,
                    sum_scalar: Some(sum_f64),
                    mean_scalar: if matches!(kind, ReductionKind::Mean) {
                        Some(sum_f64 / n as f64)
                    } else {
                        None
                    },
                    ..ScalarReductionResult::default_fields(n)
                })
            }
            ReductionKind::Min => {
                let v = block_tree_reduce_f32(&stream, &state.kernels.min, &device_buf, n)?;
                Ok(ScalarReductionResult {
                    element_count: n,
                    min_scalar: Some(f64::from(v)),
                    ..ScalarReductionResult::default_fields(n)
                })
            }
            ReductionKind::Max => {
                let v = block_tree_reduce_f32(&stream, &state.kernels.max, &device_buf, n)?;
                Ok(ScalarReductionResult {
                    element_count: n,
                    max_scalar: Some(f64::from(v)),
                    ..ScalarReductionResult::default_fields(n)
                })
            }
            _ => Err("gpu_unsupported_op".into()),
        }
    })
}

fn block_tree_reduce_f32(
    stream: &Arc<CudaStream>,
    kernel: &CudaFunction,
    input: &CudaSlice<f32>,
    n: usize,
) -> Result<f32, String> {
    let mut current = input.clone();
    let mut len = n;
    loop {
        if len == 0 {
            return Ok(0.0);
        }
        if len == 1 {
            let host = stream.clone_dtoh(&current).map_err(|e| format!("{e}"))?;
            return Ok(host[0]);
        }
        let grid = grid_dim(len);
        let partials = stream.alloc_zeros(grid).map_err(|e| format!("{e}"))?;
        launch_block_reduce(stream, kernel, &current, &partials, len)?;
        current = partials;
        len = grid;
    }
}

fn grid_dim(n: usize) -> usize {
    let threads = BLOCK_THREADS as usize;
    n.div_ceil(threads).max(1)
}

fn launch_block_reduce(
    stream: &Arc<CudaStream>,
    kernel: &CudaFunction,
    input: &CudaSlice<f32>,
    output: &CudaSlice<f32>,
    n: usize,
) -> Result<(), String> {
    let grid = grid_dim(n);
    let cfg = LaunchConfig {
        grid_dim: (grid as u32, 1, 1),
        block_dim: (BLOCK_THREADS, 1, 1),
        shared_mem_bytes: 0,
    };
    let n_i32 = i32::try_from(n).map_err(|_| "gpu_length_overflow".to_string())?;
    unsafe {
        stream
            .launch_builder(kernel)
            .arg(input)
            .arg(output)
            .arg(&n_i32)
            .launch(cfg)
            .map_err(|e| format!("{e}"))?;
    }
    stream.synchronize().map_err(|e| format!("{e}"))
}
