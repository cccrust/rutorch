use lazy_static::lazy_static;
use metal::*;

const SHADER_CODE: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void add(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]], device float* out [[buffer(2)]], uint id [[thread_position_in_grid]]) {
    out[id] = a[id] + b[id];
}
kernel void mul(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]], device float* out [[buffer(2)]], uint id [[thread_position_in_grid]]) {
    out[id] = a[id] * b[id];
}
kernel void add_assign(device float* target [[buffer(0)]], device const float* source [[buffer(1)]], uint id [[thread_position_in_grid]]) {
    target[id] += source[id];
}
kernel void relu(device const float* a [[buffer(0)]], device float* out [[buffer(1)]], uint id [[thread_position_in_grid]]) {
    out[id] = max(0.0, a[id]);
}
kernel void relu_backward(device const float* a [[buffer(0)]], device const float* grad_in [[buffer(1)]], device float* grad_out [[buffer(2)]], uint id [[thread_position_in_grid]]) {
    grad_out[id] = a[id] > 0.0 ? grad_in[id] : 0.0;
}
kernel void matmul(device const float* A [[buffer(0)]], device const float* B [[buffer(1)]], device float* C [[buffer(2)]], constant uint& M [[buffer(3)]], constant uint& K [[buffer(4)]], constant uint& N [[buffer(5)]], uint2 id [[thread_position_in_grid]]) {
    if (id.x >= N || id.y >= M) return;
    float sum = 0.0;
    for (uint i = 0; i < K; ++i) { sum += A[id.y * K + i] * B[i * N + id.x]; }
    C[id.y * N + id.x] = sum;
}
kernel void transpose(device const float* in [[buffer(0)]], device float* out [[buffer(1)]], constant uint& rows [[buffer(2)]], constant uint& cols [[buffer(3)]], uint2 id [[thread_position_in_grid]]) {
    if (id.x >= cols || id.y >= rows) return;
    out[id.x * rows + id.y] = in[id.y * cols + id.x];
}
kernel void sum_all(device const float* in [[buffer(0)]], device float* out [[buffer(1)]], constant uint& length [[buffer(2)]], uint id [[thread_position_in_grid]]) {
    if (id > 0) return;
    float s = 0.0;
    for (uint i = 0; i < length; ++i) { s += in[i]; }
    out[0] = s;
}
kernel void sum_backward(device float* target_grad [[buffer(0)]], device const float* out_grad [[buffer(1)]], constant uint& length [[buffer(2)]], uint id [[thread_position_in_grid]]) {
    if (id >= length) return;
    target_grad[id] += out_grad[0];
}

// === 新增：Power (次方) ===
kernel void power(device const float* a [[buffer(0)]], device float* out [[buffer(1)]], constant float& p [[buffer(2)]], uint id [[thread_position_in_grid]]) {
    out[id] = pow(a[id], p);
}
kernel void power_backward(device const float* a [[buffer(0)]], device const float* grad_out [[buffer(1)]], device float* target_grad [[buffer(2)]], constant float& p [[buffer(3)]], uint id [[thread_position_in_grid]]) {
    target_grad[id] += p * pow(a[id], p - 1.0) * grad_out[id];
}

// === 新增：Log (自然對數) ===
kernel void log_fw(device const float* a [[buffer(0)]], device float* out [[buffer(1)]], uint id [[thread_position_in_grid]]) {
    out[id] = log(a[id]);
}
kernel void log_bw(device const float* a [[buffer(0)]], device const float* grad_out [[buffer(1)]], device float* target_grad [[buffer(2)]], uint id [[thread_position_in_grid]]) {
    target_grad[id] += (1.0 / a[id]) * grad_out[id];
}

// === 新增：Softmax (2D, 針對列做正規化) ===
// 這裡 id 代表的是第幾列 (row)
kernel void softmax_fw(device const float* in [[buffer(0)]], device float* out [[buffer(1)]], constant uint& cols [[buffer(2)]], uint id [[thread_position_in_grid]]) {
    uint offset = id * cols;
    // 尋找最大值以確保數值穩定性
    float max_val = in[offset];
    for (uint i = 1; i < cols; ++i) { max_val = max(max_val, in[offset + i]); }
    
    float sum = 0.0;
    for (uint i = 0; i < cols; ++i) {
        float e = exp(in[offset + i] - max_val);
        out[offset + i] = e;
        sum += e;
    }
    for (uint i = 0; i < cols; ++i) { out[offset + i] /= sum; }
}
kernel void softmax_bw(device const float* sm_out [[buffer(0)]], device const float* grad_out [[buffer(1)]], device float* target_grad [[buffer(2)]], constant uint& cols [[buffer(3)]], uint id [[thread_position_in_grid]]) {
    uint offset = id * cols;
    float s = 0.0;
    for (uint i = 0; i < cols; ++i) { s += grad_out[offset + i] * sm_out[offset + i]; }
    for (uint i = 0; i < cols; ++i) { target_grad[offset + i] += (grad_out[offset + i] - s) * sm_out[offset + i]; }
}

// === 新增：Log-Softmax (2D) ===
kernel void log_softmax_fw(device const float* in [[buffer(0)]], device float* out [[buffer(1)]], constant uint& cols [[buffer(2)]], uint id [[thread_position_in_grid]]) {
    uint offset = id * cols;
    float max_val = in[offset];
    for (uint i = 1; i < cols; ++i) { max_val = max(max_val, in[offset + i]); }

    float sum = 0.0;
    for (uint i = 0; i < cols; ++i) { sum += exp(in[offset + i] - max_val); }
    float log_sum = log(sum);

    for (uint i = 0; i < cols; ++i) {
        out[offset + i] = in[offset + i] - max_val - log_sum;
    }
}

kernel void log_softmax_bw(device const float* log_sm_out [[buffer(0)]], device const float* grad_out [[buffer(1)]], device float* target_grad [[buffer(2)]], constant uint& cols [[buffer(3)]], uint id [[thread_position_in_grid]]) {
    uint offset = id * cols;
    float sum_grad = 0.0;
    for (uint i = 0; i < cols; ++i) { sum_grad += grad_out[offset + i]; }
    for (uint i = 0; i < cols; ++i) {
        float sm = exp(log_sm_out[offset + i]);
        target_grad[offset + i] += grad_out[offset + i] - sm * sum_grad;
    }
}

// === 新增：優化器與梯度管理 ===

// 清空梯度 (設為 0)
kernel void zero_grad(device float* grad [[buffer(0)]], uint id [[thread_position_in_grid]]) {
    grad[id] = 0.0;
}

// SGD 權重更新： param = param - lr * grad
kernel void sgd_step(device float* param [[buffer(0)]], device const float* grad [[buffer(1)]], constant float& lr [[buffer(2)]], uint id [[thread_position_in_grid]]) {
    param[id] -= lr * grad[id];
}

// === 新增：廣播加法 (Bias 加法) ===
// 將一維的 b (長度為 cols) 廣播加到二維的 a (大小為 rows x cols) 上
kernel void add_broadcast(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]], device float* out [[buffer(2)]], constant uint& cols [[buffer(3)]], uint id [[thread_position_in_grid]]) {
    // id % cols 可以精準地讓每一列對應到正確的 b 元素
    out[id] = a[id] + b[id % cols];
}

// 廣播加法的反向傳播 (專門計算 Bias 的梯度)
// Bias 的梯度等於「所有批次(Batch)的梯度在垂直方向的加總」
kernel void add_broadcast_bw_b(device const float* grad_out [[buffer(0)]], device float* grad_b [[buffer(1)]], constant uint& rows [[buffer(2)]], constant uint& cols [[buffer(3)]], uint id [[thread_position_in_grid]]) {
    if (id >= cols) return;
    float sum = 0.0;
    // 將每個 column (對應一個 bias) 的梯度往下加總
    for(uint r = 0; r < rows; r++) {
        sum += grad_out[r * cols + id];
    }
    grad_b[id] += sum;
}
"#;

struct MetalContext { device: Device, queue: CommandQueue, library: Library }
impl MetalContext {
    fn new() -> Self {
        let device = Device::system_default().expect("找不到 Metal 設備！");
        let queue = device.new_command_queue();
        let options = CompileOptions::new();
        let library = device.new_library_with_source(SHADER_CODE, &options).unwrap();
        Self { device, queue, library }
    }
}
lazy_static! { static ref METAL_CTX: MetalContext = MetalContext::new(); }

#[derive(Clone)]
pub struct GpuBuffer { pub buffer: Buffer, pub length: usize }
impl GpuBuffer {
    pub fn new(data: &[f32]) -> Self {
        let size = (data.len() * std::mem::size_of::<f32>()) as u64;
        let buffer = METAL_CTX.device.new_buffer_with_data(data.as_ptr() as *const _, size, MTLResourceOptions::StorageModeShared);
        Self { buffer, length: data.len() }
    }
    pub fn zeros(length: usize) -> Self { Self::new(&vec![0.0f32; length]) }
    pub fn to_vec(&self) -> Vec<f32> {
        let mut vec = vec![0.0f32; self.length];
        let ptr = self.buffer.contents() as *const f32;
        unsafe { std::ptr::copy_nonoverlapping(ptr, vec.as_mut_ptr(), self.length); }
        vec
    }
}

pub fn dispatch_1d(name: &str, buffers: &[&Buffer], length: usize) {
    let ctx = &METAL_CTX;
    let func = ctx.library.get_function(name, None).unwrap();
    let pipeline = ctx.device.new_compute_pipeline_state_with_function(&func).unwrap();
    let cmd_buffer = ctx.queue.new_command_buffer();
    let encoder = cmd_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    for (i, buf) in buffers.iter().enumerate() { encoder.set_buffer(i as u64, Some(*buf), 0); }
    let grid_size = MTLSize::new(length as u64, 1, 1);
    let tg_size = MTLSize::new(std::cmp::min(length as u64, 32).max(1), 1, 1);
    encoder.dispatch_threads(grid_size, tg_size);
    encoder.end_encoding();
    cmd_buffer.commit(); cmd_buffer.wait_until_completed();
}

pub fn dispatch_1d_with_u32(name: &str, buffers: &[&Buffer], length: usize, val: u32) {
    let ctx = &METAL_CTX;
    let func = ctx.library.get_function(name, None).unwrap();
    let pipeline = ctx.device.new_compute_pipeline_state_with_function(&func).unwrap();
    let cmd_buffer = ctx.queue.new_command_buffer();
    let encoder = cmd_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    for (i, buf) in buffers.iter().enumerate() { encoder.set_buffer(i as u64, Some(*buf), 0); }
    encoder.set_bytes(buffers.len() as u64, 4, &val as *const _ as *const _); // 綁定最後一個參數
    let grid_size = MTLSize::new(length as u64, 1, 1);
    let tg_size = MTLSize::new(std::cmp::min(length as u64, 32).max(1), 1, 1);
    encoder.dispatch_threads(grid_size, tg_size);
    encoder.end_encoding();
    cmd_buffer.commit(); cmd_buffer.wait_until_completed();
}

pub fn dispatch_1d_with_2_u32(name: &str, buffers: &[&Buffer], length: usize, val1: u32, val2: u32) {
    let ctx = &METAL_CTX;
    let func = ctx.library.get_function(name, None).unwrap();
    let pipeline = ctx.device.new_compute_pipeline_state_with_function(&func).unwrap();
    let cmd_buffer = ctx.queue.new_command_buffer();
    let encoder = cmd_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    
    for (i, buf) in buffers.iter().enumerate() { encoder.set_buffer(i as u64, Some(*buf), 0); }
    encoder.set_bytes(buffers.len() as u64, 4, &val1 as *const _ as *const _);
    encoder.set_bytes((buffers.len() + 1) as u64, 4, &val2 as *const _ as *const _);
    
    let grid_size = MTLSize::new(length as u64, 1, 1);
    let tg_size = MTLSize::new(std::cmp::min(length as u64, 32).max(1), 1, 1);
    encoder.dispatch_threads(grid_size, tg_size);
    encoder.end_encoding();
    cmd_buffer.commit(); cmd_buffer.wait_until_completed();
}

pub fn dispatch_1d_with_f32(name: &str, buffers: &[&Buffer], length: usize, val: f32) {
    let ctx = &METAL_CTX;
    let func = ctx.library.get_function(name, None).unwrap();
    let pipeline = ctx.device.new_compute_pipeline_state_with_function(&func).unwrap();
    let cmd_buffer = ctx.queue.new_command_buffer();
    let encoder = cmd_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    for (i, buf) in buffers.iter().enumerate() { encoder.set_buffer(i as u64, Some(*buf), 0); }
    encoder.set_bytes(buffers.len() as u64, 4, &val as *const _ as *const _);
    let grid_size = MTLSize::new(length as u64, 1, 1);
    let tg_size = MTLSize::new(std::cmp::min(length as u64, 32).max(1), 1, 1);
    encoder.dispatch_threads(grid_size, tg_size);
    encoder.end_encoding();
    cmd_buffer.commit(); cmd_buffer.wait_until_completed();
}

pub fn dispatch_matmul(name: &str, a: &Buffer, b: &Buffer, c: &Buffer, m: u32, k: u32, n: u32) {
    let ctx = &METAL_CTX;
    let func = ctx.library.get_function(name, None).unwrap();
    let pipeline = ctx.device.new_compute_pipeline_state_with_function(&func).unwrap();
    let cmd_buffer = ctx.queue.new_command_buffer();
    let encoder = cmd_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(a), 0); encoder.set_buffer(1, Some(b), 0); encoder.set_buffer(2, Some(c), 0);
    encoder.set_bytes(3, 4, &m as *const _ as *const _); encoder.set_bytes(4, 4, &k as *const _ as *const _); encoder.set_bytes(5, 4, &n as *const _ as *const _);
    let grid_size = MTLSize::new(n as u64, m as u64, 1);
    let tg_size = MTLSize::new(8, 8, 1);
    encoder.dispatch_threads(grid_size, tg_size);
    encoder.end_encoding();
    cmd_buffer.commit(); cmd_buffer.wait_until_completed();
}

pub fn dispatch_transpose(in_buf: &Buffer, out_buf: &Buffer, rows: u32, cols: u32) {
    let ctx = &METAL_CTX;
    let func = ctx.library.get_function("transpose", None).unwrap();
    let pipeline = ctx.device.new_compute_pipeline_state_with_function(&func).unwrap();
    let cmd_buffer = ctx.queue.new_command_buffer();
    let encoder = cmd_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(in_buf), 0); encoder.set_buffer(1, Some(out_buf), 0);
    encoder.set_bytes(2, 4, &rows as *const _ as *const _); encoder.set_bytes(3, 4, &cols as *const _ as *const _);
    let grid_size = MTLSize::new(cols as u64, rows as u64, 1);
    let tg_size = MTLSize::new(8, 8, 1);
    encoder.dispatch_threads(grid_size, tg_size);
    encoder.end_encoding();
    cmd_buffer.commit(); cmd_buffer.wait_until_completed();
}

pub fn add(a: &GpuBuffer, b: &GpuBuffer) -> GpuBuffer {
    let out = GpuBuffer::zeros(a.length);
    dispatch_1d("add", &[&a.buffer, &b.buffer, &out.buffer], a.length);
    out
}

pub fn mul(a: &GpuBuffer, b: &GpuBuffer) -> GpuBuffer {
    let out = GpuBuffer::zeros(a.length);
    dispatch_1d("mul", &[&a.buffer, &b.buffer, &out.buffer], a.length);
    out
}

pub fn matmul(a: &GpuBuffer, b: &GpuBuffer, m: usize, k: usize, n: usize) -> GpuBuffer {
    let out = GpuBuffer::zeros(m * n);
    dispatch_matmul("matmul", &a.buffer, &b.buffer, &out.buffer, m as u32, k as u32, n as u32);
    out
}

pub fn relu(a: &GpuBuffer) -> GpuBuffer {
    let out = GpuBuffer::zeros(a.length);
    dispatch_1d("relu", &[&a.buffer, &out.buffer], a.length);
    out
}

pub fn sum(a: &GpuBuffer) -> GpuBuffer {
    let out = GpuBuffer::zeros(1);
    dispatch_1d_with_u32("sum_all", &[&a.buffer, &out.buffer], a.length, a.length as u32);
    out
}

pub fn pow(a: &GpuBuffer, p: f32) -> GpuBuffer {
    let out = GpuBuffer::zeros(a.length);
    dispatch_1d_with_f32("power", &[&a.buffer, &out.buffer], a.length, p);
    out
}

pub fn log_fw(a: &GpuBuffer) -> GpuBuffer {
    let out = GpuBuffer::zeros(a.length);
    dispatch_1d("log_fw", &[&a.buffer, &out.buffer], a.length);
    out
}

pub fn softmax_fw(a: &GpuBuffer, rows: usize, cols: usize) -> GpuBuffer {
    let out = GpuBuffer::zeros(rows * cols);
    dispatch_1d_with_u32("softmax_fw", &[&a.buffer, &out.buffer], rows, cols as u32);
    out
}

pub fn log_softmax_fw(a: &GpuBuffer, rows: usize, cols: usize) -> GpuBuffer {
    let out = GpuBuffer::zeros(rows * cols);
    dispatch_1d_with_u32("log_softmax_fw", &[&a.buffer, &out.buffer], rows, cols as u32);
    out
}

pub fn add_broadcast(a: &GpuBuffer, b: &GpuBuffer, cols: usize) -> GpuBuffer {
    let out = GpuBuffer::zeros(a.length);
    dispatch_1d_with_u32("add_broadcast", &[&a.buffer, &b.buffer, &out.buffer], a.length, cols as u32);
    out
}

pub fn add_assign(target: &mut GpuBuffer, source: &GpuBuffer) {
    dispatch_1d("add_assign", &[&target.buffer, &source.buffer], target.length);
}

pub fn relu_bw(a: &GpuBuffer, grad_out: &GpuBuffer) -> GpuBuffer {
    let grad_in = GpuBuffer::zeros(a.length);
    dispatch_1d("relu_backward", &[&a.buffer, &grad_out.buffer, &grad_in.buffer], a.length);
    grad_in
}

pub fn sum_bw(grad_out: &GpuBuffer, len: usize) -> GpuBuffer {
    let grad_in = GpuBuffer::zeros(len);
    dispatch_1d_with_u32("sum_backward", &[&grad_in.buffer, &grad_out.buffer], len, len as u32);
    grad_in
}

pub fn pow_bw(a: &GpuBuffer, grad_out: &GpuBuffer, p: f32) -> GpuBuffer {
    let grad_in = GpuBuffer::zeros(a.length);
    dispatch_1d_with_f32("power_backward", &[&a.buffer, &grad_out.buffer, &grad_in.buffer], a.length, p);
    grad_in
}

pub fn log_bw(a: &GpuBuffer, grad_out: &GpuBuffer) -> GpuBuffer {
    let grad_in = GpuBuffer::zeros(a.length);
    dispatch_1d("log_bw", &[&a.buffer, &grad_out.buffer, &grad_in.buffer], a.length);
    grad_in
}

pub fn transpose(a: &GpuBuffer, rows: usize, cols: usize) -> GpuBuffer {
    let out = GpuBuffer::zeros(rows * cols);
    dispatch_transpose(&a.buffer, &out.buffer, rows as u32, cols as u32);
    out
}

pub fn softmax_bw(sm_out: &GpuBuffer, grad_out: &GpuBuffer, target_grad: &mut GpuBuffer, rows: usize, cols: usize) {
    dispatch_1d_with_u32("softmax_bw", &[&sm_out.buffer, &grad_out.buffer, &target_grad.buffer], rows, cols as u32);
}

pub fn log_softmax_bw(log_sm_out: &GpuBuffer, grad_out: &GpuBuffer, target_grad: &mut GpuBuffer, rows: usize, cols: usize) {
    dispatch_1d_with_u32("log_softmax_bw", &[&log_sm_out.buffer, &grad_out.buffer, &target_grad.buffer], rows, cols as u32);
}

pub fn add_broadcast_bw_b(grad_out: &GpuBuffer, grad_b: &mut GpuBuffer, rows: usize, cols: usize) {
    dispatch_1d_with_2_u32("add_broadcast_bw_b", &[&grad_out.buffer, &grad_b.buffer], cols, rows as u32, cols as u32);
}

pub fn zero_grad(grad: &mut GpuBuffer) {
    dispatch_1d("zero_grad", &[&grad.buffer], grad.length);
}

pub fn sgd_step(param: &mut GpuBuffer, grad: &GpuBuffer, lr: f32) {
    dispatch_1d_with_f32("sgd_step", &[&param.buffer, &grad.buffer], param.length, lr);
}
