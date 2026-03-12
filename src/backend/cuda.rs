#[derive(Clone, Debug)]
pub struct CudaBuffer {
    pub length: usize,
}

impl CudaBuffer {
    pub fn new(data: &[f32]) -> Self {
        Self { length: data.len() }
    }
    pub fn zeros(length: usize) -> Self {
        Self { length }
    }
    pub fn to_vec(&self) -> Vec<f32> {
        unimplemented!("CUDA backend is a placeholder")
    }
}

pub fn add(_a: &CudaBuffer, _b: &CudaBuffer) -> CudaBuffer { unimplemented!("CUDA backend is a placeholder") }
pub fn mul(_a: &CudaBuffer, _b: &CudaBuffer) -> CudaBuffer { unimplemented!("CUDA backend is a placeholder") }
pub fn matmul(_a: &CudaBuffer, _b: &CudaBuffer, _m: usize, _k: usize, _n: usize) -> CudaBuffer { unimplemented!("CUDA backend is a placeholder") }
pub fn relu(_a: &CudaBuffer) -> CudaBuffer { unimplemented!("CUDA backend is a placeholder") }
pub fn sum(_a: &CudaBuffer) -> CudaBuffer { unimplemented!("CUDA backend is a placeholder") }
pub fn pow(_a: &CudaBuffer, _p: f32) -> CudaBuffer { unimplemented!("CUDA backend is a placeholder") }
pub fn log_fw(_a: &CudaBuffer) -> CudaBuffer { unimplemented!("CUDA backend is a placeholder") }
pub fn softmax_fw(_a: &CudaBuffer, _rows: usize, _cols: usize) -> CudaBuffer { unimplemented!("CUDA backend is a placeholder") }
pub fn log_softmax_fw(_a: &CudaBuffer, _rows: usize, _cols: usize) -> CudaBuffer { unimplemented!("CUDA backend is a placeholder") }
pub fn add_broadcast(_a: &CudaBuffer, _b: &CudaBuffer, _cols: usize) -> CudaBuffer { unimplemented!("CUDA backend is a placeholder") }

pub fn add_assign(_target: &mut CudaBuffer, _source: &CudaBuffer) { unimplemented!("CUDA backend is a placeholder") }
pub fn relu_bw(_a: &CudaBuffer, _grad_out: &CudaBuffer) -> CudaBuffer { unimplemented!("CUDA backend is a placeholder") }
pub fn sum_bw(_grad_out: &CudaBuffer, _len: usize) -> CudaBuffer { unimplemented!("CUDA backend is a placeholder") }
pub fn pow_bw(_a: &CudaBuffer, _grad_out: &CudaBuffer, _p: f32) -> CudaBuffer { unimplemented!("CUDA backend is a placeholder") }
pub fn log_bw(_a: &CudaBuffer, _grad_out: &CudaBuffer) -> CudaBuffer { unimplemented!("CUDA backend is a placeholder") }
pub fn transpose(_a: &CudaBuffer, _rows: usize, _cols: usize) -> CudaBuffer { unimplemented!("CUDA backend is a placeholder") }
pub fn softmax_bw(_sm_out: &CudaBuffer, _grad_out: &CudaBuffer, _rows: usize, _cols: usize) -> CudaBuffer { unimplemented!("CUDA backend is a placeholder") }
pub fn log_softmax_bw(_log_sm_out: &CudaBuffer, _grad_out: &CudaBuffer, _rows: usize, _cols: usize) -> CudaBuffer { unimplemented!("CUDA backend is a placeholder") }
pub fn add_broadcast_bw_b(_grad_out: &CudaBuffer, _rows: usize, _cols: usize) -> CudaBuffer { unimplemented!("CUDA backend is a placeholder") }
pub fn zero_grad(_grad: &mut CudaBuffer) { unimplemented!("CUDA backend is a placeholder") }
pub fn sgd_step(_param: &mut CudaBuffer, _grad: &CudaBuffer, _lr: f32) { unimplemented!("CUDA backend is a placeholder") }
