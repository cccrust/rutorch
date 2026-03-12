pub mod cpu;
pub mod cuda;
#[cfg(target_os = "macos")]
pub mod metal;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Device {
    Cpu,
    MacMetal,
    Cuda,
}

impl Default for Device {
    fn default() -> Self {
        cfg_if::cfg_if! {
            if #[cfg(target_os = "macos")] {
                Device::MacMetal
            } else {
                Device::Cpu
            }
        }
    }
}

pub enum Storage {
    Cpu(cpu::CpuBuffer),
    #[cfg(target_os = "macos")]
    Metal(metal::GpuBuffer),
    Cuda(cuda::CudaBuffer),
}

impl Clone for Storage {
    fn clone(&self) -> Self {
        match self {
            Storage::Cpu(c) => Storage::Cpu(c.clone()),
            #[cfg(target_os = "macos")]
            Storage::Metal(m) => Storage::Metal(m.clone()),
            Storage::Cuda(c) => Storage::Cuda(c.clone()),
        }
    }
}

impl Storage {
    pub fn length(&self) -> usize {
        match self {
            Storage::Cpu(c) => c.length,
            #[cfg(target_os = "macos")]
            Storage::Metal(m) => m.length,
            Storage::Cuda(c) => c.length,
        }
    }
    
    pub fn to_vec(&self) -> Vec<f32> {
        match self {
            Storage::Cpu(c) => c.to_vec(),
            #[cfg(target_os = "macos")]
            Storage::Metal(m) => m.to_vec(),
            Storage::Cuda(c) => c.to_vec(),
        }
    }
    
    pub fn device(&self) -> Device {
        match self {
            Storage::Cpu(_) => Device::Cpu,
            #[cfg(target_os = "macos")]
            Storage::Metal(_) => Device::MacMetal,
            Storage::Cuda(_) => Device::Cuda,
        }
    }

    pub fn zeros(length: usize, device: Device) -> Self {
        match device {
            Device::Cpu => Storage::Cpu(cpu::CpuBuffer::zeros(length)),
            #[cfg(target_os = "macos")]
            Device::MacMetal => Storage::Metal(metal::GpuBuffer::zeros(length)),
            Device::Cuda => Storage::Cuda(cuda::CudaBuffer::zeros(length)),
        }
    }

    pub fn new(data: &[f32], device: Device) -> Self {
        match device {
            Device::Cpu => Storage::Cpu(cpu::CpuBuffer::new(data)),
            #[cfg(target_os = "macos")]
            Device::MacMetal => Storage::Metal(metal::GpuBuffer::new(data)),
            Device::Cuda => Storage::Cuda(cuda::CudaBuffer::new(data)),
        }
    }

    // Backend dispatch methods
    pub fn add(&self, other: &Self) -> Self {
        match (self, other) {
            (Storage::Cpu(a), Storage::Cpu(b)) => Storage::Cpu(cpu::add(a, b)),
            #[cfg(target_os = "macos")]
            (Storage::Metal(a), Storage::Metal(b)) => Storage::Metal(metal::add(a, b)),
            (Storage::Cuda(a), Storage::Cuda(b)) => Storage::Cuda(cuda::add(a, b)),
            _ => panic!("Backend mismatch or unsupported backend operation"),
        }
    }

    pub fn mul(&self, other: &Self) -> Self {
        match (self, other) {
            (Storage::Cpu(a), Storage::Cpu(b)) => Storage::Cpu(cpu::mul(a, b)),
            #[cfg(target_os = "macos")]
            (Storage::Metal(a), Storage::Metal(b)) => Storage::Metal(metal::mul(a, b)),
            (Storage::Cuda(a), Storage::Cuda(b)) => Storage::Cuda(cuda::mul(a, b)),
            _ => panic!("Backend mismatch"),
        }
    }

    pub fn matmul(&self, other: &Self, m: usize, k: usize, n: usize) -> Self {
        match (self, other) {
            (Storage::Cpu(a), Storage::Cpu(b)) => Storage::Cpu(cpu::matmul(a, b, m, k, n)),
            #[cfg(target_os = "macos")]
            (Storage::Metal(a), Storage::Metal(b)) => Storage::Metal(metal::matmul(a, b, m, k, n)),
            (Storage::Cuda(a), Storage::Cuda(b)) => Storage::Cuda(cuda::matmul(a, b, m, k, n)),
            _ => panic!("Backend mismatch"),
        }
    }

    pub fn relu(&self) -> Self {
        match self {
            Storage::Cpu(a) => Storage::Cpu(cpu::relu(a)),
            #[cfg(target_os = "macos")]
            Storage::Metal(a) => Storage::Metal(metal::relu(a)),
            Storage::Cuda(a) => Storage::Cuda(cuda::relu(a)),
        }
    }

    pub fn sum(&self) -> Self {
        match self {
            Storage::Cpu(a) => Storage::Cpu(cpu::sum(a)),
            #[cfg(target_os = "macos")]
            Storage::Metal(a) => Storage::Metal(metal::sum(a)),
            Storage::Cuda(a) => Storage::Cuda(cuda::sum(a)),
        }
    }

    pub fn pow(&self, p: f32) -> Self {
        match self {
            Storage::Cpu(a) => Storage::Cpu(cpu::pow(a, p)),
            #[cfg(target_os = "macos")]
            Storage::Metal(a) => Storage::Metal(metal::pow(a, p)),
            Storage::Cuda(a) => Storage::Cuda(cuda::pow(a, p)),
        }
    }

    pub fn log_fw(&self) -> Self {
        match self {
            Storage::Cpu(a) => Storage::Cpu(cpu::log_fw(a)),
            #[cfg(target_os = "macos")]
            Storage::Metal(a) => Storage::Metal(metal::log_fw(a)),
            Storage::Cuda(a) => Storage::Cuda(cuda::log_fw(a)),
        }
    }

    pub fn softmax_fw(&self, rows: usize, cols: usize) -> Self {
        match self {
            Storage::Cpu(a) => Storage::Cpu(cpu::softmax_fw(a, rows, cols)),
            #[cfg(target_os = "macos")]
            Storage::Metal(a) => Storage::Metal(metal::softmax_fw(a, rows, cols)),
            Storage::Cuda(a) => Storage::Cuda(cuda::softmax_fw(a, rows, cols)),
        }
    }

    pub fn add_broadcast(&self, b: &Self, cols: usize) -> Self {
        match (self, b) {
            (Storage::Cpu(a), Storage::Cpu(b_)) => Storage::Cpu(cpu::add_broadcast(a, b_, cols)),
            #[cfg(target_os = "macos")]
            (Storage::Metal(a), Storage::Metal(b_)) => Storage::Metal(metal::add_broadcast(a, b_, cols)),
            (Storage::Cuda(a), Storage::Cuda(b_)) => Storage::Cuda(cuda::add_broadcast(a, b_, cols)),
            _ => panic!("Backend mismatch"),
        }
    }

    pub fn add_assign(&mut self, source: &Self) {
        match (self, source) {
            (Storage::Cpu(a), Storage::Cpu(b)) => cpu::add_assign(a, b),
            #[cfg(target_os = "macos")]
            (Storage::Metal(a), Storage::Metal(b)) => metal::add_assign(a, b),
            (Storage::Cuda(a), Storage::Cuda(b)) => cuda::add_assign(a, b),
            _ => panic!("Backend mismatch"),
        }
    }

    pub fn relu_bw(&self, grad_out: &Self) -> Self {
        match (self, grad_out) {
            (Storage::Cpu(a), Storage::Cpu(g)) => Storage::Cpu(cpu::relu_bw(a, g)),
            #[cfg(target_os = "macos")]
            (Storage::Metal(a), Storage::Metal(g)) => Storage::Metal(metal::relu_bw(a, g)),
            (Storage::Cuda(a), Storage::Cuda(g)) => Storage::Cuda(cuda::relu_bw(a, g)),
            _ => panic!("Backend mismatch"),
        }
    }

    pub fn sum_bw(&self, grad_out: &Self) -> Self {
        match (self, grad_out) {
            (Storage::Cpu(a), Storage::Cpu(g)) => Storage::Cpu(cpu::sum_bw(g, a.length)),
            #[cfg(target_os = "macos")]
            (Storage::Metal(a), Storage::Metal(g)) => Storage::Metal(metal::sum_bw(g, a.length)),
            (Storage::Cuda(a), Storage::Cuda(g)) => Storage::Cuda(cuda::sum_bw(g, a.length)),
            _ => panic!("Backend mismatch"),
        }
    }

    pub fn pow_bw(&self, grad_out: &Self, p: f32) -> Self {
        match (self, grad_out) {
            (Storage::Cpu(a), Storage::Cpu(g)) => Storage::Cpu(cpu::pow_bw(a, g, p)),
            #[cfg(target_os = "macos")]
            (Storage::Metal(a), Storage::Metal(g)) => Storage::Metal(metal::pow_bw(a, g, p)),
            (Storage::Cuda(a), Storage::Cuda(g)) => Storage::Cuda(cuda::pow_bw(a, g, p)),
            _ => panic!("Backend mismatch"),
        }
    }

    pub fn log_bw(&self, grad_out: &Self) -> Self {
        match (self, grad_out) {
            (Storage::Cpu(a), Storage::Cpu(g)) => Storage::Cpu(cpu::log_bw(a, g)),
            #[cfg(target_os = "macos")]
            (Storage::Metal(a), Storage::Metal(g)) => Storage::Metal(metal::log_bw(a, g)),
            (Storage::Cuda(a), Storage::Cuda(g)) => Storage::Cuda(cuda::log_bw(a, g)),
            _ => panic!("Backend mismatch"),
        }
    }

    pub fn transpose(&self, rows: usize, cols: usize) -> Self {
        match self {
            Storage::Cpu(a) => Storage::Cpu(cpu::transpose(a, rows, cols)),
            #[cfg(target_os = "macos")]
            Storage::Metal(a) => Storage::Metal(metal::transpose(a, rows, cols)),
            Storage::Cuda(a) => Storage::Cuda(cuda::transpose(a, rows, cols)),
        }
    }

    pub fn softmax_bw(&self, grad_out: &Self, target_grad: &mut Self, rows: usize, cols: usize) {
        match (self, grad_out, target_grad) {
            (Storage::Cpu(sm), Storage::Cpu(go), Storage::Cpu(tg)) => {
                let bw = cpu::softmax_bw(sm, go, rows, cols);
                cpu::add_assign(tg, &bw);
            },
            #[cfg(target_os = "macos")]
            (Storage::Metal(sm), Storage::Metal(go), Storage::Metal(tg)) => metal::softmax_bw(sm, go, tg, rows, cols),
            (Storage::Cuda(_sm), Storage::Cuda(_go), Storage::Cuda(_tg)) => unimplemented!(),
            _ => panic!("Backend mismatch"),
        }
    }

    pub fn add_broadcast_bw_b(&self, grad_b: &mut Self, rows: usize, cols: usize) {
        match (self, grad_b) {
            (Storage::Cpu(go), Storage::Cpu(gb)) => {
                let bw = cpu::add_broadcast_bw_b(go, rows, cols);
                cpu::add_assign(gb, &bw);
            },
            #[cfg(target_os = "macos")]
            (Storage::Metal(go), Storage::Metal(gb)) => metal::add_broadcast_bw_b(go, gb, rows, cols),
            (Storage::Cuda(_go), Storage::Cuda(_gb)) => unimplemented!(),
            _ => panic!("Backend mismatch"),
        }
    }

    pub fn zero_grad(&mut self) {
        match self {
            Storage::Cpu(a) => cpu::zero_grad(a),
            #[cfg(target_os = "macos")]
            Storage::Metal(a) => metal::zero_grad(a),
            Storage::Cuda(a) => cuda::zero_grad(a),
        }
    }

    pub fn sgd_step(&mut self, grad: &Self, lr: f32) {
        match (self, grad) {
            (Storage::Cpu(a), Storage::Cpu(g)) => cpu::sgd_step(a, g, lr),
            #[cfg(target_os = "macos")]
            (Storage::Metal(a), Storage::Metal(g)) => metal::sgd_step(a, g, lr),
            (Storage::Cuda(a), Storage::Cuda(g)) => cuda::sgd_step(a, g, lr),
            _ => panic!("Backend mismatch"),
        }
    }
}
