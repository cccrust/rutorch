#[derive(Clone, Debug)]
pub struct CpuBuffer {
    pub data: Vec<f32>,
    pub length: usize,
}

impl CpuBuffer {
    pub fn new(data: &[f32]) -> Self {
        Self {
            data: data.to_vec(),
            length: data.len(),
        }
    }

    pub fn zeros(length: usize) -> Self {
        Self {
            data: vec![0.0; length],
            length,
        }
    }

    pub fn to_vec(&self) -> Vec<f32> {
        self.data.clone()
    }
}

pub fn add(a: &CpuBuffer, b: &CpuBuffer) -> CpuBuffer {
    let mut out = CpuBuffer::zeros(a.length);
    for i in 0..a.length {
        out.data[i] = a.data[i] + b.data[i];
    }
    out
}

pub fn mul(a: &CpuBuffer, b: &CpuBuffer) -> CpuBuffer {
    let mut out = CpuBuffer::zeros(a.length);
    for i in 0..a.length {
        out.data[i] = a.data[i] * b.data[i];
    }
    out
}

pub fn matmul(a: &CpuBuffer, b: &CpuBuffer, m: usize, k: usize, n: usize) -> CpuBuffer {
    let mut out = CpuBuffer::zeros(m * n);
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for l in 0..k {
                sum += a.data[i * k + l] * b.data[l * n + j];
            }
            out.data[i * n + j] = sum;
        }
    }
    out
}

pub fn relu(a: &CpuBuffer) -> CpuBuffer {
    let mut out = CpuBuffer::zeros(a.length);
    for i in 0..a.length {
        out.data[i] = if a.data[i] > 0.0 { a.data[i] } else { 0.0 };
    }
    out
}

pub fn sum(a: &CpuBuffer) -> CpuBuffer {
    let mut s = 0.0;
    for x in &a.data {
        s += x;
    }
    let mut out = CpuBuffer::zeros(1);
    out.data[0] = s;
    out
}

pub fn pow(a: &CpuBuffer, p: f32) -> CpuBuffer {
    let mut out = CpuBuffer::zeros(a.length);
    for i in 0..a.length {
        out.data[i] = a.data[i].powf(p);
    }
    out
}

pub fn log_fw(a: &CpuBuffer) -> CpuBuffer {
    let mut out = CpuBuffer::zeros(a.length);
    for i in 0..a.length {
        out.data[i] = a.data[i].ln();
    }
    out
}

pub fn softmax_fw(a: &CpuBuffer, rows: usize, cols: usize) -> CpuBuffer {
    let mut out = CpuBuffer::zeros(a.length);
    for r in 0..rows {
        let offset = r * cols;
        let mut max_val = a.data[offset];
        for c in 1..cols {
            if a.data[offset + c] > max_val {
                max_val = a.data[offset + c];
            }
        }
        let mut sum = 0.0;
        for c in 0..cols {
            let e = (a.data[offset + c] - max_val).exp();
            out.data[offset + c] = e;
            sum += e;
        }
        for c in 0..cols {
            out.data[offset + c] /= sum;
        }
    }
    out
}

pub fn log_softmax_fw(a: &CpuBuffer, rows: usize, cols: usize) -> CpuBuffer {
    let mut out = CpuBuffer::zeros(a.length);
    for r in 0..rows {
        let offset = r * cols;
        let mut max_val = a.data[offset];
        for c in 1..cols {
            if a.data[offset + c] > max_val {
                max_val = a.data[offset + c];
            }
        }
        let mut sum = 0.0;
        for c in 0..cols {
            sum += (a.data[offset + c] - max_val).exp();
        }
        let log_sum = sum.ln();
        for c in 0..cols {
            out.data[offset + c] = a.data[offset + c] - max_val - log_sum;
        }
    }
    out
}

pub fn add_broadcast(a: &CpuBuffer, b: &CpuBuffer, cols: usize) -> CpuBuffer {
    let mut out = CpuBuffer::zeros(a.length);
    for i in 0..a.length {
        out.data[i] = a.data[i] + b.data[i % cols];
    }
    out
}

// Backward generic assignment helpers
pub fn add_assign(target: &mut CpuBuffer, source: &CpuBuffer) {
    for i in 0..target.length {
        target.data[i] += source.data[i];
    }
}

pub fn relu_bw(a: &CpuBuffer, grad_out: &CpuBuffer) -> CpuBuffer {
    let mut grad_in = CpuBuffer::zeros(a.length);
    for i in 0..a.length {
        grad_in.data[i] = if a.data[i] > 0.0 { grad_out.data[i] } else { 0.0 };
    }
    grad_in
}

pub fn sum_bw(grad_out: &CpuBuffer, len: usize) -> CpuBuffer {
    let mut grad_in = CpuBuffer::zeros(len);
    for i in 0..len {
        grad_in.data[i] += grad_out.data[0];
    }
    grad_in
}

pub fn pow_bw(a: &CpuBuffer, grad_out: &CpuBuffer, p: f32) -> CpuBuffer {
    let mut grad_in = CpuBuffer::zeros(a.length);
    for i in 0..a.length {
        grad_in.data[i] = p * a.data[i].powf(p - 1.0) * grad_out.data[i];
    }
    grad_in
}

pub fn log_bw(a: &CpuBuffer, grad_out: &CpuBuffer) -> CpuBuffer {
    let mut grad_in = CpuBuffer::zeros(a.length);
    for i in 0..a.length {
        grad_in.data[i] = (1.0 / a.data[i]) * grad_out.data[i];
    }
    grad_in
}

pub fn transpose(a: &CpuBuffer, rows: usize, cols: usize) -> CpuBuffer {
    let mut out = CpuBuffer::zeros(a.length);
    for i in 0..rows {
        for j in 0..cols {
            out.data[j * rows + i] = a.data[i * cols + j];
        }
    }
    out
}

pub fn softmax_bw(sm_out: &CpuBuffer, grad_out: &CpuBuffer, rows: usize, cols: usize) -> CpuBuffer {
    let mut grad_in = CpuBuffer::zeros(sm_out.length);
    for r in 0..rows {
        let offset = r * cols;
        let mut s = 0.0;
        for c in 0..cols {
            s += grad_out.data[offset + c] * sm_out.data[offset + c];
        }
        for c in 0..cols {
           grad_in.data[offset + c] = (grad_out.data[offset + c] - s) * sm_out.data[offset + c];
        }
    }
    grad_in
}

pub fn log_softmax_bw(log_sm_out: &CpuBuffer, grad_out: &CpuBuffer, rows: usize, cols: usize) -> CpuBuffer {
    let mut grad_in = CpuBuffer::zeros(log_sm_out.length);
    for r in 0..rows {
        let offset = r * cols;
        let mut sum_grad = 0.0;
        for c in 0..cols {
            sum_grad += grad_out.data[offset + c];
        }
        for c in 0..cols {
            let sm = log_sm_out.data[offset + c].exp();
            grad_in.data[offset + c] = grad_out.data[offset + c] - sm * sum_grad;
        }
    }
    grad_in
}

pub fn add_broadcast_bw_b(grad_out: &CpuBuffer, rows: usize, cols: usize) -> CpuBuffer {
    let mut grad_b = CpuBuffer::zeros(cols);
    for r in 0..rows {
        for c in 0..cols {
            grad_b.data[c] += grad_out.data[r * cols + c];
        }
    }
    grad_b
}

pub fn zero_grad(grad: &mut CpuBuffer) {
    for i in 0..grad.length {
        grad.data[i] = 0.0;
    }
}

pub fn sgd_step(param: &mut CpuBuffer, grad: &CpuBuffer, lr: f32) {
    for i in 0..param.length {
        param.data[i] -= lr * grad.data[i];
    }
}
