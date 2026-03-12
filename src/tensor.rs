#![allow(dead_code)]
use std::cell::RefCell;
use std::collections::HashSet;
use std::hash::{Hash, Hasher};
use std::ptr;
use std::rc::Rc;
use rand::thread_rng;
use rand_distr::{Normal, Distribution};

use crate::backend::{Device, Storage};

fn numel(shape: &[usize]) -> usize {
    shape.iter().product()
}

fn strides(shape: &[usize]) -> Vec<usize> {
    let mut s = vec![1; shape.len()];
    if shape.len() < 2 {
        return s;
    }
    for i in (0..shape.len() - 1).rev() {
        s[i] = s[i + 1] * shape[i + 1];
    }
    s
}

fn permute_vec(input: &[f32], shape: &[usize], dims: &[usize]) -> Vec<f32> {
    let out_shape: Vec<usize> = dims.iter().map(|&i| shape[i]).collect();
    let in_strides = strides(shape);
    let out_strides = strides(&out_shape);
    let total = numel(&out_shape);
    let mut out = vec![0.0f32; total];

    for out_idx in 0..total {
        let mut rem = out_idx;
        let mut out_coord = vec![0usize; out_shape.len()];
        for i in 0..out_shape.len() {
            out_coord[i] = rem / out_strides[i];
            rem %= out_strides[i];
        }
        let mut in_coord = vec![0usize; shape.len()];
        for i in 0..dims.len() {
            in_coord[dims[i]] = out_coord[i];
        }
        let mut in_idx = 0usize;
        for i in 0..shape.len() {
            in_idx += in_coord[i] * in_strides[i];
        }
        out[out_idx] = input[in_idx];
    }

    out
}



// ==========================================
// 3. 張量引擎與反向傳播 (Autograd)
// ==========================================

pub enum Op {
    Leaf,
    Add(Tensor, Tensor),
    Mul(Tensor, Tensor),
    Matmul(Tensor, Tensor),
    Relu(Tensor),
    Sum(Tensor),
    Pow(Tensor, f32),   // 次方
    Sigmoid(Tensor),    // Sigmoid
    Tanh(Tensor),       // Tanh
    Lerp(Tensor, Tensor, Tensor), // a + (b - a) * gate
    Log(Tensor),        // 對數
    Softmax(Tensor),    // Softmax
    LogSoftmax(Tensor), // Log-Softmax
    LogSumExp(Tensor),  // LogSumExp
    Embedding(Tensor, Vec<usize>, usize, usize), // weight, ids, batch, time
    Reshape(Tensor),
    Transpose(Tensor),
    Permute(Tensor, Vec<usize>),
    Slice(Tensor, usize, usize),
    Concat(Tensor, Tensor, usize),
    Max(Tensor, Vec<usize>),
    AddBroadcast(Tensor, Tensor), // 廣播加法
}

pub struct TensorInner {
    pub data: Storage,
    pub grad: Storage,
    pub shape: Vec<usize>,
    pub op: Op,
    pub device: Device,
}

#[derive(Clone)]
pub struct Tensor(pub Rc<RefCell<TensorInner>>);

impl PartialEq for Tensor { fn eq(&self, other: &Self) -> bool { Rc::ptr_eq(&self.0, &other.0) } }
impl Eq for Tensor {}
impl Hash for Tensor { fn hash<H: Hasher>(&self, state: &mut H) { ptr::hash(Rc::as_ptr(&self.0), state) } }


impl Tensor {
    /// 建立 Tensor，預設使用當前系統的預設裝置 (Device::default())
    pub fn new(data: &[f32], shape: &[usize]) -> Self {
        Self::new_on(data, shape, Device::default())
    }

    /// 在指定裝置上建立 Tensor
    pub fn new_on(data: &[f32], shape: &[usize], device: Device) -> Self {
        let storage = Storage::new(data, device);
        let grad = Storage::zeros(data.len(), device);
        Self(Rc::new(RefCell::new(TensorInner { data: storage, grad, shape: shape.to_vec(), op: Op::Leaf, device })))
    }

    pub fn data(&self) -> Vec<f32> { self.0.borrow().data.to_vec() }
    pub fn grad(&self) -> Vec<f32> { self.0.borrow().grad.to_vec() }
    pub fn shape(&self) -> Vec<usize> { self.0.borrow().shape.clone() }
    pub fn device(&self) -> Device { self.0.borrow().device }

    pub fn set_data(&self, data: &[f32]) {
        let mut inner = self.0.borrow_mut();
        if data.len() != inner.data.length() {
            panic!("set_data: length mismatch");
        }
        inner.data = Storage::new(data, inner.device);
    }

    pub fn set_grad(&self, grad: &[f32]) {
        let mut inner = self.0.borrow_mut();
        if grad.len() != inner.grad.length() {
            panic!("set_grad: length mismatch");
        }
        inner.grad = Storage::new(grad, inner.device);
    }

    /// 在預設裝置上產生標準常態分佈隨機張量
    pub fn randn(shape: &[usize]) -> Self {
        Self::randn_on(shape, Device::default())
    }

    /// 在指定裝置上產生標準常態分佈隨機張量
    pub fn randn_on(shape: &[usize], device: Device) -> Self {
        let mut rng = thread_rng();
        let normal = Normal::new(0.0, 0.1).unwrap(); 
        
        let length: usize = shape.iter().product();
        let mut data = Vec::with_capacity(length);
        for _ in 0..length {
            data.push(normal.sample(&mut rng) as f32);
        }
        
        Self::new_on(&data, shape, device)
    }

    /// 將張量轉移到目標裝置
    pub fn to_device(&self, device: Device) -> Self {
        if self.device() == device {
            return self.clone();
        }
        let data = self.data();
        let shape = self.shape();
        Self::new_on(&data, &shape, device)
    }

    
    pub fn add(&self, other: &Tensor) -> Tensor {
        let a_shape = self.shape();
        let b_shape = other.shape();

        if a_shape == b_shape {
            let storage = self.0.borrow().data.add(&other.0.borrow().data);
            return Self(Rc::new(RefCell::new(TensorInner {
                data: storage,
                grad: Storage::zeros(self.0.borrow().data.length(), self.device()),
                shape: a_shape,
                op: Op::Add(self.clone(), other.clone()),
                device: self.device(),
            })));
        }

        if a_shape.len() == 2 && b_shape.len() == 1 && a_shape[1] == b_shape[0] {
            return self.add_broadcast(other);
        }

        if a_shape.len() == 1 && b_shape.len() == 2 && b_shape[1] == a_shape[0] {
            return other.add_broadcast(self);
        }

        panic!("Unsupported broadcast for add: {:?} + {:?}", a_shape, b_shape);
    }

    pub fn mul(&self, other: &Tensor) -> Tensor {
        let storage = self.0.borrow().data.mul(&other.0.borrow().data);
        Self(Rc::new(RefCell::new(TensorInner {
            data: storage, grad: Storage::zeros(self.0.borrow().data.length(), self.device()),
            shape: self.shape(), op: Op::Mul(self.clone(), other.clone()), device: self.device()
        })))
    }

    pub fn mul_scalar(&self, s: f32) -> Tensor {
        let len = self.0.borrow().data.length();
        let scalar = Tensor::new_on(&vec![s; len], &self.shape(), self.device());
        self.mul(&scalar)
    }

    pub fn matmul(&self, other: &Tensor) -> Tensor {
        let m = self.shape()[0];
        let k = self.shape()[1];
        let n = other.shape()[1];
        let storage = self.0.borrow().data.matmul(&other.0.borrow().data, m, k, n);
        Self(Rc::new(RefCell::new(TensorInner {
            data: storage, grad: Storage::zeros(m * n, self.device()),
            shape: vec![m, n], op: Op::Matmul(self.clone(), other.clone()), device: self.device()
        })))
    }

    pub fn relu(&self) -> Tensor {
        let storage = self.0.borrow().data.relu();
        Self(Rc::new(RefCell::new(TensorInner {
            data: storage, grad: Storage::zeros(self.0.borrow().data.length(), self.device()),
            shape: self.shape(), op: Op::Relu(self.clone()), device: self.device()
        })))
    }

    pub fn sigmoid(&self) -> Tensor {
        let data = self.data();
        let mut out = Vec::with_capacity(data.len());
        for v in data {
            out.push(1.0 / (1.0 + (-v).exp()));
        }
        let storage = Storage::new(&out, self.device());
        Self(Rc::new(RefCell::new(TensorInner {
            data: storage,
            grad: Storage::zeros(out.len(), self.device()),
            shape: self.shape(),
            op: Op::Sigmoid(self.clone()),
            device: self.device(),
        })))
    }

    pub fn tanh(&self) -> Tensor {
        let data = self.data();
        let mut out = Vec::with_capacity(data.len());
        for v in data {
            out.push(v.tanh());
        }
        let storage = Storage::new(&out, self.device());
        Self(Rc::new(RefCell::new(TensorInner {
            data: storage,
            grad: Storage::zeros(out.len(), self.device()),
            shape: self.shape(),
            op: Op::Tanh(self.clone()),
            device: self.device(),
        })))
    }

    pub fn lerp(&self, other: &Tensor, gate: &Tensor) -> Tensor {
        let diff = other.sub(self);
        let out = self.add(&diff.mul(gate));
        let mut inner = out.0.borrow_mut();
        inner.op = Op::Lerp(self.clone(), other.clone(), gate.clone());
        drop(inner);
        out
    }

    pub fn sum(&self) -> Tensor {
        let storage = self.0.borrow().data.sum();
        Self(Rc::new(RefCell::new(TensorInner {
            data: storage, grad: Storage::zeros(1, self.device()),
            shape: vec![1, 1], op: Op::Sum(self.clone()), device: self.device()
        })))
    }

    pub fn pow(&self, p: f32) -> Tensor {
        let storage = self.0.borrow().data.pow(p);
        Self(Rc::new(RefCell::new(TensorInner {
            data: storage, grad: Storage::zeros(self.0.borrow().data.length(), self.device()),
            shape: self.shape(), op: Op::Pow(self.clone(), p), device: self.device()
        })))
    }

    pub fn log(&self) -> Tensor {
        let storage = self.0.borrow().data.log_fw();
        Self(Rc::new(RefCell::new(TensorInner {
            data: storage, grad: Storage::zeros(self.0.borrow().data.length(), self.device()),
            shape: self.shape(), op: Op::Log(self.clone()), device: self.device()
        })))
    }

    pub fn softmax(&self) -> Tensor {
        let shape = self.shape();
        let (rows, cols) = if shape.len() == 2 { (shape[0], shape[1]) } else { (1, shape[0]) };
        let storage = self.0.borrow().data.softmax_fw(rows, cols);
        Self(Rc::new(RefCell::new(TensorInner {
            data: storage, grad: Storage::zeros(rows * cols, self.device()),
            shape: self.shape(), op: Op::Softmax(self.clone()), device: self.device()
        })))
    }

    pub fn log_softmax(&self) -> Tensor {
        let shape = self.shape();
        let (rows, cols) = if shape.len() == 2 { (shape[0], shape[1]) } else { (1, shape[0]) };
        let storage = self.0.borrow().data.log_softmax_fw(rows, cols);
        Self(Rc::new(RefCell::new(TensorInner {
            data: storage, grad: Storage::zeros(rows * cols, self.device()),
            shape: self.shape(), op: Op::LogSoftmax(self.clone()), device: self.device()
        })))
    }

    /// 教學用：對最後一維做 logsumexp (穩定版)
    pub fn logsumexp(&self) -> Tensor {
        let shape = self.shape();
        let (rows, cols) = if shape.len() == 2 { (shape[0], shape[1]) } else { (1, shape[0]) };
        let data = self.data();
        let mut out = vec![0.0f32; rows];
        for r in 0..rows {
            let offset = r * cols;
            let mut max_val = data[offset];
            for c in 1..cols {
                if data[offset + c] > max_val { max_val = data[offset + c]; }
            }
            let mut sum = 0.0;
            for c in 0..cols {
                sum += (data[offset + c] - max_val).exp();
            }
            out[r] = max_val + sum.ln();
        }
        let storage = Storage::new(&out, self.device());
        Self(Rc::new(RefCell::new(TensorInner {
            data: storage,
            grad: Storage::zeros(rows, self.device()),
            shape: vec![rows, 1],
            op: Op::LogSumExp(self.clone()),
            device: self.device(),
        })))
    }

    pub fn reshape(&self, shape: &[usize]) -> Tensor {
        let total = numel(shape);
        if total != self.0.borrow().data.length() {
            panic!("reshape: element count mismatch");
        }
        let storage = self.0.borrow().data.clone();
        Self(Rc::new(RefCell::new(TensorInner {
            data: storage,
            grad: Storage::zeros(total, self.device()),
            shape: shape.to_vec(),
            op: Op::Reshape(self.clone()),
            device: self.device(),
        })))
    }

    pub fn transpose(&self) -> Tensor {
        let shape = self.shape();
        if shape.len() != 2 {
            panic!("transpose only supports 2D tensors");
        }
        let rows = shape[0];
        let cols = shape[1];
        let storage = self.0.borrow().data.transpose(rows, cols);
        Self(Rc::new(RefCell::new(TensorInner {
            data: storage,
            grad: Storage::zeros(rows * cols, self.device()),
            shape: vec![cols, rows],
            op: Op::Transpose(self.clone()),
            device: self.device(),
        })))
    }

    pub fn permute(&self, dims: &[usize]) -> Tensor {
        let shape = self.shape();
        if dims.len() != shape.len() {
            panic!("permute: dims length mismatch");
        }
        let mut seen = vec![false; shape.len()];
        for &d in dims {
            if d >= shape.len() || seen[d] {
                panic!("permute: invalid dims");
            }
            seen[d] = true;
        }
        let data = self.data();
        let out_data = permute_vec(&data, &shape, dims);
        let out_shape: Vec<usize> = dims.iter().map(|&i| shape[i]).collect();
        let storage = Storage::new(&out_data, self.device());
        Self(Rc::new(RefCell::new(TensorInner {
            data: storage,
            grad: Storage::zeros(out_data.len(), self.device()),
            shape: out_shape,
            op: Op::Permute(self.clone(), dims.to_vec()),
            device: self.device(),
        })))
    }

    pub fn slice(&self, start: usize, end: usize) -> Tensor {
        let shape = self.shape();
        if start >= end {
            panic!("slice: invalid range");
        }
        if shape.len() == 1 {
            if end > shape[0] { panic!("slice: out of bounds"); }
            let data = self.data();
            let out_data = data[start..end].to_vec();
            let storage = Storage::new(&out_data, self.device());
            return Self(Rc::new(RefCell::new(TensorInner {
                data: storage,
                grad: Storage::zeros(out_data.len(), self.device()),
                shape: vec![end - start],
                op: Op::Slice(self.clone(), start, end),
                device: self.device(),
            })));
        }
        if shape.len() == 2 {
            let rows = shape[0];
            let cols = shape[1];
            if end > rows { panic!("slice: out of bounds"); }
            let data = self.data();
            let mut out = Vec::with_capacity((end - start) * cols);
            for r in start..end {
                let offset = r * cols;
                out.extend_from_slice(&data[offset..offset + cols]);
            }
            let storage = Storage::new(&out, self.device());
            return Self(Rc::new(RefCell::new(TensorInner {
                data: storage,
                grad: Storage::zeros(out.len(), self.device()),
                shape: vec![end - start, cols],
                op: Op::Slice(self.clone(), start, end),
                device: self.device(),
            })));
        }
        panic!("slice only supports 1D or 2D tensors (rows slice)");
    }

    pub fn concat(&self, other: &Tensor, axis: usize) -> Tensor {
        if axis != 0 {
            panic!("concat: only axis=0 supported");
        }
        let a_shape = self.shape();
        let b_shape = other.shape();
        if a_shape.len() != b_shape.len() {
            panic!("concat: rank mismatch");
        }
        if a_shape.len() == 1 {
            let mut out = self.data();
            out.extend_from_slice(&other.data());
            let storage = Storage::new(&out, self.device());
            return Self(Rc::new(RefCell::new(TensorInner {
                data: storage,
                grad: Storage::zeros(out.len(), self.device()),
                shape: vec![a_shape[0] + b_shape[0]],
                op: Op::Concat(self.clone(), other.clone(), axis),
                device: self.device(),
            })));
        }
        if a_shape.len() == 2 {
            if a_shape[1] != b_shape[1] {
                panic!("concat: cols mismatch");
            }
            let mut out = self.data();
            out.extend_from_slice(&other.data());
            let storage = Storage::new(&out, self.device());
            return Self(Rc::new(RefCell::new(TensorInner {
                data: storage,
                grad: Storage::zeros(out.len(), self.device()),
                shape: vec![a_shape[0] + b_shape[0], a_shape[1]],
                op: Op::Concat(self.clone(), other.clone(), axis),
                device: self.device(),
            })));
        }
        panic!("concat only supports 1D or 2D tensors");
    }

    pub fn mean(&self) -> Tensor {
        let total = self.0.borrow().data.length() as f32;
        let denom = Tensor::new_on(&[1.0 / total], &[1, 1], self.device());
        self.sum().mul(&denom)
    }

    pub fn max(&self) -> Tensor {
        let data = self.data();
        if data.is_empty() {
            panic!("max: empty tensor");
        }
        let mut max_val = data[0];
        for &v in &data[1..] {
            if v > max_val { max_val = v; }
        }
        let mut idxs = Vec::new();
        for (i, &v) in data.iter().enumerate() {
            if v == max_val { idxs.push(i); }
        }
        let storage = Storage::new(&[max_val], self.device());
        Self(Rc::new(RefCell::new(TensorInner {
            data: storage,
            grad: Storage::zeros(1, self.device()),
            shape: vec![1, 1],
            op: Op::Max(self.clone(), idxs),
            device: self.device(),
        })))
    }

    pub fn neg(&self) -> Tensor {
        let len = self.0.borrow().data.length();
        let neg_ones = Tensor::new_on(&vec![-1.0; len], &self.shape(), self.device());
        self.mul(&neg_ones)
    }

    pub fn sub(&self, other: &Tensor) -> Tensor {
        self.add(&other.neg())
    }

    pub fn div(&self, other: &Tensor) -> Tensor {
        self.mul(&other.pow(-1.0))
    }

    pub fn cross_entropy(&self, yb: &Tensor) -> Tensor {
        let log_probs = self.log();
        let zb = yb.mul(&log_probs);
        zb.sum().neg()
    }

    /// 穩定版 NLL loss：logits -> log_softmax -> -sum(y * log_probs)
    pub fn nll_loss(&self, yb: &Tensor) -> Tensor {
        let log_probs = self.log_softmax();
        let zb = yb.mul(&log_probs);
        zb.sum().neg()
    }

    pub fn add_broadcast(&self, b: &Tensor) -> Tensor {
        let cols = b.0.borrow().data.length();
        let storage = self.0.borrow().data.add_broadcast(&b.0.borrow().data, cols);
        Self(Rc::new(RefCell::new(TensorInner {
            data: storage, grad: Storage::zeros(self.0.borrow().data.length(), self.device()),
            shape: self.shape(), op: Op::AddBroadcast(self.clone(), b.clone()), device: self.device()
        })))
    }
    
    pub fn backward(&self) {
        let mut topo = Vec::new();
        let mut visited = HashSet::new();

        fn build_topo(v: &Tensor, visited: &mut HashSet<Tensor>, topo: &mut Vec<Tensor>) {
            if !visited.contains(v) {
                visited.insert(v.clone());
                match &v.0.borrow().op {
                    Op::Add(a, b) | Op::Mul(a, b) | Op::Matmul(a, b) | Op::AddBroadcast(a, b) => { build_topo(a, visited, topo); build_topo(b, visited, topo); }
                    Op::Relu(a) | Op::Sigmoid(a) | Op::Tanh(a) | Op::Sum(a) | Op::Pow(a, _) | Op::Log(a) | Op::Softmax(a) | Op::LogSoftmax(a) | Op::LogSumExp(a) | Op::Embedding(a, _, _, _) | Op::Lerp(a, _, _)
                    | Op::Reshape(a) | Op::Transpose(a) | Op::Permute(a, _) | Op::Slice(a, _, _) | Op::Max(a, _) => { build_topo(a, visited, topo); }
                    Op::Lerp(a, b, g) => { build_topo(a, visited, topo); build_topo(b, visited, topo); build_topo(g, visited, topo); }
                    Op::Concat(a, b, _) => { build_topo(a, visited, topo); build_topo(b, visited, topo); }
                    Op::Leaf => {}
                }
                topo.push(v.clone());
            }
        }
        build_topo(self, &mut visited, &mut topo);

        let len = self.0.borrow().data.length();
        let ones = vec![1.0; len];
        self.0.borrow_mut().grad = Storage::new(&ones, self.device());

        for node in topo.into_iter().rev() {
            let inner = node.0.borrow_mut();
            let grad = inner.grad.clone();

            match &inner.op {
                Op::Add(a, b) => {
                    a.0.borrow_mut().grad.add_assign(&grad);
                    b.0.borrow_mut().grad.add_assign(&grad);
                }
                Op::Mul(a, b) => {
                    let tmp_a = b.0.borrow().data.mul(&grad);
                    a.0.borrow_mut().grad.add_assign(&tmp_a);

                    let tmp_b = a.0.borrow().data.mul(&grad);
                    b.0.borrow_mut().grad.add_assign(&tmp_b);
                }
                Op::Matmul(a, b) => {
                    let m = a.shape()[0];
                    let k = a.shape()[1];
                    let n = b.shape()[1];
                    let b_t = b.0.borrow().data.transpose(k, n);
                    let a_grad_update = grad.matmul(&b_t, m, n, k);
                    a.0.borrow_mut().grad.add_assign(&a_grad_update);

                    let a_t = a.0.borrow().data.transpose(m, k);
                    let b_grad_update = a_t.matmul(&grad, k, m, n);
                    b.0.borrow_mut().grad.add_assign(&b_grad_update);
                }
                Op::Relu(a) => {
                    let tmp = a.0.borrow().data.relu_bw(&grad);
                    a.0.borrow_mut().grad.add_assign(&tmp);
                }
                Op::Sigmoid(a) => {
                    let sig = inner.data.to_vec();
                    let grad_out = grad.to_vec();
                    let mut grad_in = vec![0.0f32; sig.len()];
                    for i in 0..sig.len() {
                        grad_in[i] = sig[i] * (1.0 - sig[i]) * grad_out[i];
                    }
                    let grad_storage = Storage::new(&grad_in, a.device());
                    a.0.borrow_mut().grad.add_assign(&grad_storage);
                }
                Op::Tanh(a) => {
                    let t = inner.data.to_vec();
                    let grad_out = grad.to_vec();
                    let mut grad_in = vec![0.0f32; t.len()];
                    for i in 0..t.len() {
                        grad_in[i] = (1.0 - t[i] * t[i]) * grad_out[i];
                    }
                    let grad_storage = Storage::new(&grad_in, a.device());
                    a.0.borrow_mut().grad.add_assign(&grad_storage);
                }
                Op::Lerp(a, b, g) => {
                    let gate = g.data();
                    let grad_out = grad.to_vec();
                    let mut grad_a = vec![0.0f32; grad_out.len()];
                    let mut grad_b = vec![0.0f32; grad_out.len()];
                    let mut grad_g = vec![0.0f32; grad_out.len()];
                    for i in 0..grad_out.len() {
                        grad_a[i] = grad_out[i] * (1.0 - gate[i]);
                        grad_b[i] = grad_out[i] * gate[i];
                        grad_g[i] = grad_out[i] * (b.data()[i] - a.data()[i]);
                    }
                    let ga = Storage::new(&grad_a, a.device());
                    let gb = Storage::new(&grad_b, b.device());
                    let gg = Storage::new(&grad_g, g.device());
                    a.0.borrow_mut().grad.add_assign(&ga);
                    b.0.borrow_mut().grad.add_assign(&gb);
                    g.0.borrow_mut().grad.add_assign(&gg);
                }
                Op::Sum(a) => {
                    let tmp = a.0.borrow().data.sum_bw(&grad);
                    a.0.borrow_mut().grad.add_assign(&tmp);
                }
                Op::Pow(a, p) => {
                    let tmp = a.0.borrow().data.pow_bw(&grad, *p);
                    a.0.borrow_mut().grad.add_assign(&tmp);
                }
                Op::Log(a) => {
                    let tmp = a.0.borrow().data.log_bw(&grad);
                    a.0.borrow_mut().grad.add_assign(&tmp);
                }
                Op::Softmax(a) => {
                    let shape = a.shape();
                    let (rows, cols) = if shape.len() == 2 { (shape[0], shape[1]) } else { (1, shape[0]) };
                    inner.data.softmax_bw(&grad, &mut a.0.borrow_mut().grad, rows, cols);
                }
                Op::LogSoftmax(a) => {
                    let shape = a.shape();
                    let (rows, cols) = if shape.len() == 2 { (shape[0], shape[1]) } else { (1, shape[0]) };
                    inner.data.log_softmax_bw(&grad, &mut a.0.borrow_mut().grad, rows, cols);
                }
                Op::LogSumExp(a) => {
                    let shape = a.shape();
                    let (rows, cols) = if shape.len() == 2 { (shape[0], shape[1]) } else { (1, shape[0]) };
                    let a_data = a.data();
                    let grad_out = grad.to_vec();
                    let mut grad_in = vec![0.0f32; rows * cols];

                    for r in 0..rows {
                        let offset = r * cols;
                        let mut max_val = a_data[offset];
                        for c in 1..cols {
                            if a_data[offset + c] > max_val { max_val = a_data[offset + c]; }
                        }
                        let mut sum = 0.0;
                        for c in 0..cols {
                            sum += (a_data[offset + c] - max_val).exp();
                        }
                        for c in 0..cols {
                            let sm = (a_data[offset + c] - max_val).exp() / sum;
                            grad_in[offset + c] = grad_out[r] * sm;
                        }
                    }

                    let grad_storage = Storage::new(&grad_in, a.device());
                    a.0.borrow_mut().grad.add_assign(&grad_storage);
                }
                Op::Embedding(weight, ids, batch, time) => {
                    let dim = weight.shape()[1];
                    let grad_out = grad.to_vec();
                    let mut grad_w = vec![0.0f32; weight.0.borrow().data.length()];
                    for b in 0..*batch {
                        for t in 0..*time {
                            let id = ids[b * *time + t];
                            let w_off = id * dim;
                            let g_off = (b * *time + t) * dim;
                            for d in 0..dim {
                                grad_w[w_off + d] += grad_out[g_off + d];
                            }
                        }
                    }
                    let grad_storage = Storage::new(&grad_w, weight.device());
                    weight.0.borrow_mut().grad.add_assign(&grad_storage);
                }
                Op::Reshape(a) => {
                    a.0.borrow_mut().grad.add_assign(&grad);
                }
                Op::Transpose(a) => {
                    let shape = a.shape();
                    if shape.len() != 2 { panic!("transpose backward only supports 2D"); }
                    let rows = shape[0];
                    let cols = shape[1];
                    let grad_t = grad.transpose(cols, rows);
                    a.0.borrow_mut().grad.add_assign(&grad_t);
                }
                Op::Permute(a, dims) => {
                    let a_shape = a.shape();
                    let out_shape: Vec<usize> = dims.iter().map(|&i| a_shape[i]).collect();
                    let mut inv = vec![0usize; dims.len()];
                    for (i, &d) in dims.iter().enumerate() { inv[d] = i; }
                    let grad_out = grad.to_vec();
                    let grad_in_vec = permute_vec(&grad_out, &out_shape, &inv);
                    let grad_in = Storage::new(&grad_in_vec, a.device());
                    a.0.borrow_mut().grad.add_assign(&grad_in);
                }
                Op::Slice(a, start, end) => {
                    let shape = a.shape();
                    let mut grad_in = vec![0.0f32; a.0.borrow().data.length()];
                    if shape.len() == 1 {
                        let go = grad.to_vec();
                        for i in 0..(end - start) {
                            grad_in[start + i] = go[i];
                        }
                    } else if shape.len() == 2 {
                        let cols = shape[1];
                        let go = grad.to_vec();
                        let mut idx = 0usize;
                        for r in *start..*end {
                            let base = r * cols;
                            for c in 0..cols {
                                grad_in[base + c] = go[idx];
                                idx += 1;
                            }
                        }
                    } else {
                        panic!("slice backward only supports 1D or 2D");
                    }
                    let grad_storage = Storage::new(&grad_in, a.device());
                    a.0.borrow_mut().grad.add_assign(&grad_storage);
                }
                Op::Concat(a, b, axis) => {
                    if *axis != 0 { panic!("concat backward only supports axis=0"); }
                    let a_shape = a.shape();
                    let go = grad.to_vec();
                    if a_shape.len() == 1 {
                        let a_len = a_shape[0];
                        let grad_a = Storage::new(&go[0..a_len], a.device());
                        let grad_b = Storage::new(&go[a_len..], b.device());
                        a.0.borrow_mut().grad.add_assign(&grad_a);
                        b.0.borrow_mut().grad.add_assign(&grad_b);
                    } else if a_shape.len() == 2 {
                        let cols = a_shape[1];
                        let a_rows = a_shape[0];
                        let a_len = a_rows * cols;
                        let grad_a = Storage::new(&go[0..a_len], a.device());
                        let grad_b = Storage::new(&go[a_len..], b.device());
                        a.0.borrow_mut().grad.add_assign(&grad_a);
                        b.0.borrow_mut().grad.add_assign(&grad_b);
                    } else {
                        panic!("concat backward only supports 1D or 2D");
                    }
                }
                Op::Max(a, idxs) => {
                    let mut grad_in = vec![0.0f32; a.0.borrow().data.length()];
                    let go = grad.to_vec();
                    let g = go[0];
                    for &i in idxs {
                        grad_in[i] += g;
                    }
                    let grad_storage = Storage::new(&grad_in, a.device());
                    a.0.borrow_mut().grad.add_assign(&grad_storage);
                }
                Op::AddBroadcast(a, b) => {
                    a.0.borrow_mut().grad.add_assign(&grad);
                    
                    let cols = b.0.borrow().data.length();
                    let rows = grad.length() / cols;
                    grad.add_broadcast_bw_b(&mut b.0.borrow_mut().grad, rows, cols);
                }
                Op::Leaf => {}
            }
        }
    }

    // --- 優化器方法 (Optimizer) ---

    /// 清空這個張量的梯度 (在每個 Batch 訓練開始前呼叫)
    pub fn zero_grad(&self) {
        self.0.borrow_mut().grad.zero_grad();
    }

    /// SGD 梯度下降更新：W = W - lr * W.grad
    pub fn step(&self, lr: f32) {
        let mut inner = self.0.borrow_mut();
        let grad = inner.grad.clone();
        inner.data.sgd_step(&grad, lr);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(a: f32, b: f32, eps: f32) {
        if (a - b).abs() > eps {
            panic!("assert_close failed: {} vs {}", a, b);
        }
    }

    fn assert_vec_close(a: &[f32], b: &[f32], eps: f32) {
        if a.len() != b.len() {
            panic!("len mismatch: {} vs {}", a.len(), b.len());
        }
        for i in 0..a.len() {
            assert_close(a[i], b[i], eps);
        }
    }

    fn gradcheck_single(
        data: &[f32],
        shape: &[usize],
        eps: f32,
        tol: f32,
        f: impl Fn(&Tensor) -> Tensor,
    ) {
        let x = Tensor::new_on(data, shape, Device::Cpu);
        let loss = f(&x);
        loss.backward();
        let grad_auto = x.grad();

        let base = x.data();
        let mut grad_num = vec![0.0f32; base.len()];
        for i in 0..base.len() {
            let mut plus = base.clone();
            plus[i] += eps;
            let lp = f(&Tensor::new_on(&plus, shape, Device::Cpu)).data()[0];

            let mut minus = base.clone();
            minus[i] -= eps;
            let lm = f(&Tensor::new_on(&minus, shape, Device::Cpu)).data()[0];
            grad_num[i] = (lp - lm) / (2.0 * eps);
        }

        for i in 0..grad_auto.len() {
            let diff = (grad_auto[i] - grad_num[i]).abs();
            if diff > tol {
                panic!(
                    "gradcheck_single failed at {}: auto={}, num={}, diff={}",
                    i, grad_auto[i], grad_num[i], diff
                );
            }
        }
    }

    fn gradcheck_two_inputs(
        a_data: &[f32],
        a_shape: &[usize],
        b_data: &[f32],
        b_shape: &[usize],
        eps: f32,
        tol: f32,
        f: impl Fn(&Tensor, &Tensor) -> Tensor,
    ) {
        let a = Tensor::new_on(a_data, a_shape, Device::Cpu);
        let b = Tensor::new_on(b_data, b_shape, Device::Cpu);
        let loss = f(&a, &b);
        loss.backward();
        let grad_a = a.grad();
        let grad_b = b.grad();

        let a_base = a.data();
        let mut grad_a_num = vec![0.0f32; a_base.len()];
        for i in 0..a_base.len() {
            let mut plus = a_base.clone();
            plus[i] += eps;
            let lp = f(&Tensor::new_on(&plus, a_shape, Device::Cpu), &b).data()[0];

            let mut minus = a_base.clone();
            minus[i] -= eps;
            let lm = f(&Tensor::new_on(&minus, a_shape, Device::Cpu), &b).data()[0];
            grad_a_num[i] = (lp - lm) / (2.0 * eps);
        }

        let b_base = b.data();
        let mut grad_b_num = vec![0.0f32; b_base.len()];
        for i in 0..b_base.len() {
            let mut plus = b_base.clone();
            plus[i] += eps;
            let lp = f(&a, &Tensor::new_on(&plus, b_shape, Device::Cpu)).data()[0];

            let mut minus = b_base.clone();
            minus[i] -= eps;
            let lm = f(&a, &Tensor::new_on(&minus, b_shape, Device::Cpu)).data()[0];
            grad_b_num[i] = (lp - lm) / (2.0 * eps);
        }

        for i in 0..grad_a.len() {
            let diff = (grad_a[i] - grad_a_num[i]).abs();
            if diff > tol {
                panic!(
                    "gradcheck_two_inputs (a) failed at {}: auto={}, num={}, diff={}",
                    i, grad_a[i], grad_a_num[i], diff
                );
            }
        }
        for i in 0..grad_b.len() {
            let diff = (grad_b[i] - grad_b_num[i]).abs();
            if diff > tol {
                panic!(
                    "gradcheck_two_inputs (b) failed at {}: auto={}, num={}, diff={}",
                    i, grad_b[i], grad_b_num[i], diff
                );
            }
        }
    }

    #[test]
    fn reshape_keeps_data_and_grad() {
        let x = Tensor::new_on(&[1.0, 2.0, 3.0, 4.0], &[2, 2], Device::Cpu);
        let y = x.reshape(&[4, 1]);
        assert_eq!(y.shape(), vec![4, 1]);
        assert_eq!(y.data(), vec![1.0, 2.0, 3.0, 4.0]);

        let loss = y.sum();
        loss.backward();
        assert_eq!(x.grad(), vec![1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn transpose_2d() {
        let x = Tensor::new_on(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], Device::Cpu);
        let y = x.transpose();
        assert_eq!(y.shape(), vec![3, 2]);
        assert_eq!(y.data(), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn permute_3d() {
        let data: Vec<f32> = (0..12).map(|v| v as f32).collect();
        let x = Tensor::new_on(&data, &[2, 3, 2], Device::Cpu);
        let y = x.permute(&[1, 0, 2]);
        assert_eq!(y.shape(), vec![3, 2, 2]);

        let mut expected = vec![0.0f32; 12];
        for i in 0..2 {
            for j in 0..3 {
                for k in 0..2 {
                    let in_idx = i * 3 * 2 + j * 2 + k;
                    let out_i = j;
                    let out_j = i;
                    let out_k = k;
                    let out_idx = out_i * 2 * 2 + out_j * 2 + out_k;
                    expected[out_idx] = data[in_idx];
                }
            }
        }
        assert_eq!(y.data(), expected);
    }

    #[test]
    fn slice_1d_and_grad() {
        let data: Vec<f32> = (0..10).map(|v| v as f32).collect();
        let x = Tensor::new_on(&data, &[10], Device::Cpu);
        let y = x.slice(2, 5);
        assert_eq!(y.shape(), vec![3]);
        assert_eq!(y.data(), vec![2.0, 3.0, 4.0]);

        let loss = y.sum();
        loss.backward();
        assert_eq!(x.grad(), vec![0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn slice_2d_rows_and_grad() {
        let x = Tensor::new_on(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2], Device::Cpu);
        let y = x.slice(1, 3);
        assert_eq!(y.shape(), vec![2, 2]);
        assert_eq!(y.data(), vec![3.0, 4.0, 5.0, 6.0]);

        let loss = y.sum();
        loss.backward();
        assert_eq!(x.grad(), vec![0.0, 0.0, 1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn concat_1d_and_grad() {
        let a = Tensor::new_on(&[1.0, 2.0], &[2], Device::Cpu);
        let b = Tensor::new_on(&[3.0, 4.0, 5.0], &[3], Device::Cpu);
        let y = a.concat(&b, 0);
        assert_eq!(y.shape(), vec![5]);
        assert_eq!(y.data(), vec![1.0, 2.0, 3.0, 4.0, 5.0]);

        let loss = y.sum();
        loss.backward();
        assert_eq!(a.grad(), vec![1.0, 1.0]);
        assert_eq!(b.grad(), vec![1.0, 1.0, 1.0]);
    }

    #[test]
    fn concat_2d_and_grad() {
        let a = Tensor::new_on(&[1.0, 2.0, 3.0, 4.0], &[2, 2], Device::Cpu);
        let b = Tensor::new_on(&[5.0, 6.0], &[1, 2], Device::Cpu);
        let y = a.concat(&b, 0);
        assert_eq!(y.shape(), vec![3, 2]);
        assert_eq!(y.data(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        let loss = y.sum();
        loss.backward();
        assert_eq!(a.grad(), vec![1.0, 1.0, 1.0, 1.0]);
        assert_eq!(b.grad(), vec![1.0, 1.0]);
    }

    #[test]
    fn mean_and_grad() {
        let x = Tensor::new_on(&[1.0, 2.0, 3.0, 4.0], &[4], Device::Cpu);
        let y = x.mean();
        assert_eq!(y.shape(), vec![1, 1]);
        assert_close(y.data()[0], 2.5, 1e-6);
        y.backward();
        assert_vec_close(&x.grad(), &[0.25, 0.25, 0.25, 0.25], 1e-6);
    }

    #[test]
    fn max_and_grad() {
        let x = Tensor::new_on(&[1.0, 5.0, 3.0, 5.0], &[4], Device::Cpu);
        let y = x.max();
        assert_eq!(y.data(), vec![5.0]);
        y.backward();
        assert_eq!(x.grad(), vec![0.0, 1.0, 0.0, 1.0]);
    }

    #[test]
    fn add_broadcast_2d_1d_and_grad() {
        let a = Tensor::new_on(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], Device::Cpu);
        let b = Tensor::new_on(&[10.0, 20.0, 30.0], &[3], Device::Cpu);
        let y = a.add(&b);
        assert_eq!(y.shape(), vec![2, 3]);
        assert_eq!(y.data(), vec![11.0, 22.0, 33.0, 14.0, 25.0, 36.0]);

        let loss = y.sum();
        loss.backward();
        assert_eq!(a.grad(), vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
        assert_eq!(b.grad(), vec![2.0, 2.0, 2.0]);
    }

    #[test]
    fn gradcheck_matmul() {
        let a = &[0.2, -1.1, 0.7, 1.3, -0.4, 0.9];
        let b = &[1.0, -0.5, 0.8, 0.3, -1.2, 0.4];
        gradcheck_two_inputs(
            a, &[2, 3],
            b, &[3, 2],
            1e-3, 1e-2,
            |x, y| x.matmul(y).sum(),
        );
    }

    #[test]
    fn gradcheck_relu() {
        let data = &[0.3, 1.1, 2.5, 0.7];
        gradcheck_single(data, &[2, 2], 1e-3, 1e-3, |x| x.relu().sum());
    }

    #[test]
    fn gradcheck_add_broadcast() {
        let a = &[0.2, -0.3, 1.0, 0.5, -0.7, 2.0];
        let b = &[0.1, -0.2, 0.3];
        gradcheck_two_inputs(
            a, &[2, 3],
            b, &[3],
            1e-3, 1e-2,
            |x, y| x.add(y).sum(),
        );
    }

    #[test]
    fn gradcheck_permute() {
        let data: Vec<f32> = (0..12).map(|v| v as f32 * 0.1 + 0.2).collect();
        gradcheck_single(&data, &[2, 3, 2], 1e-3, 1e-3, |x| x.permute(&[1, 0, 2]).sum());
    }

    #[test]
    fn gradcheck_slice_rows() {
        let data: Vec<f32> = (0..12).map(|v| v as f32 * 0.1 - 0.3).collect();
        gradcheck_single(&data, &[4, 3], 1e-3, 1e-3, |x| x.slice(1, 3).sum());
    }
}
