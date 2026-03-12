#![allow(dead_code)]
use std::cell::RefCell;
use std::collections::HashSet;
use std::hash::{Hash, Hasher};
use std::ptr;
use std::rc::Rc;
use rand::thread_rng;
use rand_distr::{Normal, Distribution};

use crate::backend::{Device, Storage};




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
    Log(Tensor),        // 對數
    Softmax(Tensor),    // Softmax
    LogSoftmax(Tensor), // Log-Softmax
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
        let storage = self.0.borrow().data.add(&other.0.borrow().data);
        Self(Rc::new(RefCell::new(TensorInner {
            data: storage, grad: Storage::zeros(self.0.borrow().data.length(), self.device()),
            shape: self.shape(), op: Op::Add(self.clone(), other.clone()), device: self.device()
        })))
    }

    pub fn mul(&self, other: &Tensor) -> Tensor {
        let storage = self.0.borrow().data.mul(&other.0.borrow().data);
        Self(Rc::new(RefCell::new(TensorInner {
            data: storage, grad: Storage::zeros(self.0.borrow().data.length(), self.device()),
            shape: self.shape(), op: Op::Mul(self.clone(), other.clone()), device: self.device()
        })))
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
                    Op::Relu(a) | Op::Sum(a) | Op::Pow(a, _) | Op::Log(a) | Op::Softmax(a) | Op::LogSoftmax(a) => { build_topo(a, visited, topo); }
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
            let mut inner = node.0.borrow_mut();
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
                Op::AddBroadcast(a, b) => {
                    a.0.borrow_mut().grad.add_assign(&grad);
                    
                    let cols = b.0.borrow().data.length();
                    let rows = grad.length() / cols;
                    inner.data.add_broadcast_bw_b(&mut b.0.borrow_mut().grad, rows, cols);
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
