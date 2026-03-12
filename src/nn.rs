use crate::tensor::Tensor;
use crate::backend::Device;

// ==========================================
// 1. 全連接層 (Linear Layer)
// ==========================================
pub struct Linear {
    pub weight: Tensor,
    pub bias: Tensor,
}

impl Linear {
    /// 建立一個包含隨機權重與全零偏差值的 Linear 層 (預設裝置)
    pub fn new(in_features: usize, out_features: usize) -> Self {
        Self::new_on(in_features, out_features, Device::default())
    }

    /// 在指定裝置建立 Linear 層
    pub fn new_on(in_features: usize, out_features: usize, device: Device) -> Self {
        Self {
            weight: Tensor::randn_on(&[in_features, out_features], device),
            bias: Tensor::new_on(&vec![0.0; out_features], &[out_features], device),
        }
    }

    /// 正向傳播： Y = X @ W + b
    pub fn forward(&self, x: &Tensor) -> Tensor {
        x.matmul(&self.weight).add_broadcast(&self.bias)
    }

    /// 取得這一層的所有可訓練參數 (為了交給優化器)
    pub fn parameters(&self) -> Vec<Tensor> {
        vec![self.weight.clone(), self.bias.clone()]
    }
}

// ==========================================
// 2. 隨機梯度下降優化器 (SGD Optimizer)
// ==========================================
pub struct SGD {
    params: Vec<Tensor>,
    lr: f32,
}

impl SGD {
    /// 將所有需要訓練的參數收集起來
    pub fn new(params: Vec<Tensor>, lr: f32) -> Self {
        Self { params, lr }
    }

    /// 一鍵清空所有參數的梯度
    pub fn zero_grad(&self) {
        for p in &self.params {
            p.zero_grad();
        }
    }

    /// 一鍵更新所有參數
    pub fn step(&self) {
        for p in &self.params {
            p.step(self.lr);
        }
    }
}