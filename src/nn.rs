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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;
    use crate::backend::Device;

    #[test]
    fn linear_forward_shapes() {
        let layer = Linear::new_on(3, 2, Device::Cpu);
        let x = Tensor::new_on(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], Device::Cpu);
        let y = layer.forward(&x);
        assert_eq!(y.shape(), vec![2, 2]);
    }

    #[test]
    fn linear_backward_grad_shapes() {
        let layer = Linear::new_on(3, 2, Device::Cpu);
        let x = Tensor::new_on(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], Device::Cpu);
        let y = layer.forward(&x).sum();
        y.backward();
        assert_eq!(layer.weight.grad().len(), 6);
        assert_eq!(layer.bias.grad().len(), 2);
        assert_eq!(x.grad().len(), 6);
    }

}
