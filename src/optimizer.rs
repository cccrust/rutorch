use crate::tensor::Tensor;

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

// ==========================================
// 3. Adam Optimizer
// ==========================================
pub struct Adam {
    params: Vec<Tensor>,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    m: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
    t: usize,
}

impl Adam {
    pub fn new(params: Vec<Tensor>, lr: f32, beta1: f32, beta2: f32, eps: f32) -> Self {
        let mut m = Vec::with_capacity(params.len());
        let mut v = Vec::with_capacity(params.len());
        for p in &params {
            let len = p.data().len();
            m.push(vec![0.0; len]);
            v.push(vec![0.0; len]);
        }
        Self { params, lr, beta1, beta2, eps, m, v, t: 0 }
    }

    pub fn zero_grad(&self) {
        for p in &self.params {
            p.zero_grad();
        }
    }

    pub fn step(&mut self) {
        self.t += 1;
        let t = self.t as f32;
        for (i, p) in self.params.iter().enumerate() {
            let grad = p.grad();
            let mut data = p.data();
            for j in 0..data.len() {
                self.m[i][j] = self.beta1 * self.m[i][j] + (1.0 - self.beta1) * grad[j];
                self.v[i][j] = self.beta2 * self.v[i][j] + (1.0 - self.beta2) * grad[j] * grad[j];
                let m_hat = self.m[i][j] / (1.0 - self.beta1.powf(t));
                let v_hat = self.v[i][j] / (1.0 - self.beta2.powf(t));
                data[j] -= self.lr * m_hat / (v_hat.sqrt() + self.eps);
            }
            p.set_data(&data);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::Device;
    use crate::nn::Linear;
    use crate::tensor::Tensor;

    fn assert_close(a: f32, b: f32, eps: f32) {
        if (a - b).abs() > eps {
            panic!("assert_close failed: {} vs {}", a, b);
        }
    }

    #[test]
    fn sgd_updates_parameters() {
        let layer = Linear::new_on(2, 1, Device::Cpu);
        let x = Tensor::new_on(&[1.0, 2.0], &[1, 2], Device::Cpu);
        let y = layer.forward(&x).sum();
        y.backward();

        let w_before = layer.weight.data();
        let b_before = layer.bias.data();

        let opt = SGD::new(layer.parameters(), 0.1);
        opt.step();

        let w_after = layer.weight.data();
        let b_after = layer.bias.data();

        let mut changed = false;
        for i in 0..w_before.len() {
            if (w_before[i] - w_after[i]).abs() > 0.0 { changed = true; break; }
        }
        for i in 0..b_before.len() {
            if (b_before[i] - b_after[i]).abs() > 0.0 { changed = true; break; }
        }
        if !changed {
            panic!("SGD did not update parameters");
        }
    }

    #[test]
    fn sgd_zero_grad_clears() {
        let layer = Linear::new_on(2, 1, Device::Cpu);
        let x = Tensor::new_on(&[1.0, 2.0], &[1, 2], Device::Cpu);
        let y = layer.forward(&x).sum();
        y.backward();

        let opt = SGD::new(layer.parameters(), 0.1);
        opt.zero_grad();

        let w_grad = layer.weight.grad();
        let b_grad = layer.bias.grad();
        assert_close(w_grad.iter().sum::<f32>(), 0.0, 1e-6);
        assert_close(b_grad.iter().sum::<f32>(), 0.0, 1e-6);
    }

    #[test]
    fn adam_updates_parameters() {
        let layer = Linear::new_on(2, 1, Device::Cpu);
        let x = Tensor::new_on(&[1.0, 2.0], &[1, 2], Device::Cpu);
        let y = layer.forward(&x).sum();
        y.backward();

        let w_before = layer.weight.data();
        let b_before = layer.bias.data();

        let mut opt = Adam::new(layer.parameters(), 0.01, 0.9, 0.999, 1e-8);
        opt.step();

        let w_after = layer.weight.data();
        let b_after = layer.bias.data();

        let mut changed = false;
        for i in 0..w_before.len() {
            if (w_before[i] - w_after[i]).abs() > 0.0 { changed = true; break; }
        }
        for i in 0..b_before.len() {
            if (b_before[i] - b_after[i]).abs() > 0.0 { changed = true; break; }
        }
        if !changed {
            panic!("Adam did not update parameters");
        }
    }
}
