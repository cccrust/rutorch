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
// 2. Embedding Layer
// ==========================================
pub struct Embedding {
    pub weight: Tensor, // [vocab, dim]
}

impl Embedding {
    pub fn new(vocab: usize, dim: usize) -> Self {
        Self::new_on(vocab, dim, Device::default())
    }

    pub fn new_on(vocab: usize, dim: usize, device: Device) -> Self {
        Self { weight: Tensor::randn_on(&[vocab, dim], device) }
    }

    /// ids: Vec<usize> with shape [batch, time] (flattened)
    pub fn forward_ids(&self, ids: &[usize], batch: usize, time: usize) -> Tensor {
        let dim = self.weight.shape()[1];
        let w = self.weight.data();
        let mut out = vec![0.0f32; batch * time * dim];
        for b in 0..batch {
            for t in 0..time {
                let id = ids[b * time + t];
                let w_off = id * dim;
                let out_off = (b * time + t) * dim;
                out[out_off..out_off + dim].copy_from_slice(&w[w_off..w_off + dim]);
            }
        }
        let storage = crate::backend::Storage::new(&out, self.weight.device());
        Tensor(std::rc::Rc::new(std::cell::RefCell::new(crate::tensor::TensorInner {
            data: storage,
            grad: crate::backend::Storage::zeros(out.len(), self.weight.device()),
            shape: vec![batch, time, dim],
            op: crate::tensor::Op::Embedding(self.weight.clone(), ids.to_vec(), batch, time),
            device: self.weight.device(),
        })))
    }

    pub fn parameters(&self) -> Vec<Tensor> {
        vec![self.weight.clone()]
    }
}

// ==========================================
// 3. Simple RNN (single layer)
// ==========================================
pub struct RNN {
    pub wx: Tensor,  // [in_dim, hidden]
    pub wh: Tensor,  // [hidden, hidden]
    pub b: Tensor,   // [hidden]
}

impl RNN {
    pub fn new(in_dim: usize, hidden: usize) -> Self {
        Self::new_on(in_dim, hidden, Device::default())
    }

    pub fn new_on(in_dim: usize, hidden: usize, device: Device) -> Self {
        Self {
            wx: Tensor::randn_on(&[in_dim, hidden], device),
            wh: Tensor::randn_on(&[hidden, hidden], device),
            b: Tensor::new_on(&vec![0.0; hidden], &[hidden], device),
        }
    }

    /// x: [batch, time, in_dim] -> returns h: [batch, time, hidden]
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let shape = x.shape();
        let batch = shape[0];
        let time = shape[1];
        let in_dim = shape[2];
        let hidden = self.wh.shape()[0];

        let x2d = x.reshape(&[batch * time, in_dim]);
        let mut h_prev = Tensor::new_on(&vec![0.0; batch * hidden], &[batch, hidden], x.device());
        let mut h_seq = Vec::with_capacity(batch * time * hidden);

        for t in 0..time {
            let mut xt_data = Vec::with_capacity(batch * in_dim);
            for b in 0..batch {
                let idx = b * time + t;
                let off = idx * in_dim;
                xt_data.extend_from_slice(&x2d.data()[off..off + in_dim]);
            }
            let xt = Tensor::new_on(&xt_data, &[batch, in_dim], x.device());
            let h_lin = xt.matmul(&self.wx).add(&h_prev.matmul(&self.wh)).add_broadcast(&self.b);
            let h = h_lin.tanh();
            h_seq.extend_from_slice(&h.data());
            h_prev = h;
        }

        let h_time = Tensor::new_on(&h_seq, &[time, batch, hidden], x.device());
        h_time.permute(&[1, 0, 2])
    }

    /// x: [batch, time, in_dim], h0: [batch, hidden] -> (h_seq, h_last)
    pub fn forward_with_state(&self, x: &Tensor, h0: &Tensor) -> (Tensor, Tensor) {
        let shape = x.shape();
        let batch = shape[0];
        let time = shape[1];
        let in_dim = shape[2];
        let hidden = self.wh.shape()[0];

        let x2d = x.reshape(&[batch * time, in_dim]);
        let mut h_prev = h0.clone();
        let mut h_seq = Vec::with_capacity(batch * time * hidden);

        for t in 0..time {
            let mut xt_data = Vec::with_capacity(batch * in_dim);
            let x2d_data = x2d.data();
            for b in 0..batch {
                let idx = b * time + t;
                let off = idx * in_dim;
                xt_data.extend_from_slice(&x2d_data[off..off + in_dim]);
            }
            let xt = Tensor::new_on(&xt_data, &[batch, in_dim], x.device());
            let h_lin = xt.matmul(&self.wx).add(&h_prev.matmul(&self.wh)).add_broadcast(&self.b);
            let h = h_lin.tanh();
            h_seq.extend_from_slice(&h.data());
            h_prev = h;
        }

        let h_time = Tensor::new_on(&h_seq, &[time, batch, hidden], x.device());
        (h_time.permute(&[1, 0, 2]), h_prev)
    }

    /// Single step: x [batch, in_dim], h_prev [batch, hidden] -> h [batch, hidden]
    pub fn step(&self, x: &Tensor, h_prev: &Tensor) -> Tensor {
        let h_lin = x.matmul(&self.wx).add(&h_prev.matmul(&self.wh)).add_broadcast(&self.b);
        h_lin.tanh()
    }

    pub fn parameters(&self) -> Vec<Tensor> {
        vec![self.wx.clone(), self.wh.clone(), self.b.clone()]
    }
}

// ==========================================
// 4. GRU (single layer)
// ==========================================
pub struct GRU {
    pub wx_r: Tensor, pub wh_r: Tensor, pub b_r: Tensor,
    pub wx_z: Tensor, pub wh_z: Tensor, pub b_z: Tensor,
    pub wx_n: Tensor, pub wh_n: Tensor, pub b_n: Tensor,
}

impl GRU {
    pub fn new(in_dim: usize, hidden: usize) -> Self {
        Self::new_on(in_dim, hidden, Device::default())
    }

    pub fn new_on(in_dim: usize, hidden: usize, device: Device) -> Self {
        Self {
            wx_r: Tensor::randn_on(&[in_dim, hidden], device),
            wh_r: Tensor::randn_on(&[hidden, hidden], device),
            b_r: Tensor::new_on(&vec![0.0; hidden], &[hidden], device),
            wx_z: Tensor::randn_on(&[in_dim, hidden], device),
            wh_z: Tensor::randn_on(&[hidden, hidden], device),
            b_z: Tensor::new_on(&vec![0.0; hidden], &[hidden], device),
            wx_n: Tensor::randn_on(&[in_dim, hidden], device),
            wh_n: Tensor::randn_on(&[hidden, hidden], device),
            b_n: Tensor::new_on(&vec![0.0; hidden], &[hidden], device),
        }
    }

    pub fn forward_with_state(&self, x: &Tensor, h0: &Tensor) -> (Tensor, Tensor) {
        let shape = x.shape();
        let batch = shape[0];
        let time = shape[1];
        let in_dim = shape[2];
        let hidden = self.wh_r.shape()[0];

        let x2d = x.reshape(&[batch * time, in_dim]);
        let mut h_prev = h0.clone();
        let mut h_seq = Vec::with_capacity(batch * time * hidden);

        for t in 0..time {
            let mut xt_data = Vec::with_capacity(batch * in_dim);
            let x2d_data = x2d.data();
            for b in 0..batch {
                let idx = b * time + t;
                let off = idx * in_dim;
                xt_data.extend_from_slice(&x2d_data[off..off + in_dim]);
            }
            let xt = Tensor::new_on(&xt_data, &[batch, in_dim], x.device());

            let r = xt.matmul(&self.wx_r).add(&h_prev.matmul(&self.wh_r)).add_broadcast(&self.b_r).sigmoid();
            let z = xt.matmul(&self.wx_z).add(&h_prev.matmul(&self.wh_z)).add_broadcast(&self.b_z).sigmoid();
            let n = xt.matmul(&self.wx_n).add(&(r.mul(&h_prev)).matmul(&self.wh_n)).add_broadcast(&self.b_n).tanh();
            let h = h_prev.lerp(&n, &z);

            h_seq.extend_from_slice(&h.data());
            h_prev = h;
        }

        let h_time = Tensor::new_on(&h_seq, &[time, batch, hidden], x.device());
        (h_time.permute(&[1, 0, 2]), h_prev)
    }

    pub fn step(&self, x: &Tensor, h_prev: &Tensor) -> Tensor {
        let r = x.matmul(&self.wx_r).add(&h_prev.matmul(&self.wh_r)).add_broadcast(&self.b_r).sigmoid();
        let z = x.matmul(&self.wx_z).add(&h_prev.matmul(&self.wh_z)).add_broadcast(&self.b_z).sigmoid();
        let n = x.matmul(&self.wx_n).add(&(r.mul(&h_prev)).matmul(&self.wh_n)).add_broadcast(&self.b_n).tanh();
        h_prev.lerp(&n, &z)
    }

    pub fn parameters(&self) -> Vec<Tensor> {
        vec![
            self.wx_r.clone(), self.wh_r.clone(), self.b_r.clone(),
            self.wx_z.clone(), self.wh_z.clone(), self.b_z.clone(),
            self.wx_n.clone(), self.wh_n.clone(), self.b_n.clone(),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;
    use crate::backend::Device;

    fn assert_vec_close(a: &[f32], b: &[f32], eps: f32) {
        if a.len() != b.len() { panic!("len mismatch"); }
        for i in 0..a.len() {
            if (a[i] - b[i]).abs() > eps {
                panic!("assert_vec_close failed at {}: {} vs {}", i, a[i], b[i]);
            }
        }
    }

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

    #[test]
    fn embedding_forward_and_backward() {
        let emb = Embedding::new_on(4, 3, Device::Cpu);
        emb.weight.set_data(&[
            0.0, 0.1, 0.2,
            1.0, 1.1, 1.2,
            2.0, 2.1, 2.2,
            3.0, 3.1, 3.2,
        ]);
        let ids = vec![2, 1, 3, 0];
        let out = emb.forward_ids(&ids, 2, 2);
        assert_eq!(out.shape(), vec![2, 2, 3]);
        assert_vec_close(
            &out.data(),
            &[
                2.0, 2.1, 2.2,
                1.0, 1.1, 1.2,
                3.0, 3.1, 3.2,
                0.0, 0.1, 0.2,
            ],
            1e-6,
        );
        let loss = out.sum();
        loss.backward();
        let grad = emb.weight.grad();
        assert_vec_close(
            &grad,
            &[
                1.0, 1.0, 1.0,
                1.0, 1.0, 1.0,
                1.0, 1.0, 1.0,
                1.0, 1.0, 1.0,
            ],
            1e-6,
        );
    }

    #[test]
    fn rnn_forward_matches_step() {
        let rnn = RNN::new_on(2, 3, Device::Cpu);
        rnn.wx.set_data(&[
            0.1, 0.2, 0.3,
            0.4, 0.5, 0.6,
        ]);
        rnn.wh.set_data(&[
            0.7, 0.1, -0.2,
            0.0, 0.3, 0.2,
            -0.1, 0.4, 0.5,
        ]);
        rnn.b.set_data(&[0.01, -0.02, 0.03]);

        let x = Tensor::new_on(&[
            1.0, 2.0,
            3.0, 4.0,
            5.0, 6.0,
            7.0, 8.0,
        ], &[2, 2, 2], Device::Cpu); // batch=2, time=2, in=2

        let h_seq = rnn.forward(&x);
        let h_data = h_seq.data();

        let mut h = Tensor::new_on(&vec![0.0; 2 * 3], &[2, 3], Device::Cpu);
        for t in 0..2 {
            let mut xt_data = Vec::with_capacity(2 * 2);
            let x2d = x.reshape(&[2 * 2, 2]);
            let x2d_data = x2d.data();
            for b in 0..2 {
                let idx = b * 2 + t;
                let off = idx * 2;
                xt_data.extend_from_slice(&x2d_data[off..off + 2]);
            }
            let xt = Tensor::new_on(&xt_data, &[2, 2], Device::Cpu);
            h = rnn.step(&xt, &h);
            let h_t = h.data();
            let mut h_from_seq = Vec::with_capacity(2 * 3);
            for b in 0..2 {
                let base = b * 2 * 3 + t * 3;
                h_from_seq.extend_from_slice(&h_data[base..base + 3]);
            }
            assert_vec_close(&h_from_seq, &h_t, 1e-6);
        }
    }

    #[test]
    fn rnn_forward_matches_manual_tanh() {
        let rnn = RNN::new_on(2, 2, Device::Cpu);
        rnn.wx.set_data(&[
            0.1, 0.2,
            0.3, 0.4,
        ]);
        rnn.wh.set_data(&[
            0.5, -0.1,
            0.2, 0.7,
        ]);
        rnn.b.set_data(&[0.01, -0.02]);

        let x = Tensor::new_on(&[
            1.0, 2.0,
            3.0, 4.0,
        ], &[1, 2, 2], Device::Cpu); // batch=1, time=2, in=2

        let h_seq = rnn.forward(&x);
        let h = h_seq.data();

        let tanh = |v: f32| v.tanh();
        let mut h_prev = [0.0f32, 0.0f32];

        for t in 0..2 {
            let x_t = if t == 0 { [1.0f32, 2.0f32] } else { [3.0f32, 4.0f32] };
            let mut h_t = [0.0f32, 0.0f32];
            for j in 0..2 {
                let wx = x_t[0] * rnn.wx.data()[0 * 2 + j] + x_t[1] * rnn.wx.data()[1 * 2 + j];
                let wh = h_prev[0] * rnn.wh.data()[0 * 2 + j] + h_prev[1] * rnn.wh.data()[1 * 2 + j];
                let z = wx + wh + rnn.b.data()[j];
                h_t[j] = tanh(z);
            }
            let offset = t * 2;
            assert_vec_close(&h[offset..offset + 2], &h_t, 1e-6);
            h_prev = h_t;
        }
    }
}
