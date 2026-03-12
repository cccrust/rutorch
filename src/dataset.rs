use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Normal, Distribution};

use crate::tensor::Tensor;
use crate::backend::Device;

#[derive(Clone)]
pub struct Dataset {
    pub x: Vec<f32>,       // flat [n, dim]
    pub y: Vec<usize>,     // class index per sample
    pub n: usize,
    pub dim: usize,
    pub classes: usize,
}

impl Dataset {
    pub fn one_hot_batch(&self, idxs: &[usize], device: Device) -> (Tensor, Tensor) {
        let mut xb = Vec::with_capacity(idxs.len() * self.dim);
        let mut yb = vec![0.0f32; idxs.len() * self.classes];
        for (i, &idx) in idxs.iter().enumerate() {
            let off = idx * self.dim;
            xb.extend_from_slice(&self.x[off..off + self.dim]);
            let cls = self.y[idx];
            yb[i * self.classes + cls] = 1.0;
        }
        let x = Tensor::new_on(&xb, &[idxs.len(), self.dim], device);
        let y = Tensor::new_on(&yb, &[idxs.len(), self.classes], device);
        (x, y)
    }
}

pub struct DataLoader {
    dataset: Dataset,
    batch_size: usize,
    shuffle: bool,
    indices: Vec<usize>,
    pos: usize,
    rng: StdRng,
}

impl DataLoader {
    pub fn new(dataset: Dataset, batch_size: usize, shuffle: bool, seed: u64) -> Self {
        let mut indices: Vec<usize> = (0..dataset.n).collect();
        let mut rng = StdRng::seed_from_u64(seed);
        if shuffle {
            indices.shuffle(&mut rng);
        }
        Self { dataset, batch_size, shuffle, indices, pos: 0, rng }
    }

    pub fn reset(&mut self) {
        self.pos = 0;
        if self.shuffle {
            self.indices.shuffle(&mut self.rng);
        }
    }

    pub fn next_batch(&mut self, device: Device) -> Option<(Tensor, Tensor)> {
        if self.pos >= self.indices.len() {
            return None;
        }
        let end = (self.pos + self.batch_size).min(self.indices.len());
        let idxs = &self.indices[self.pos..end];
        self.pos = end;
        Some(self.dataset.one_hot_batch(idxs, device))
    }
}

pub fn make_spiral(n_per_class: usize, noise: f32, rotations: f32, seed: u64) -> Dataset {
    let mut rng = StdRng::seed_from_u64(seed);
    let normal = Normal::new(0.0, noise).unwrap();
    let mut x = Vec::with_capacity(n_per_class * 2 * 2);
    let mut y = Vec::with_capacity(n_per_class * 2);
    for class in 0..2 {
        for i in 0..n_per_class {
            let r = i as f32 / n_per_class as f32;
            let t = rotations * 2.0 * std::f32::consts::PI * r + class as f32 * std::f32::consts::PI;
            let dx = normal.sample(&mut rng);
            let dy = normal.sample(&mut rng);
            x.push(r * t.cos() + dx);
            x.push(r * t.sin() + dy);
            y.push(class);
        }
    }
    Dataset { x, y, n: n_per_class * 2, dim: 2, classes: 2 }
}

pub fn make_moons(n_per_class: usize, noise: f32, seed: u64) -> Dataset {
    let mut rng = StdRng::seed_from_u64(seed);
    let normal = Normal::new(0.0, noise).unwrap();
    let mut x = Vec::with_capacity(n_per_class * 2 * 2);
    let mut y = Vec::with_capacity(n_per_class * 2);
    for i in 0..n_per_class {
        let t = std::f32::consts::PI * i as f32 / n_per_class as f32;
        let dx = normal.sample(&mut rng);
        let dy = normal.sample(&mut rng);
        x.push(t.cos() + dx);
        x.push(t.sin() + dy);
        y.push(0);
    }
    for i in 0..n_per_class {
        let t = std::f32::consts::PI * i as f32 / n_per_class as f32;
        let dx = normal.sample(&mut rng);
        let dy = normal.sample(&mut rng);
        x.push(1.0 - t.cos() + dx);
        x.push(1.0 - t.sin() - 0.5 + dy);
        y.push(1);
    }
    Dataset { x, y, n: n_per_class * 2, dim: 2, classes: 2 }
}

pub fn make_circles(n_per_class: usize, noise: f32, factor: f32, seed: u64) -> Dataset {
    let mut rng = StdRng::seed_from_u64(seed);
    let normal = Normal::new(0.0, noise).unwrap();
    let mut x = Vec::with_capacity(n_per_class * 2 * 2);
    let mut y = Vec::with_capacity(n_per_class * 2);
    for i in 0..n_per_class {
        let t = 2.0 * std::f32::consts::PI * i as f32 / n_per_class as f32;
        let dx = normal.sample(&mut rng);
        let dy = normal.sample(&mut rng);
        x.push(t.cos() + dx);
        x.push(t.sin() + dy);
        y.push(0);
    }
    for i in 0..n_per_class {
        let t = 2.0 * std::f32::consts::PI * i as f32 / n_per_class as f32;
        let dx = normal.sample(&mut rng);
        let dy = normal.sample(&mut rng);
        x.push(factor * t.cos() + dx);
        x.push(factor * t.sin() + dy);
        y.push(1);
    }
    Dataset { x, y, n: n_per_class * 2, dim: 2, classes: 2 }
}

pub fn make_blobs(n_per_class: usize, noise: f32, centers: &[(f32, f32)], seed: u64) -> Dataset {
    let mut rng = StdRng::seed_from_u64(seed);
    let normal = Normal::new(0.0, noise).unwrap();
    let classes = centers.len();
    let mut x = Vec::with_capacity(n_per_class * classes * 2);
    let mut y = Vec::with_capacity(n_per_class * classes);
    for (cls, (cx, cy)) in centers.iter().enumerate() {
        for _ in 0..n_per_class {
            let dx = normal.sample(&mut rng);
            let dy = normal.sample(&mut rng);
            x.push(cx + dx);
            x.push(cy + dy);
            y.push(cls);
        }
    }
    Dataset { x, y, n: n_per_class * classes, dim: 2, classes }
}
