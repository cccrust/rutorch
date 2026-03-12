use std::collections::HashMap;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand::rngs::StdRng;

#[derive(Debug, Clone)]
pub struct CharVocab {
    stoi: HashMap<char, usize>,
    itos: Vec<char>,
}

impl CharVocab {
    pub fn new_from_text(text: &str) -> Self {
        let mut itos: Vec<char> = text.chars().collect();
        itos.sort();
        itos.dedup();
        let mut stoi = HashMap::new();
        for (i, ch) in itos.iter().enumerate() {
            stoi.insert(*ch, i);
        }
        Self { stoi, itos }
    }

    pub fn len(&self) -> usize { self.itos.len() }

    pub fn encode(&self, text: &str) -> Vec<usize> {
        text.chars().map(|ch| *self.stoi.get(&ch).expect("unknown char")).collect()
    }

    pub fn encode_lossy(&self, text: &str) -> Vec<usize> {
        let fallback = 0usize;
        text.chars()
            .map(|ch| self.stoi.get(&ch).copied().unwrap_or(fallback))
            .collect()
    }

    pub fn decode(&self, ids: &[usize]) -> String {
        let mut s = String::with_capacity(ids.len());
        for &i in ids {
            s.push(self.itos[i]);
        }
        s
    }
}

#[derive(Debug, Clone)]
pub struct CharDataset {
    pub data: Vec<usize>,
    pub block_size: usize,
}

impl CharDataset {
    pub fn new(data: Vec<usize>, block_size: usize) -> Self {
        if block_size < 1 { panic!("block_size must be >= 1"); }
        Self { data, block_size }
    }

    pub fn len(&self) -> usize {
        if self.data.len() <= self.block_size { 0 } else { self.data.len() - self.block_size }
    }

    pub fn get(&self, idx: usize) -> (Vec<usize>, Vec<usize>) {
        let start = idx;
        let end = idx + self.block_size;
        let x = self.data[start..end].to_vec();
        let y = self.data[start + 1..end + 1].to_vec();
        (x, y)
    }
}

pub struct CharDataLoader {
    dataset: CharDataset,
    batch_size: usize,
    shuffle: bool,
    indices: Vec<usize>,
    pos: usize,
    rng: StdRng,
}

impl CharDataLoader {
    pub fn new(dataset: CharDataset, batch_size: usize, shuffle: bool, seed: u64) -> Self {
        let mut indices: Vec<usize> = (0..dataset.len()).collect();
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

    pub fn next_batch(&mut self) -> Option<(Vec<usize>, Vec<usize>, usize, usize)> {
        if self.pos >= self.indices.len() {
            return None;
        }
        let end = (self.pos + self.batch_size).min(self.indices.len());
        let idxs = &self.indices[self.pos..end];
        self.pos = end;
        let time = self.dataset.block_size;
        let batch = idxs.len();
        let mut xb = Vec::with_capacity(batch * time);
        let mut yb = Vec::with_capacity(batch * time);
        for &i in idxs {
            let (x, y) = self.dataset.get(i);
            xb.extend_from_slice(&x);
            yb.extend_from_slice(&y);
        }
        Some((xb, yb, batch, time))
    }
}
