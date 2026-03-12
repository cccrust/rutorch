mod backend;
mod tensor;
mod nn;
mod optimizer;
mod dataset;
mod text;

use backend::Device;
use tensor::Tensor;
use nn::Linear;
use nn::{RNN, GRU};
use optimizer::{SGD, Adam};
use dataset::{DataLoader, make_spiral, make_blobs};
use text::{CharVocab, CharDataset, CharDataLoader};

fn gradcheck_log_softmax(device: Device) {
    println!("\n=== gradcheck: log_softmax on {:?} ===", device);

    let x = Tensor::new_on(&[
        0.5, -1.2, 3.0,
        2.1,  0.7, -0.3,
    ], &[2, 3], device);

    let loss = x.log_softmax().sum();
    loss.backward();
    let grad_auto = x.grad();

    let eps = 1e-3f32;
    let base = x.data();
    let mut grad_num = vec![0.0f32; base.len()];

    for i in 0..base.len() {
        let mut plus = base.clone();
        plus[i] += eps;
        let loss_plus = Tensor::new_on(&plus, &[2, 3], device).log_softmax().sum().data()[0];

        let mut minus = base.clone();
        minus[i] -= eps;
        let loss_minus = Tensor::new_on(&minus, &[2, 3], device).log_softmax().sum().data()[0];

        grad_num[i] = (loss_plus - loss_minus) / (2.0 * eps);
    }

    let mut max_abs_diff = 0.0f32;
    for i in 0..grad_auto.len() {
        let d = (grad_auto[i] - grad_num[i]).abs();
        if d > max_abs_diff { max_abs_diff = d; }
    }

    println!("max |grad_auto - grad_num| = {:.6}", max_abs_diff);
    println!("auto grad sample: {:?}", &grad_auto[0..3]);
    println!("num  grad sample: {:?}", &grad_num[0..3]);
}

fn demo_stability(device: Device) {
    println!("\n=== stability demo on {:?} ===", device);
    let logits = Tensor::new_on(&[
        50.0, 0.0, -50.0,
        100.0, -100.0, 0.0,
    ], &[2, 3], device);
    let y_true = Tensor::new_on(&[
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
    ], &[2, 3], device);

    let naive = logits.softmax().log().mul(&y_true).sum().neg().data()[0];
    let stable = logits.nll_loss(&y_true).data()[0];

    println!("naive (softmax->log): {:.6}", naive);
    println!("stable (log_softmax): {:.6}", stable);

    let sm = logits.softmax().data();
    let lsm = logits.log_softmax().data();
    println!("logits row0: {:?}", &logits.data()[0..3]);
    println!("softmax row0: {:?}", &sm[0..3]);
    println!("log_softmax row0: {:?}", &lsm[0..3]);
}

fn train_xor(device: Device) {
    println!("\n🚀 rutorch 神經網路：使用 {:?} 後端執行中！", device);

    let x = Tensor::new_on(&[
        0.0, 0.0,
        0.0, 1.0,
        1.0, 0.0,
        1.0, 1.0,
    ], &[4, 2], device);

    let y_true = Tensor::new_on(&[
        1.0, 0.0,
        0.0, 1.0,
        0.0, 1.0,
        1.0, 0.0,
    ], &[4, 2], device);

    println!("🏗️ 建構神經網路模型與優化器...");
    
    let layer1 = Linear::new_on(2, 16, device);
    let layer2 = Linear::new_on(16, 2, device);

    let mut all_params = Vec::new();
    all_params.extend(layer1.parameters());
    all_params.extend(layer2.parameters());
    
    let optimizer = SGD::new(all_params, 0.05);

    let epochs = 1500;

    for epoch in 1..=epochs {
        optimizer.zero_grad();

        let hidden = layer1.forward(&x).relu();
        let logits = layer2.forward(&hidden);
        
        let loss = logits.nll_loss(&y_true);

        loss.backward();
        optimizer.step();

        if epoch % 150 == 0 || epoch == 1 {
            println!("Epoch {:04}/{} | Loss: {:.6}", epoch, epochs, loss.data()[0]);
            if epoch == 1 {
                let w1_grad = layer1.weight.grad();
                println!("  L1 W Grad sample: {:?}", &w1_grad[0..4]);
                let w2_grad = layer2.weight.grad();
                println!("  L2 W Grad sample: {:?}", &w2_grad[0..4]);
            }
        }
    }

    println!("✅ 訓練完成！看 XOR 問題是否破解：");
    let preds = layer2.forward(&layer1.forward(&x).relu()).softmax().data();
    
    println!("0 XOR 0 (應為[1, 0]):[{:.4}, {:.4}]", preds[0], preds[1]);
    println!("0 XOR 1 (應為 [0, 1]): [{:.4}, {:.4}]", preds[2], preds[3]);
    println!("1 XOR 0 (應為 [0, 1]):[{:.4}, {:.4}]", preds[4], preds[5]);
    println!("1 XOR 1 (應為[1, 0]): [{:.4}, {:.4}]", preds[6], preds[7]);
}

fn main() {
    println!("=== Rutorch Backend 測試 ===");

    let arg = std::env::args().nth(1).unwrap_or_else(|| "xor".to_string());
    match arg.as_str() {
        "xor" => {
            #[cfg(target_os = "macos")]
            {
                gradcheck_log_softmax(Device::MacMetal);
                demo_stability(Device::MacMetal);
                train_xor(Device::MacMetal);
            }
            gradcheck_log_softmax(Device::Cpu);
            demo_stability(Device::Cpu);
            train_xor(Device::Cpu);
            // train_xor(Device::Cuda); // 目前是佔位符
        }
        "spiral" => {
            demo_spiral_boundary(Device::Cpu);
        }
        "blob" => {
            demo_blob_boundary(Device::Cpu);
        }
        "char" => {
            let device_arg = std::env::args().nth(2).unwrap_or_else(|| "cpu".to_string());
            let model_arg = std::env::args().nth(3).unwrap_or_else(|| "rnn".to_string());
            let epochs_arg = std::env::args().nth(4);
            let device = match device_arg.as_str() {
                "cpu" | "CPU" => Some(Device::Cpu),
                "gpu" | "GPU" => {
                    #[cfg(target_os = "macos")]
                    { Some(Device::MacMetal) }
                    #[cfg(not(target_os = "macos"))]
                    { None }
                }
                _ => None,
            };
            if device.is_none() {
                println!("Usage: cargo run --release -- char [cpu|gpu] [rnn|gru|gpt] [epochs]");
                return;
            }
            let device = device.unwrap();
            let epochs = epochs_arg
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(80);
            match model_arg.as_str() {
                "rnn" | "RNN" => demo_char_pipeline(device, "rnn", epochs),
                "gru" | "GRU" => demo_char_pipeline(device, "gru", epochs),
                "gpt" | "GPT" => demo_char_gpt(device),
                _ => println!("Usage: cargo run --release -- char [cpu|gpu] [rnn|gru|gpt] [epochs]"),
            }
        }
        "all" => {
            #[cfg(target_os = "macos")]
            {
                gradcheck_log_softmax(Device::MacMetal);
                demo_stability(Device::MacMetal);
                train_xor(Device::MacMetal);
            }
            gradcheck_log_softmax(Device::Cpu);
            demo_stability(Device::Cpu);
            train_xor(Device::Cpu);
            demo_spiral_boundary(Device::Cpu);
            demo_blob_boundary(Device::Cpu);
            demo_char_pipeline(Device::Cpu, "rnn", 80);
        }
        _ => {
            println!("Usage: cargo run --release -- [xor|spiral|blob|char|all]");
        }
    }
}

fn demo_char_pipeline(device: Device, kind: &str, epochs: usize) {
    println!("\n=== char pipeline demo on {:?} ===", device);
    let text = std::fs::read_to_string("data/exp.txt").expect("failed to read data/exp.txt");
    let vocab = CharVocab::new_from_text(&text);
    let ids = vocab.encode(&text);
    let block_size = 30;
    let batch_size = 20;
    let cols = ids.len() / batch_size;
    let ids = ids[..batch_size * cols].to_vec();

    let vocab_size = vocab.len();
    let emb_dim = 32;
    let hidden = 128;
    let emb = nn::Embedding::new_on(vocab_size, emb_dim, device);
    let model = if matches!(kind, "gru" | "GRU") {
        CharModel::Gru(GRU::new_on(emb_dim, hidden, device))
    } else {
        CharModel::Rnn(RNN::new_on(emb_dim, hidden, device))
    };
    let proj = nn::Linear::new_on(hidden, vocab_size, device);
    let mut params = Vec::new();
    params.extend(emb.parameters());
    params.extend(model.parameters());
    params.extend(proj.parameters());
    let mut opt = Adam::new(params.clone(), 0.01, 0.9, 0.999, 1e-8);

    let mut h = Tensor::new_on(&vec![0.0; batch_size * hidden], &[batch_size, hidden], device);
    for epoch in 1..=epochs {
        let mut epoch_loss = 0.0f32;
        let mut tokens = 0usize;
        let steps = (cols - 1) / block_size;
        for s in 0..steps {
            let start = s * block_size;
            let end = start + block_size;
            let mut xb = Vec::with_capacity(batch_size * block_size);
            let mut yb = Vec::with_capacity(batch_size * block_size);
            for b in 0..batch_size {
                let base = b * cols;
                xb.extend_from_slice(&ids[base + start..base + end]);
                yb.extend_from_slice(&ids[base + start + 1..base + end + 1]);
            }

            let x_emb = emb.forward_ids(&xb, batch_size, block_size);
            let y_one_hot = one_hot_targets(&yb, vocab_size, device);
            let (h_seq, h_last) = model.forward_with_state(&x_emb, &h);
            let logits = h_seq.reshape(&[batch_size * block_size, hidden]);
            let out = proj.forward(&logits);
            let loss = out.nll_loss(&y_one_hot);

            opt.zero_grad();
            loss.backward();
            clip_grad_norm(&params, 0.5);
            opt.step();

            h = Tensor::new_on(&h_last.data(), &[batch_size, hidden], device);
            epoch_loss += loss.data()[0];
            tokens += batch_size * block_size;
        }
        let avg_loss = if tokens > 0 { epoch_loss / tokens as f32 } else { 0.0 };
        println!("epoch {:03} | loss {:.4}", epoch, avg_loss);
    }

    let prompt = "2+3";
    let generated = generate_text(prompt, 40, &vocab, &emb, &model, &proj, device);
    println!("prompt: {}", prompt);
    println!("sample: {}", generated);
}

fn one_hot_targets(ids: &[usize], vocab: usize, device: Device) -> Tensor {
    let mut data = vec![0.0f32; ids.len() * vocab];
    for (i, &id) in ids.iter().enumerate() {
        data[i * vocab + id] = 1.0;
    }
    Tensor::new_on(&data, &[ids.len(), vocab], device)
}

fn generate_text(prompt: &str, max_new: usize, vocab: &CharVocab, emb: &nn::Embedding, model: &CharModel, proj: &nn::Linear, device: Device) -> String {
    let mut ids = vocab.encode_lossy(prompt);
    let hidden = model.hidden_size();
    let emb_dim = emb.weight.shape()[1];
    let mut h = Tensor::new_on(&vec![0.0; hidden], &[1, hidden], device);

    // Prime with prompt
    for &id in &ids {
        let x = emb.forward_ids(&[id], 1, 1).reshape(&[1, emb_dim]);
        h = model.step(&x, &h);
    }

    for _ in 0..max_new {
        let logits = proj.forward(&h);
        let probs = logits.softmax().data();
        let mut best = 0usize;
        let mut best_val = probs[0];
        for i in 1..probs.len() {
            if probs[i] > best_val {
                best_val = probs[i];
                best = i;
            }
        }
        ids.push(best);
        let x = emb.forward_ids(&[best], 1, 1).reshape(&[1, emb_dim]);
        h = model.step(&x, &h);
    }
    vocab.decode(&ids)
}

enum CharModel {
    Rnn(RNN),
    Gru(GRU),
}

impl CharModel {
    fn parameters(&self) -> Vec<Tensor> {
        match self {
            CharModel::Rnn(r) => r.parameters(),
            CharModel::Gru(g) => g.parameters(),
        }
    }

    fn hidden_size(&self) -> usize {
        match self {
            CharModel::Rnn(r) => r.wh.shape()[0],
            CharModel::Gru(g) => g.wh_r.shape()[0],
        }
    }

    fn forward_with_state(&self, x: &Tensor, h0: &Tensor) -> (Tensor, Tensor) {
        match self {
            CharModel::Rnn(r) => r.forward_with_state(x, h0),
            CharModel::Gru(g) => g.forward_with_state(x, h0),
        }
    }

    fn step(&self, x: &Tensor, h_prev: &Tensor) -> Tensor {
        match self {
            CharModel::Rnn(r) => r.step(x, h_prev),
            CharModel::Gru(g) => g.step(x, h_prev),
        }
    }
}

fn clip_grad_norm(params: &[Tensor], max_norm: f32) {
    let mut total = 0.0f32;
    for p in params {
        let g = p.grad();
        for v in g {
            total += v * v;
        }
    }
    let norm = total.sqrt();
    if norm <= max_norm || norm == 0.0 { return; }
    let scale = max_norm / norm;
    for p in params {
        let mut g = p.grad();
        for v in &mut g {
            *v *= scale;
        }
        p.set_grad(&g);
    }
}

fn demo_char_gpt(device: Device) {
    println!("\n=== char gpt demo on {:?} ===", device);
    let text = std::fs::read_to_string("data/exp.txt").expect("failed to read data/exp.txt");
    let vocab = CharVocab::new_from_text(&text);
    let ids = vocab.encode(&text);
    let block_size = 16;
    let dataset = CharDataset::new(ids, block_size);
    let mut loader = CharDataLoader::new(dataset, 64, true, 123);

    let vocab_size = vocab.len();
    let emb_dim = 32;
    let ff_dim = 64;
    let emb = nn::Embedding::new_on(vocab_size, emb_dim, device);
    let pos_emb = nn::Embedding::new_on(block_size, emb_dim, device);

    let q_proj = nn::Linear::new_on(emb_dim, emb_dim, device);
    let k_proj = nn::Linear::new_on(emb_dim, emb_dim, device);
    let v_proj = nn::Linear::new_on(emb_dim, emb_dim, device);
    let o_proj = nn::Linear::new_on(emb_dim, emb_dim, device);

    let ff1 = nn::Linear::new_on(emb_dim, ff_dim, device);
    let ff2 = nn::Linear::new_on(ff_dim, emb_dim, device);

    let head = nn::Linear::new_on(emb_dim, vocab_size, device);

    let mut params = Vec::new();
    params.extend(emb.parameters());
    params.extend(pos_emb.parameters());
    params.extend(q_proj.parameters());
    params.extend(k_proj.parameters());
    params.extend(v_proj.parameters());
    params.extend(o_proj.parameters());
    params.extend(ff1.parameters());
    params.extend(ff2.parameters());
    params.extend(head.parameters());
    let mut opt = Adam::new(params, 0.01, 0.9, 0.999, 1e-8);

    let epochs = 20;
    for epoch in 1..=epochs {
        loader.reset();
        let mut epoch_loss = 0.0f32;
        let mut tokens = 0usize;
        while let Some((xb, yb, batch, time)) = loader.next_batch() {
            let x_tok = emb.forward_ids(&xb, batch, time);
            let pos_ids = position_ids(batch, time);
            let x_pos = pos_emb.forward_ids(&pos_ids, batch, time);
            let x = x_tok.add(&x_pos);

            let q = q_proj.forward(&x.reshape(&[batch * time, emb_dim])).reshape(&[batch, time, emb_dim]);
            let k = k_proj.forward(&x.reshape(&[batch * time, emb_dim])).reshape(&[batch, time, emb_dim]);
            let v = v_proj.forward(&x.reshape(&[batch * time, emb_dim])).reshape(&[batch, time, emb_dim]);

            let attn = causal_attention(&q, &k, &v, time, device);
            let attn_out = o_proj.forward(&attn.reshape(&[batch * time, emb_dim])).reshape(&[batch, time, emb_dim]);

            let ff = ff2.forward(&ff1.forward(&attn_out.reshape(&[batch * time, emb_dim])).relu());
            let ff_out = ff.reshape(&[batch, time, emb_dim]);

            let logits = head.forward(&ff_out.reshape(&[batch * time, emb_dim]));
            let y_one_hot = one_hot_targets(&yb, vocab_size, device);
            let loss = logits.nll_loss(&y_one_hot);

            opt.zero_grad();
            loss.backward();
            opt.step();

            epoch_loss += loss.data()[0];
            tokens += batch * time;
        }
        let avg_loss = if tokens > 0 { epoch_loss / tokens as f32 } else { 0.0 };
        println!("epoch {:03} | loss {:.4}", epoch, avg_loss);
    }

    let prompt = "2+3";
    let generated = generate_gpt(prompt, 40, &vocab, &emb, &pos_emb, &q_proj, &k_proj, &v_proj, &o_proj, &ff1, &ff2, &head, block_size, device);
    println!("prompt: {}", prompt);
    println!("sample: {}", generated);
}

fn position_ids(batch: usize, time: usize) -> Vec<usize> {
    let mut ids = Vec::with_capacity(batch * time);
    for _ in 0..batch {
        for t in 0..time {
            ids.push(t);
        }
    }
    ids
}

fn causal_mask(time: usize, device: Device) -> Tensor {
    let mut data = vec![0.0f32; time * time];
    for i in 0..time {
        for j in (i + 1)..time {
            data[i * time + j] = -1e9;
        }
    }
    Tensor::new_on(&data, &[time, time], device)
}

fn causal_attention(q: &Tensor, k: &Tensor, v: &Tensor, time: usize, device: Device) -> Tensor {
    let shape = q.shape();
    let batch = shape[0];
    let dim = shape[2];
    let scale = 1.0f32 / (dim as f32).sqrt();
    let mask = causal_mask(time, device);

    let q2d = q.reshape(&[batch * time, dim]);
    let k2d = k.reshape(&[batch * time, dim]);
    let v2d = v.reshape(&[batch * time, dim]);

    let mut out: Option<Tensor> = None;
    for b in 0..batch {
        let q_b = q2d.slice(b * time, (b + 1) * time).reshape(&[time, dim]);
        let k_b = k2d.slice(b * time, (b + 1) * time).reshape(&[time, dim]);
        let v_b = v2d.slice(b * time, (b + 1) * time).reshape(&[time, dim]);
        let scores = q_b.matmul(&k_b.transpose()).mul_scalar(scale).add(&mask);
        let attn = scores.softmax();
        let out_b = attn.matmul(&v_b); // [time, dim]
        out = Some(match out {
            None => out_b,
            Some(acc) => acc.concat(&out_b, 0),
        });
    }
    out.unwrap().reshape(&[batch, time, dim])
}

fn generate_gpt(
    prompt: &str,
    max_new: usize,
    vocab: &CharVocab,
    emb: &nn::Embedding,
    pos_emb: &nn::Embedding,
    q_proj: &nn::Linear,
    k_proj: &nn::Linear,
    v_proj: &nn::Linear,
    o_proj: &nn::Linear,
    ff1: &nn::Linear,
    ff2: &nn::Linear,
    head: &nn::Linear,
    block_size: usize,
    device: Device,
) -> String {
    let mut ids = vocab.encode_lossy(prompt);
    for _ in 0..max_new {
        let start = if ids.len() > block_size { ids.len() - block_size } else { 0 };
        let window = &ids[start..];
        let time = window.len();
        let x_tok = emb.forward_ids(window, 1, time);
        let pos_ids = (0..time).collect::<Vec<_>>();
        let x_pos = pos_emb.forward_ids(&pos_ids, 1, time);
        let x = x_tok.add(&x_pos);

        let q = q_proj.forward(&x.reshape(&[time, emb.weight.shape()[1]])).reshape(&[1, time, emb.weight.shape()[1]]);
        let k = k_proj.forward(&x.reshape(&[time, emb.weight.shape()[1]])).reshape(&[1, time, emb.weight.shape()[1]]);
        let v = v_proj.forward(&x.reshape(&[time, emb.weight.shape()[1]])).reshape(&[1, time, emb.weight.shape()[1]]);

        let attn = causal_attention(&q, &k, &v, time, device);
        let attn_out = o_proj.forward(&attn.reshape(&[time, emb.weight.shape()[1]])).reshape(&[1, time, emb.weight.shape()[1]]);
        let ff = ff2.forward(&ff1.forward(&attn_out.reshape(&[time, emb.weight.shape()[1]])).relu());
        let ff_out = ff.reshape(&[1, time, emb.weight.shape()[1]]);

        let logits = head.forward(&ff_out.reshape(&[time, emb.weight.shape()[1]]));
        let probs = logits.softmax().data();
        let mut best = 0usize;
        let mut best_val = probs[(time - 1) * vocab.len()];
        for i in 0..vocab.len() {
            let v = probs[(time - 1) * vocab.len() + i];
            if v > best_val {
                best_val = v;
                best = i;
            }
        }
        ids.push(best);
    }
    vocab.decode(&ids)
}

fn demo_spiral_boundary(device: Device) {
    use std::fs::{create_dir_all, File};
    use std::io::Write;

    println!("\n=== spiral demo on {:?} ===", device);
    let ds = make_spiral(300, 0.05, 4.0, 42);
    let mut loader = DataLoader::new(ds.clone(), 32, true, 7);

    let layer1 = Linear::new_on(2, 128, device);
    let layer2 = Linear::new_on(128, 128, device);
    let layer3 = Linear::new_on(128, 2, device);
    let mut params = Vec::new();
    params.extend(layer1.parameters());
    params.extend(layer2.parameters());
    params.extend(layer3.parameters());
    let opt = SGD::new(params, 0.03);

    let epochs = 3000;
    for epoch in 1..=epochs {
        loader.reset();
        let mut epoch_loss = 0.0f32;
        let mut batches = 0usize;
        while let Some((x, y)) = loader.next_batch(device) {
            opt.zero_grad();
            let h1 = layer1.forward(&x).relu();
            let h2 = layer2.forward(&h1).relu();
            let logits = layer3.forward(&h2);
            let loss = logits.nll_loss(&y);
            epoch_loss += loss.data()[0];
            batches += 1;
            loss.backward();
            opt.step();
        }
        if epoch % 100 == 0 || epoch == 1 {
            let acc = spiral_accuracy(&ds, device, &layer1, &layer2, &layer3);
            println!("epoch {:04} | loss {:.4} | acc {:.3}", epoch, epoch_loss / batches as f32, acc);
        }
    }

    create_dir_all("out").ok();
    let mut f_points = File::create("out/spiral_points.csv").unwrap();
    writeln!(f_points, "x,y,label").unwrap();
    for i in 0..ds.n {
        let off = i * ds.dim;
        writeln!(f_points, "{},{},{}", ds.x[off], ds.x[off + 1], ds.y[i]).unwrap();
    }

    let mut f_grid = File::create("out/spiral_grid.csv").unwrap();
    writeln!(f_grid, "x,y,p0,p1").unwrap();
    let (min_x, max_x, min_y, max_y) = dataset_bounds(&ds, 0.3);
    let mut grid = Vec::new();
    let mut gx = min_x;
    while gx <= max_x {
        let mut gy = min_y;
        while gy <= max_y {
            grid.push(gx);
            grid.push(gy);
            gy += 0.05;
        }
        gx += 0.05;
    }
    let rows = grid.len() / 2;
    let grid_t = Tensor::new_on(&grid, &[rows, 2], device);
    let probs = layer3.forward(&layer2.forward(&layer1.forward(&grid_t).relu()).relu()).softmax().data();
    for i in 0..rows {
        let x = grid[i * 2];
        let y = grid[i * 2 + 1];
        let p0 = probs[i * 2];
        let p1 = probs[i * 2 + 1];
        writeln!(f_grid, "{},{},{},{}", x, y, p0, p1).unwrap();
    }

    println!("wrote out/spiral_points.csv and out/spiral_grid.csv");
}

fn demo_blob_boundary(device: Device) {
    use std::fs::{create_dir_all, File};
    use std::io::Write;

    println!("\n=== blob demo on {:?} ===", device);
    let centers = [(-1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)];
    let ds = make_blobs(150, 0.25, &centers, 123);
    let mut loader = DataLoader::new(ds.clone(), 32, true, 7);

    let layer1 = Linear::new_on(2, 32, device);
    let layer2 = Linear::new_on(32, ds.classes, device);
    let mut params = Vec::new();
    params.extend(layer1.parameters());
    params.extend(layer2.parameters());
    let mut opt = Adam::new(params, 0.01, 0.9, 0.999, 1e-8);

    let epochs = 400;
    for epoch in 1..=epochs {
        loader.reset();
        let mut epoch_loss = 0.0f32;
        let mut batches = 0usize;
        while let Some((x, y)) = loader.next_batch(device) {
            opt.zero_grad();
            let h1 = layer1.forward(&x).relu();
            let logits = layer2.forward(&h1);
            let loss = logits.nll_loss(&y);
            epoch_loss += loss.data()[0];
            batches += 1;
            loss.backward();
            opt.step();
        }
        if epoch % 100 == 0 || epoch == 1 {
            let acc = blob_accuracy(&ds, device, &layer1, &layer2);
            println!("epoch {:04} | loss {:.4} | acc {:.3}", epoch, epoch_loss / batches as f32, acc);
        }
    }

    create_dir_all("out").ok();
    let mut f_points = File::create("out/blob_points.csv").unwrap();
    writeln!(f_points, "x,y,label").unwrap();
    for i in 0..ds.n {
        let off = i * ds.dim;
        writeln!(f_points, "{},{},{}", ds.x[off], ds.x[off + 1], ds.y[i]).unwrap();
    }

    let mut f_grid = File::create("out/blob_grid.csv").unwrap();
    writeln!(f_grid, "x,y,p0,p1,p2").unwrap();
    let (min_x, max_x, min_y, max_y) = dataset_bounds(&ds, 0.5);
    let mut grid = Vec::new();
    let mut gx = min_x;
    while gx <= max_x {
        let mut gy = min_y;
        while gy <= max_y {
            grid.push(gx);
            grid.push(gy);
            gy += 0.05;
        }
        gx += 0.05;
    }
    let rows = grid.len() / 2;
    let grid_t = Tensor::new_on(&grid, &[rows, 2], device);
    let probs = layer2.forward(&layer1.forward(&grid_t).relu()).softmax().data();
    for i in 0..rows {
        let x = grid[i * 2];
        let y = grid[i * 2 + 1];
        let p0 = probs[i * ds.classes];
        let p1 = probs[i * ds.classes + 1];
        let p2 = probs[i * ds.classes + 2];
        writeln!(f_grid, "{},{},{},{},{}", x, y, p0, p1, p2).unwrap();
    }

    println!("wrote out/blob_points.csv and out/blob_grid.csv");
}

fn dataset_bounds(ds: &dataset::Dataset, pad: f32) -> (f32, f32, f32, f32) {
    let mut min_x = ds.x[0];
    let mut max_x = ds.x[0];
    let mut min_y = ds.x[1];
    let mut max_y = ds.x[1];
    for i in 0..ds.n {
        let off = i * ds.dim;
        let x = ds.x[off];
        let y = ds.x[off + 1];
        if x < min_x { min_x = x; }
        if x > max_x { max_x = x; }
        if y < min_y { min_y = y; }
        if y > max_y { max_y = y; }
    }
    (min_x - pad, max_x + pad, min_y - pad, max_y + pad)
}

fn spiral_accuracy(ds: &dataset::Dataset, device: Device, l1: &Linear, l2: &Linear, l3: &Linear) -> f32 {
    let x = Tensor::new_on(&ds.x, &[ds.n, ds.dim], device);
    let logits = l3.forward(&l2.forward(&l1.forward(&x).relu()).relu());
    let probs = logits.softmax().data();
    let mut correct = 0usize;
    for i in 0..ds.n {
        let p0 = probs[i * 2];
        let p1 = probs[i * 2 + 1];
        let pred = if p1 > p0 { 1 } else { 0 };
        if pred == ds.y[i] { correct += 1; }
    }
    correct as f32 / ds.n as f32
}

fn blob_accuracy(ds: &dataset::Dataset, device: Device, l1: &Linear, l2: &Linear) -> f32 {
    let x = Tensor::new_on(&ds.x, &[ds.n, ds.dim], device);
    let logits = l2.forward(&l1.forward(&x).relu());
    let probs = logits.softmax().data();
    let mut correct = 0usize;
    for i in 0..ds.n {
        let mut best = 0usize;
        let mut best_val = probs[i * ds.classes];
        for c in 1..ds.classes {
            let v = probs[i * ds.classes + c];
            if v > best_val {
                best_val = v;
                best = c;
            }
        }
        if best == ds.y[i] { correct += 1; }
    }
    correct as f32 / ds.n as f32
}
