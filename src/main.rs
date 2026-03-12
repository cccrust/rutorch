mod backend;
mod tensor;
mod nn;
mod optimizer;
mod dataset;

use backend::Device;
use tensor::Tensor;
use nn::Linear;
use optimizer::SGD;
use dataset::{DataLoader, make_spiral, make_blobs};

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
        }
        _ => {
            println!("Usage: cargo run --release -- [xor|spiral|blob|all]");
        }
    }
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
    let opt = SGD::new(params, 0.05);

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
