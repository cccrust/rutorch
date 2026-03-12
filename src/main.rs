mod backend;
mod tensor;
mod nn;

use backend::Device;
use tensor::Tensor;
use nn::{Linear, SGD};

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
