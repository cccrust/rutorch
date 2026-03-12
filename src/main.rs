mod backend;
mod tensor;
mod nn;

use backend::Device;
use tensor::Tensor;
use nn::{Linear, SGD};

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
        
        let probs = logits.softmax();
        let loss = probs.cross_entropy(&y_true);

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
        train_xor(Device::MacMetal);
    }

    train_xor(Device::Cpu);

    // train_xor(Device::Cuda); // 目前是佔位符
}