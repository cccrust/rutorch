# Rutorch

Small Rust autograd/tensor playground with CPU + Metal backends.

## 教學範例：數值穩定的 `log_softmax + nll_loss`

`softmax -> log -> cross-entropy` 在 logits 很大或很小時，容易發生溢位或 `log(0)`。
穩定作法是 `log_softmax + nll_loss`，利用 `log-sum-exp` 技巧避免數值爆炸。

在 `src/main.rs` 內含兩段示範：

1. `gradcheck_log_softmax`：數值微分比對 autograd
2. `demo_stability`：不穩定寫法 vs 穩定寫法的 loss 對照

執行方式：

```bash
cargo run
```

你會看到類似輸出：

```text
=== gradcheck: log_softmax on Cpu ===
max |grad_auto - grad_num| = 0.0000xx

=== stability demo on Cpu ===
naive (softmax->log): ...
stable (log_softmax): ...
```

## 研究/教學方向建議

- 加入更多 `gradcheck` 範例（matmul、relu、broadcast）
- 讓 demo 能輸出 CSV/圖片，用於講解決策邊界或梯度行為
