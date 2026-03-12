# Rutorch

Small Rust autograd/tensor playground with CPU + Metal backends.

## 教學範例：數值穩定的 `log_softmax + nll_loss`

`softmax -> log -> cross-entropy` 在 logits 很大或很小時，容易發生溢位或 `log(0)`。
穩定作法是 `log_softmax + nll_loss`，利用 `log-sum-exp` 技巧避免數值爆炸。

簡短推導：

1. `softmax(z)_i = exp(z_i) / sum_j exp(z_j)`
2. `log softmax(z)_i = z_i - log(sum_j exp(z_j))`
3. 為了數值穩定，改寫成  
   `logsumexp(z) = m + log(sum_j exp(z_j - m))`，其中 `m = max_j z_j`
4. 所以 `log softmax(z)_i = z_i - logsumexp(z)`

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

## Toy Dataset 與 Decision Boundary Demo

內建 `spiral/moons/circles` 資料產生與簡易 `DataLoader`，示範如何用 CSV 看 decision boundary。

執行後會輸出：

- `out/spiral_points.csv`：資料點與標籤
- `out/spiral_grid.csv`：網格點與模型預測機率

另提供較簡單的 `blobs` 分群示範：

- `out/blob_points.csv`
- `out/blob_grid.csv`
