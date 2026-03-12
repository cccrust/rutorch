# Rutorch

Small Rust autograd/tensor playground with CPU + Metal backends, plus learning demos (XOR, decision boundary, CharGPT).

## Features

- Autograd on basic tensor ops (CPU + Metal)
- Stable `log_softmax + nll_loss`
- Toy datasets (spiral/blob) with CSV outputs
- CharGPT mini pipeline (RNN/GRU or minimal GPT)

## Quick Start

```bash
cargo run --release -- xor
```

## Demos

### XOR

```bash
cargo run --release -- xor
```

### Spiral / Blob Decision Boundary

```bash
cargo run --release -- spiral
cargo run --release -- blob
```

Outputs:

- `out/spiral_points.csv`
- `out/spiral_grid.csv`
- `out/blob_points.csv`
- `out/blob_grid.csv`

Python plotting (optional):

```bash
python py/spiral_draw.py
python py/blob_draw.py
```

### CharGPT (RNN/GRU)

Train on `data/exp.txt` (math expressions):

```bash
cargo run --release -- char cpu rnn 80
cargo run --release -- char cpu gru 80
```

GPU on macOS:

```bash
cargo run --release -- char gpu rnn 80
cargo run --release -- char gpu gru 80
```

Arguments:

- device: `cpu` | `gpu`
- model: `rnn` | `gru` | `gpt`
- epochs: optional integer (default 80)

### CharGPT (GPT)

Minimal single-head causal attention:

```bash
cargo run --release -- char cpu gpt 20
```

## Data

`data/exp.txt` contains 1000 recursive math expressions.  
You can regenerate it if needed.

## Tests

```bash
cargo test -q
```

## Notes

- GPU speedups only help when compute is large and CPU↔GPU transfers are minimized.
- Current RNN/GRU training uses contiguous batch slicing and gradient clipping.

