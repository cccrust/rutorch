# cargo run --release -- xor
# cargo run --release -- spiral
# RUST_BACKTRACE=1 cargo run --release -- char cpu rnn 40
# RUST_BACKTRACE=1 cargo run --release -- char cpu gru 20
RUST_BACKTRACE=1 cargo run --release -- char cpu gpt 20
# cargo run --release -- blob
# python py/blob_draw.py
# cargo run --release -- all
