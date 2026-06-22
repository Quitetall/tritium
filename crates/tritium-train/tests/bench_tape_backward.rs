//! Microbenchmark for tape backward pass. Regression gate for the VJP refactor.

use tritium_train::ops::ste;
use tritium_train::tape::Tape;

fn build_tape(n_layers: usize, m: usize, n: usize, k: usize) -> (Tape, usize) {
    let mut tape = Tape::new();
    let wf: Vec<f32> = (0..n * k).map(|i| ((i as f32) * 0.1 - 0.5).sin()).collect();
    let sq = ste::absmean_scale_per_row(&wf, n, k);
    let act_: Vec<f32> = (0..m * k).map(|i| ((i as f32) * 0.07).cos()).collect();
    let s: Vec<f32> = (0..n).map(|i| 0.5 + (i as f32) * 0.01).collect();
    let b: Vec<f32> = vec![0.01; n];
    let target: Vec<f32> = (0..m * n)
        .map(|i| ((i as f32) * 0.03).sin() * 0.5)
        .collect();
    let wf_id = tape.leaf(wf);
    let sq_id = tape.leaf(sq);
    let act_id = tape.leaf(act_);
    let s_id = tape.leaf(s);
    let b_id = tape.leaf(b);
    let tg_id = tape.leaf(target);
    let mut acc_loss = None;
    for _ in 0..n_layers {
        let t = tape.ste_surrogate(wf_id, sq_id, n, k);
        let y = tape.matmul(act_id, t, s_id, m, n, k);
        let z = tape.bias(y, b_id, m, n);
        let h = tape.relu2(z);
        let g = tape.add(h, h);
        let l = tape.mse(g, tg_id);
        acc_loss = Some(match acc_loss {
            None => l,
            Some(prev) => tape.add(prev, l),
        });
    }
    (tape, acc_loss.expect("at least one layer"))
}

#[test]
fn bench_tape_backward() {
    let n_layers = 30;
    let (m, n, k) = (1, 20, 128);
    let iters = 200;
    {
        let (tape, loss_id) = build_tape(n_layers, m, n, k);
        let _ = tape.backward(loss_id);
    }
    let mut times = Vec::with_capacity(iters);
    for _ in 0..iters {
        let (tape, loss_id) = build_tape(n_layers, m, n, k);
        let start = std::time::Instant::now();
        let grads = tape.backward(loss_id);
        times.push(start.elapsed());
        std::hint::black_box(&grads);
    }
    times.sort();
    let median = times[iters / 2];
    let p10 = times[iters / 10];
    let p90 = times[iters * 9 / 10];
    let mean: std::time::Duration = times.iter().sum::<std::time::Duration>() / iters as u32;
    println!("\n=== tape backward bench ({n_layers} layers, {iters} iters) ===");
    println!("  nodes per tape: ~{}", n_layers * 9 + 1);
    println!("  mean:   {mean:?}");
    println!("  p10:    {p10:?}");
    println!("  median: {median:?}");
    println!("  p90:    {p90:?}");
}
