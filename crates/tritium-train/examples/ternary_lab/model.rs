use super::numeric::Rng;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct Example {
    pub(crate) pixels: Vec<u8>,
    pub(crate) label: usize,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct Model {
    pub(crate) inputs: usize,
    pub(crate) hidden: usize,
    pub(crate) outputs: usize,
    pub(crate) input_shift: u8,
    pub(crate) output_shift: u8,
    pub(crate) trits: Vec<i8>,
    pub(crate) version: u64,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct Activation {
    pub(crate) version: u64,
    pub(crate) hidden: Vec<i64>,
    pub(crate) output: Vec<i64>,
}
impl Model {
    pub(crate) fn new(inputs: usize, hidden: usize, outputs: usize, rng: &mut Rng) -> Self {
        Self {
            inputs,
            hidden,
            outputs,
            input_shift: 8,
            output_shift: 3,
            trits: (0..inputs * hidden + hidden * outputs)
                .map(|_| rng.below(3) as i8 - 1)
                .collect(),
            version: 0,
        }
    }
    pub(crate) fn validate(&self) -> Result<(), String> {
        if self.inputs == 0
            || self.hidden == 0
            || self.outputs < 2
            || self.inputs > 784
            || self.hidden > 128
            || self.outputs > 10
            || self.input_shift > 16
            || self.output_shift > 16
            || self.trits.len() != self.inputs * self.hidden + self.hidden * self.outputs
            || self.trits.iter().any(|x| !(-1..=1).contains(x))
        {
            return Err("invalid model".into());
        }
        Ok(())
    }
    pub(crate) fn forward(&self, x: &Example) -> Activation {
        assert_eq!(x.pixels.len(), self.inputs);
        let hidden: Vec<i64> = self.trits[..self.inputs * self.hidden]
            .chunks_exact(self.inputs)
            .map(|row| {
                row.iter()
                    .zip(&x.pixels)
                    .map(|(&w, &v)| i64::from(w) * i64::from(v))
                    .sum::<i64>()
                    / (1_i64 << self.input_shift)
            })
            .map(|v| v.max(0))
            .collect();
        let output = self.trits[self.inputs * self.hidden..]
            .chunks_exact(self.hidden)
            .map(|row| {
                row.iter()
                    .zip(&hidden)
                    .map(|(&w, &v)| i64::from(w) * v)
                    .sum::<i64>()
                    / (1_i64 << self.output_shift)
            })
            .collect();
        Activation {
            version: self.version,
            hidden,
            output,
        }
    }
    pub(crate) fn loss_from(&self, x: &Example, a: &Activation) -> i128 {
        assert_eq!(a.version, self.version, "stale activation");
        assert!(x.label < self.outputs);
        a.output
            .iter()
            .enumerate()
            .map(|(i, &v)| {
                let e = i128::from(v) - if i == x.label { 256 } else { 0 };
                e * e
            })
            .sum()
    }
    pub(crate) fn loss(&self, x: &Example) -> i128 {
        self.loss_from(x, &self.forward(x))
    }
    pub(crate) fn correct(&self, x: &Example) -> bool {
        let a = self.forward(x);
        let mut best = 0;
        for i in 1..self.outputs {
            if a.output[i] > a.output[best] {
                best = i;
            }
        }
        best == x.label
    }
    // Sensitivity of the unrounded linear extension; no derivative of a weight quantizer.
    // The integer activation rounding itself is approximated, not differentiated exactly.
    pub(crate) fn sensitivity(&self, x: &Example, a: &Activation, index: usize) -> i128 {
        assert_eq!(a.version, self.version, "stale activation");
        let split = self.inputs * self.hidden;
        if index >= split {
            let row = (index - split) / self.hidden;
            let col = (index - split) % self.hidden;
            let e = i128::from(a.output[row]) - if row == x.label { 256 } else { 0 };
            2 * e * i128::from(a.hidden[col]) / (1_i128 << self.output_shift)
        } else {
            let row = index / self.inputs;
            if a.hidden[row] == 0 {
                return 0;
            }
            let delta: i128 = (0..self.outputs)
                .map(|k| {
                    let e = i128::from(a.output[k]) - if k == x.label { 256 } else { 0 };
                    2 * e * i128::from(self.trits[split + k * self.hidden + row])
                })
                .sum();
            delta * i128::from(x.pixels[index % self.inputs])
                / (1_i128 << (self.input_shift + self.output_shift))
        }
    }
    pub(crate) fn improvement(
        &mut self,
        examples: &[Example],
        index: usize,
        direction: i8,
    ) -> i128 {
        let before: i128 = examples.iter().map(|x| self.loss(x)).sum();
        let old = self.trits[index];
        assert!((-1..=1).contains(&(old + direction)));
        self.trits[index] += direction;
        // No caches survive this temporary evaluation.
        let after: i128 = examples.iter().map(|x| self.loss(x)).sum();
        self.trits[index] = old;
        before - after
    }
}

pub(crate) fn mnist(
    images: &std::path::Path,
    labels: &std::path::Path,
) -> Result<Vec<Example>, String> {
    let images = std::fs::read(images).map_err(|e| e.to_string())?;
    let labels = std::fs::read(labels).map_err(|e| e.to_string())?;
    let word = |b: &[u8], i| -> Result<usize, String> {
        Ok(
            u32::from_be_bytes(b.get(i..i + 4).ok_or("truncated IDX")?.try_into().unwrap())
                as usize,
        )
    };
    let n = word(&images, 4)?;
    if word(&images, 0)? != 2051
        || word(&labels, 0)? != 2049
        || word(&labels, 4)? != n
        || word(&images, 8)? != 28
        || word(&images, 12)? != 28
        || images.len() != 16 + n * 784
        || labels.len() != 8 + n
    {
        return Err("invalid MNIST IDX dimensions or length".into());
    }
    (0..n)
        .map(|i| {
            let label = usize::from(labels[8 + i]);
            if label >= 10 {
                return Err("invalid MNIST label".into());
            }
            Ok(Example {
                pixels: images[16 + i * 784..16 + (i + 1) * 784].to_vec(),
                label,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn enumerate_exact_candidates() {
        let mut m = Model::new(1, 1, 2, &mut Rng(4));
        m.input_shift = 0;
        m.output_shift = 0;
        let x = Example {
            pixels: vec![3],
            label: 0,
        };
        let mut optimum = i128::MAX;
        for a in -1..=1 {
            for b in -1..=1 {
                for c in -1..=1 {
                    m.trits = vec![a, b, c];
                    let old = m.clone();
                    let before = m.loss(&x);
                    optimum = optimum.min(before);
                    for i in 0..3 {
                        for d in [-1, 1] {
                            if !(-1..=1).contains(&(m.trits[i] + d)) {
                                continue;
                            }
                            let benefit = m.improvement(std::slice::from_ref(&x), i, d);
                            assert_eq!(m, old);
                            let mut candidate = m.clone();
                            candidate.trits[i] += d;
                            assert_eq!(benefit, before - candidate.loss(&x));
                        }
                    }
                }
            }
        }
        assert_eq!(optimum, 253 * 253);
    }
    #[test]
    #[should_panic(expected = "stale activation")]
    fn stale_activations_rejected() {
        let mut m = Model::new(1, 1, 2, &mut Rng(1));
        let x = Example {
            pixels: vec![1],
            label: 0,
        };
        let a = m.forward(&x);
        m.version += 1;
        m.loss_from(&x, &a);
    }
}
