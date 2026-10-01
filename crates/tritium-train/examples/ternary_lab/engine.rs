//! EAT-O: Evidence-Accumulating Ternary Optimizer.
use super::{
    model::{Activation, Example, Model},
    numeric::{Bank, Rng, bits, limit, rounded_div},
};
use serde::{Deserialize, Serialize};
use std::mem::size_of;

const BLOCK: usize = 128;
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) enum Route {
    Backprop,
    Probe,
}
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) enum History {
    Statistics,
    Replay,
    Both,
}
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) enum Precision {
    Adaptive,
    Fixed8,
    Fixed24,
    Integer32,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct Config {
    pub(crate) route: Route,
    pub(crate) history: History,
    pub(crate) precision: Precision,
    pub(crate) hysteresis: bool,
    pub(crate) simple: bool,
    pub(crate) threshold: i128,
    pub(crate) budget: usize,
    pub(crate) replay_limit: usize,
    pub(crate) coordinates: usize,
}
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct Metrics {
    pub(crate) steps: u64,
    pub(crate) transitions: u64,
    pub(crate) reversals: u64,
    pub(crate) proposals: u64,
    pub(crate) replay_rejections: u64,
    pub(crate) forward_examples: u64,
    pub(crate) sensitivity_coordinates: u64,
    pub(crate) probe_coordinates: u64,
    pub(crate) contraction_terms: u64,
    pub(crate) prediction_absolute_error: i128,
    pub(crate) audited_predictions: u64,
    pub(crate) precision_grows: u64,
    pub(crate) precision_shrinks: u64,
    pub(crate) rounded_small: u64,
    pub(crate) peak_state_bytes: usize,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct Block {
    pub(crate) bank: Bank,
    pub(crate) noise: i128,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct ReplayEntry {
    pub(crate) example: Example,
    pub(crate) activation: Activation,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
/// Accumulates evidence to govern discrete ternary weight transitions.
pub(crate) struct EatOptimizer {
    pub(crate) config: Config,
    pub(crate) model: Model,
    pub(crate) blocks: Vec<Block>,
    pub(crate) observations: Vec<u8>,
    pub(crate) age: Vec<u8>,
    pub(crate) direction: Vec<i8>,
    pub(crate) replay: Vec<ReplayEntry>,
    pub(crate) seen: u64,
    pub(crate) cursor: usize,
    pub(crate) rng: Rng,
    pub(crate) metrics: Metrics,
}
impl EatOptimizer {
    pub(crate) fn new(model: Model, config: Config, seed: u64) -> Result<Self, String> {
        model.validate()?;
        if seed == 0 || config.threshold < 0 || config.coordinates == 0 || config.replay_limit > 256
        {
            return Err("invalid experiment configuration".into());
        }
        let n = model.trits.len();
        let digits = if config.precision == Precision::Integer32 {
            0
        } else if config.precision == Precision::Fixed24 {
            24
        } else {
            8
        };
        let stride = if config.simple { 2 } else { 4 };
        let mut e = Self {
            config,
            model,
            blocks: (0..n.div_ceil(BLOCK))
                .map(|b| Block {
                    bank: Bank::new((n - b * BLOCK).min(BLOCK) * stride, digits),
                    noise: 0,
                })
                .collect(),
            observations: vec![0; n],
            age: vec![255; n],
            direction: vec![0; n],
            replay: Vec::new(),
            seen: 0,
            cursor: 0,
            rng: Rng(seed),
            metrics: Metrics::default(),
        };
        if e.state_bytes() > e.config.budget {
            return Err(format!(
                "state requires {} bytes, budget is {}; raise --budget explicitly",
                e.state_bytes(),
                e.config.budget
            ));
        }
        e.metrics.peak_state_bytes = e.state_bytes();
        Ok(e)
    }
    pub(crate) fn compact(&mut self) {
        self.blocks.shrink_to_fit();
        for block in &mut self.blocks {
            block.bank.packed.shrink_to_fit();
        }
        self.observations.shrink_to_fit();
        self.age.shrink_to_fit();
        self.direction.shrink_to_fit();
        self.replay.shrink_to_fit();
        for x in &mut self.replay {
            x.example.pixels.shrink_to_fit();
            x.activation.hidden.shrink_to_fit();
            x.activation.output.shrink_to_fit();
        }
        self.model.trits.shrink_to_fit();
    }
    pub(crate) fn state_bytes(&self) -> usize {
        // Actual Vec capacities, including metadata/containers; weights and transient working space separate.
        size_of::<Self>() - size_of::<Model>()
            + self.blocks.capacity() * size_of::<Block>()
            + self
                .blocks
                .iter()
                .map(|b| b.bank.packed.capacity())
                .sum::<usize>()
            + self.observations.capacity()
            + self.age.capacity()
            + self.direction.capacity()
            + self.replay.capacity() * size_of::<ReplayEntry>()
            + self
                .replay
                .iter()
                .map(|x| {
                    x.example.pixels.capacity()
                        + 8 * (x.activation.hidden.capacity() + x.activation.output.capacity())
                })
                .sum::<usize>()
    }
    fn remember(&mut self, x: &Example) {
        self.seen += 1;
        if self.config.history == History::Statistics || self.config.replay_limit == 0 {
            return;
        }
        if self.replay.len() < self.config.replay_limit {
            let extra = x.pixels.len()
                + size_of::<ReplayEntry>()
                + 8 * (self.model.hidden + self.model.outputs);
            if self.state_bytes() + extra <= self.config.budget {
                self.replay.reserve_exact(1);
                self.metrics.forward_examples += 1;
                self.metrics.contraction_terms += self.model.trits.len() as u64;
                self.replay.push(ReplayEntry {
                    example: x.clone(),
                    activation: self.model.forward(x),
                });
                assert!(self.state_bytes() <= self.config.budget);
                return;
            }
        }
        let slot = self.rng.below(self.seen) as usize;
        if slot < self.replay.len() {
            self.metrics.forward_examples += 1;
            self.metrics.contraction_terms += self.model.trits.len() as u64;
            self.replay[slot] = ReplayEntry {
                example: x.clone(),
                activation: self.model.forward(x),
            };
        }
    }
    fn replay_improvement(&mut self, index: usize, direction: i8) -> i128 {
        let mut before = 0;
        for entry in &mut self.replay {
            if entry.activation.version != self.model.version {
                entry.activation = self.model.forward(&entry.example);
                self.metrics.forward_examples += 1;
            }
            before += self.model.loss_from(&entry.example, &entry.activation);
        }
        let old = self.model.trits[index];
        self.model.trits[index] += direction;
        let after: i128 = self
            .replay
            .iter()
            .map(|entry| self.model.loss(&entry.example))
            .sum();
        self.metrics.forward_examples += self.replay.len() as u64;
        self.model.trits[index] = old;
        before - after
    }
    fn prepare_bank(&mut self, b: usize, sample: i128) {
        // A sample below four quanta cannot reliably distinguish nearby decisions.
        // Spend spare budget on finer resolution while preserving all represented values.
        let bank = &self.blocks[b].bank;
        if self.config.precision == Precision::Adaptive
            && bank.digits < 24
            && bank.exponent > 0
            && sample != 0
            && sample.abs() < (4_i128 << bank.exponent)
        {
            let extra = (bank.len * bits(bank.digits + 8)).div_ceil(8) - bank.packed.len();
            if extra <= self.config.budget.saturating_sub(self.state_bytes()) {
                let bank = &mut self.blocks[b].bank;
                bank.resize(bank.digits + 8);
                for i in 0..bank.len {
                    let v = bank.get(i);
                    bank.set(i, v * 2);
                }
                bank.exponent -= 1;
                self.metrics.precision_grows += 1;
            }
        }
        loop {
            let bank = &self.blocks[b].bank;
            if (sample >> bank.exponent).abs() <= i128::from(limit(bank.digits)) / 2 {
                break;
            }
            let extra = if bank.digits > 0 && bank.digits < 24 {
                (bank.len * bits(bank.digits + 8)).div_ceil(8) - bank.packed.len()
            } else {
                usize::MAX
            };
            if self.config.precision == Precision::Adaptive
                && bank.digits < 24
                && extra <= self.config.budget.saturating_sub(self.state_bytes())
            {
                let digits = bank.digits + 8;
                self.blocks[b].bank.resize(digits);
                self.metrics.precision_grows += 1;
            } else {
                self.blocks[b].bank.coarsen(&mut self.rng);
            }
        }
    }
    fn eligible(&self, i: usize, dir: i8, fast: i128, slow: i128, noise: i128) -> bool {
        if self.observations[i] < 4 {
            return false;
        }
        let mut threshold = self.config.threshold;
        if self.config.hysteresis {
            if self.age[i] < 2 {
                return false;
            }
            threshold += noise;
            if self.direction[i] == -dir && self.age[i] < 16 {
                threshold = threshold.saturating_mul(2);
            }
        }
        fast > threshold && (self.config.simple || slow > threshold)
    }
    pub(crate) fn step(&mut self, x: &Example) {
        assert_eq!(x.pixels.len(), self.model.inputs);
        assert!(x.label < self.model.outputs);
        // Replay contains only earlier training examples; current example enters after decisions.
        let n = self.model.trits.len();
        let forward_before = self.metrics.forward_examples;
        let stride = if self.config.simple { 2 } else { 4 };
        let count = self.config.coordinates.min(n);
        let mut activation = self.model.forward(x);
        self.metrics.forward_examples += 1;
        for offset in 0..count {
            let i = (self.cursor + offset) % n;
            let b = i / BLOCK;
            let base = (i % BLOCK) * stride;
            self.age[i] = self.age[i].saturating_add(1);
            self.observations[i] = self.observations[i].saturating_add(1);
            let mut selected = None;
            for (slot, dir) in [-1, 1].into_iter().enumerate() {
                if !(-1..=1).contains(&(self.model.trits[i] + dir)) {
                    continue;
                }
                let signal = match self.config.route {
                    Route::Backprop => {
                        self.metrics.sensitivity_coordinates += 1;
                        self.metrics.contraction_terms +=
                            if i < self.model.inputs * self.model.hidden {
                                self.model.outputs as u64 + 1
                            } else {
                                1
                            };
                        -self.model.sensitivity(x, &activation, i) * i128::from(dir)
                    }
                    Route::Probe => {
                        self.metrics.probe_coordinates += 1;
                        self.metrics.forward_examples += 2;
                        self.model.improvement(std::slice::from_ref(x), i, dir)
                    }
                };
                self.prepare_bank(b, signal);
                let block = &mut self.blocks[b];
                if signal != 0 && (signal >> block.bank.exponent) == 0 {
                    self.metrics.rounded_small += 1;
                }
                let entry = base + slot * stride / 2;
                let error = (signal - block.bank.value(entry)).abs();
                block.noise += rounded_div(error - block.noise, 64, &mut self.rng);
                block.bank.ema(entry, signal, 8, &mut self.rng);
                if !self.config.simple {
                    block.bank.ema(entry + 1, signal, 64, &mut self.rng);
                }
                let fast = block.bank.value(entry);
                let slow = if self.config.simple {
                    fast
                } else {
                    block.bank.value(entry + 1)
                };
                let noise = block.noise;
                if self.eligible(i, dir, fast, slow, noise)
                    && selected.is_none_or(|(_, score)| fast > score)
                {
                    selected = Some((dir, fast));
                }
            }
            if let Some((dir, prediction)) = selected {
                self.metrics.proposals += 1;
                // Audit predicted local benefit even in statistics-only mode; never use it to accept there.
                let actual = self.model.improvement(std::slice::from_ref(x), i, dir);
                self.metrics.forward_examples += 2;
                self.metrics.prediction_absolute_error += (prediction - actual).abs();
                self.metrics.audited_predictions += 1;
                if self.config.history != History::Statistics {
                    if self.replay.is_empty() {
                        continue;
                    }
                    let benefit = self.replay_improvement(i, dir);
                    if benefit <= 0 {
                        self.metrics.replay_rejections += 1;
                        continue;
                    }
                }
                if self.direction[i] == -dir {
                    self.metrics.reversals += 1;
                }
                self.direction[i] = dir;
                self.age[i] = 0;
                self.observations[i] = 0;
                self.model.trits[i] += dir;
                self.model.version += 1;
                self.metrics.transitions += 1;
                for j in 0..stride {
                    self.blocks[b].bank.set(base + j, 0);
                }
                // Changes elsewhere invalidate approximate evidence, but not historical observations.
                // Every subsequent sensitivity/probe is evaluated against the new model version.
                activation = self.model.forward(x);
                self.metrics.forward_examples += 1;
            }
            if self.config.history == History::Replay {
                // Replay-only ablation: no cross-example score history; hysteresis metadata survives.
                for j in 0..stride {
                    self.blocks[b].bank.set(base + j, 0);
                }
            }
        }
        self.metrics.contraction_terms +=
            (self.metrics.forward_examples - forward_before) * n as u64;
        self.cursor = (self.cursor + count) % n;
        self.metrics.steps += 1;
        if self.config.precision == Precision::Adaptive && self.metrics.steps.is_multiple_of(64) {
            for block in &mut self.blocks {
                let bank = &mut block.bank;
                if bank.digits > 8
                    && (0..bank.len).all(|i| bank.get(i).abs() <= limit(bank.digits - 8) / 4)
                {
                    bank.resize(bank.digits - 8);
                    self.metrics.precision_shrinks += 1;
                }
                // Exact refinement adds resolution without changing any represented value.
                if bank.exponent > 0
                    && (0..bank.len).all(|i| bank.get(i).abs() <= limit(bank.digits) / 4)
                {
                    for i in 0..bank.len {
                        let v = bank.get(i);
                        bank.set(i, v * 2);
                    }
                    bank.exponent -= 1;
                }
            }
        }
        self.remember(x);
        self.metrics.peak_state_bytes = self.metrics.peak_state_bytes.max(self.state_bytes());
        assert!(
            self.state_bytes() <= self.config.budget,
            "optimizer budget exceeded"
        );
    }
    pub(crate) fn validate(&self) -> Result<(), String> {
        self.model.validate()?;
        let n = self.model.trits.len();
        if self.rng.0 == 0
            || self.cursor >= n
            || self.blocks.len() != n.div_ceil(BLOCK)
            || self.observations.len() != n
            || self.age.len() != n
            || self.direction.len() != n
            || self.direction.iter().any(|d| !(-1..=1).contains(d))
            || self.replay.len() > self.config.replay_limit
            || self.config.replay_limit > 256
            || self.config.threshold < 0
            || self.config.coordinates == 0
            || self.metrics.transitions != self.model.version
            || self.metrics.steps != self.seen
            || self.state_bytes() > self.config.budget
        {
            return Err("invalid optimizer checkpoint".into());
        }
        for (i, b) in self.blocks.iter().enumerate() {
            b.bank.validate()?;
            if b.bank.len != (n - i * BLOCK).min(BLOCK) * if self.config.simple { 2 } else { 4 }
                || b.noise < 0
            {
                return Err("invalid evidence block".into());
            }
        }
        if self.replay.iter().any(|x| {
            x.example.pixels.len() != self.model.inputs
                || x.example.label >= self.model.outputs
                || x.activation.hidden.len() != self.model.hidden
                || x.activation.output.len() != self.model.outputs
                || x.activation.version > self.model.version
                || (x.activation.version == self.model.version
                    && x.activation != self.model.forward(&x.example))
        }) {
            return Err("invalid replay example".into());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    pub(crate) fn fixture() -> EatOptimizer {
        EatOptimizer::new(
            Model::new(2, 2, 2, &mut Rng(1)),
            Config {
                route: Route::Probe,
                history: History::Both,
                precision: Precision::Adaptive,
                hysteresis: true,
                simple: false,
                threshold: 0,
                budget: 65536,
                replay_limit: 4,
                coordinates: 8,
            },
            7,
        )
        .unwrap()
    }
    #[test]
    fn hysteresis_and_zero_recovery() {
        let mut e = fixture();
        e.observations[0] = 8;
        assert!(e.eligible(0, 1, 100, 100, 0));
        e.direction[0] = -1;
        e.age[0] = 1;
        assert!(!e.eligible(0, 1, 100, 100, 0));
        e.age[0] = 3;
        e.config.threshold = 60;
        assert!(!e.eligible(0, 1, 100, 100, 0));
        assert!(e.eligible(0, 1, 200, 200, 0));
    }
    #[test]
    fn checkpoint_resume_identical() {
        let mut a = fixture();
        let x = Example {
            pixels: vec![255, 0],
            label: 0,
        };
        for _ in 0..9 {
            a.step(&x);
        }
        let mut b: EatOptimizer = serde_json::from_slice(&serde_json::to_vec(&a).unwrap()).unwrap();
        b.compact();
        b.validate().unwrap();
        for _ in 0..30 {
            a.step(&x);
            b.step(&x);
        }
        assert_eq!(a, b);
    }
    #[test]
    fn recover_zero_output_from_accumulated_evidence() {
        let mut e = fixture();
        e.model.input_shift = 0;
        e.model.output_shift = 0;
        e.model.trits = vec![1, 0, 0, 1, 0, 0, 0, 0];
        e.config.history = History::Statistics;
        e.config.hysteresis = false;
        let x = Example {
            pixels: vec![255, 0],
            label: 0,
        };
        let before = e.model.loss(&x);
        for _ in 0..20 {
            e.step(&x);
        }
        assert!(e.metrics.transitions > 0);
        assert!(e.model.loss(&x) < before);
    }
    #[test]
    fn checkpoint_with_transitions_and_replay_resumes() {
        let mut a = fixture();
        a.model.input_shift = 0;
        a.model.output_shift = 0;
        a.model.trits = vec![1, 0, 0, 1, 0, 0, 0, 0];
        a.config.hysteresis = false;
        let x = Example {
            pixels: vec![255, 0],
            label: 0,
        };
        for _ in 0..7 {
            a.step(&x);
        }
        assert!(a.metrics.transitions > 0);
        let mut b: EatOptimizer = serde_json::from_slice(&serde_json::to_vec(&a).unwrap()).unwrap();
        b.compact();
        for _ in 0..70 {
            a.step(&x);
            b.step(&x);
        }
        assert_eq!(a, b);
    }
    #[test]
    fn adaptive_and_integer_precision_preserve_range() {
        let mut e = fixture();
        e.prepare_bank(0, 1_i128 << 30);
        assert!(e.metrics.precision_grows > 0);
        e.validate().unwrap();
        e.config.precision = Precision::Integer32;
        e.blocks[0].bank = Bank::new(32, 0);
        e.prepare_bank(0, 1_i128 << 45);
        assert!(e.blocks[0].bank.exponent > 0);
        e.validate().unwrap();
    }
    #[test]
    fn stale_replay_cache_is_recomputed_before_decision() {
        let mut e = fixture();
        let x = Example {
            pixels: vec![255, 0],
            label: 0,
        };
        e.remember(&x);
        e.model.version += 1;
        e.replay[0].activation.output.fill(999999);
        let actual = e.model.improvement(std::slice::from_ref(&x), 0, 1);
        assert_eq!(e.replay_improvement(0, 1), actual);
        assert_eq!(e.replay[0].activation, e.model.forward(&x));
    }
    #[test]
    fn simple_state_is_smaller_and_trainable() {
        let original = fixture();
        let mut cfg = original.config.clone();
        cfg.simple = true;
        let mut small = EatOptimizer::new(original.model.clone(), cfg, 3).unwrap();
        assert!(small.state_bytes() < original.state_bytes());
        small.step(&Example {
            pixels: vec![255, 0],
            label: 0,
        });
        small.validate().unwrap();
    }
    #[test]
    fn malformed_checkpoint_rejected() {
        let mut e = fixture();
        e.blocks[0].bank.packed.fill(255);
        assert!(e.validate().is_err());
        let mut e = fixture();
        e.replay.push(ReplayEntry {
            example: Example {
                pixels: vec![1],
                label: 0,
            },
            activation: Activation {
                version: 0,
                hidden: vec![],
                output: vec![],
            },
        });
        assert!(e.validate().is_err());
    }
    #[test]
    fn budget_and_saturation() {
        let mut e = fixture();
        e.config.budget = e.state_bytes();
        e.prepare_bank(0, 1_i128 << 45);
        assert!(e.blocks[0].bank.exponent > 0);
        assert_eq!(e.metrics.precision_grows, 0);
        assert_eq!(e.blocks[0].bank.clipped, 0);
        e.validate().unwrap();
    }
}
