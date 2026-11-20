//! Integer-only numerical substrate. Packed codes are offset balanced-ternary integers.
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct Rng(pub(crate) u64);
impl Rng {
    pub(crate) fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    pub(crate) fn below(&mut self, n: u64) -> u64 {
        assert!(n > 0);
        // Rejection avoids modulo bias, including for reservoir sampling.
        let cutoff = u64::MAX - u64::MAX % n;
        loop {
            let x = self.next();
            if x < cutoff {
                return x % n;
            }
        }
    }
}

pub(crate) fn rounded_div(x: i128, divisor: i128, rng: &mut Rng) -> i128 {
    assert!(divisor > 0 && divisor <= i128::from(u64::MAX));
    let q = x.div_euclid(divisor);
    let r = x.rem_euclid(divisor);
    q + i128::from(i128::from(rng.below(divisor as u64)) < r)
}

pub(crate) fn limit(digits: u8) -> i64 {
    if digits == 0 {
        i64::from(i32::MAX)
    } else {
        (3_i64.pow(u32::from(digits)) - 1) / 2
    }
}
pub(crate) fn bits(digits: u8) -> usize {
    (64 - (2 * limit(digits)).leading_zeros()) as usize
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct Bank {
    pub(crate) digits: u8,
    pub(crate) exponent: u8,
    pub(crate) len: usize,
    pub(crate) packed: Vec<u8>,
    pub(crate) rescalings: u64,
    pub(crate) clipped: u64,
}
impl Bank {
    pub(crate) fn new(len: usize, digits: u8) -> Self {
        assert!([0, 8, 16, 24].contains(&digits));
        let mut result = Self {
            digits,
            exponent: 0,
            len,
            packed: vec![0; (len * bits(digits)).div_ceil(8)],
            rescalings: 0,
            clipped: 0,
        };
        for i in 0..len {
            result.set(i, 0);
        }
        result
    }
    pub(crate) fn get(&self, index: usize) -> i64 {
        assert!(index < self.len);
        let width = bits(self.digits);
        let mut code = 0_u64;
        for k in 0..width {
            let offset = index * width + k;
            code |= u64::from((self.packed[offset / 8] >> (offset % 8)) & 1) << k;
        }
        code as i64 - limit(self.digits)
    }
    pub(crate) fn set(&mut self, index: usize, value: i64) {
        assert!(index < self.len && value.abs() <= limit(self.digits));
        let code = (value + limit(self.digits)) as u64;
        for k in 0..bits(self.digits) {
            let offset = index * bits(self.digits) + k;
            let mask = 1 << (offset % 8);
            self.packed[offset / 8] =
                (self.packed[offset / 8] & !mask) | ((((code >> k) & 1) as u8) << (offset % 8));
        }
    }
    pub(crate) fn value(&self, index: usize) -> i128 {
        i128::from(self.get(index)) << self.exponent
    }
    pub(crate) fn resize(&mut self, digits: u8) {
        let mut next = Self::new(self.len, digits);
        next.exponent = self.exponent;
        next.rescalings = self.rescalings;
        next.clipped = self.clipped;
        for i in 0..self.len {
            next.set(i, self.get(i));
        }
        *self = next;
    }
    pub(crate) fn coarsen(&mut self, rng: &mut Rng) {
        assert!(
            self.exponent < 60,
            "evidence exceeds reference arithmetic range"
        );
        for i in 0..self.len {
            self.set(i, rounded_div(i128::from(self.get(i)), 2, rng) as i64);
        }
        self.exponent += 1;
        self.rescalings += 1;
    }
    pub(crate) fn ema(&mut self, index: usize, sample: i128, rate: i128, rng: &mut Rng) {
        let scaled = rounded_div(sample, 1_i128 << self.exponent, rng);
        let old = i128::from(self.get(index));
        let next = old + rounded_div(scaled - old, rate, rng);
        let bound = i128::from(limit(self.digits));
        if next.abs() > bound {
            self.clipped += 1;
        }
        self.set(index, next.clamp(-bound, bound) as i64);
    }
    pub(crate) fn validate(&self) -> Result<(), String> {
        if ![0, 8, 16, 24].contains(&self.digits)
            || self.exponent > 60
            || self.packed.len() != (self.len * bits(self.digits)).div_ceil(8)
        {
            return Err("invalid accumulator representation".into());
        }
        for i in 0..self.len {
            if self.get(i).abs() > limit(self.digits) {
                return Err("invalid ternary code".into());
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn packed_signed_carries_and_growth() {
        for width in [8, 16, 24] {
            let mut b = Bank::new(7, width);
            for (i, x) in [-limit(width), -4, -1, 0, 1, 4, limit(width)]
                .into_iter()
                .enumerate()
            {
                b.set(i, x);
                assert_eq!(b.get(i), x);
            }
            b.validate().unwrap();
        }
        let mut b = Bank::new(2, 8);
        b.set(0, 123);
        b.resize(16);
        assert_eq!(b.get(0), 123);
    }
    #[test]
    fn consistent_subthreshold_signal_beats_alternating_noise() {
        let mut stable = Bank::new(1, 16);
        let mut noise = Bank::new(1, 16);
        let mut rng = Rng(9);
        for i in 0..512 {
            stable.ema(0, 16, 64, &mut rng);
            noise.ema(0, if i % 2 == 0 { 16 } else { -16 }, 64, &mut rng);
        }
        assert!(stable.get(0) > 10);
        assert!(noise.get(0).abs() < 8);
    }
    #[test]
    fn exact_integer_linear_regression_optimum() {
        // Complete finite search for y=x on {-2,-1,0,1,2}; no derivative or rounding oracle.
        let losses: Vec<i128> = (-1_i128..=1)
            .map(|w| {
                (-2_i128..=2)
                    .map(|x| {
                        let error = w * x - x;
                        error * error
                    })
                    .sum()
            })
            .collect();
        assert_eq!(losses, vec![40, 10, 0]);
    }
    #[test]
    fn fixed_integer_roundtrip() {
        let mut b = Bank::new(3, 0);
        for (i, value) in [-i64::from(i32::MAX), 0, i64::from(i32::MAX)]
            .into_iter()
            .enumerate()
        {
            b.set(i, value);
            assert_eq!(b.get(i), value);
        }
        b.validate().unwrap();
    }
    #[test]
    fn signed_rounding_and_scale() {
        let mut rng = Rng(1);
        let mut sum = 0;
        for _ in 0..10000 {
            sum += rounded_div(-3, 2, &mut rng);
        }
        assert!((-15300..-14700).contains(&sum));
        let mut b = Bank::new(1, 8);
        b.set(0, -100);
        b.coarsen(&mut rng);
        assert_eq!(b.value(0), -100);
    }
}
