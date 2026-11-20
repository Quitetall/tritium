//! Actual raw adapters, private controller bindings and per-read rejection.
use super::*;
use std::{borrow::Cow, cell::RefCell, rc::Rc};
use tritium_quantize::engine::ObservedWeightSource;

struct SharedSource(Rc<RefCell<InMemoryWeightSource>>);
impl WeightSource for SharedSource {
    fn inventory(&self) -> Result<Vec<QuantSourceTensor>, QuantEngineError> {
        self.0.borrow().inventory()
    }
    fn asset_names(&self) -> Vec<String> {
        self.0.borrow().asset_names()
    }
    fn tensor(&mut self, name: &str, max_bytes: usize) -> Result<Cow<'_, [u8]>, QuantEngineError> {
        Ok(Cow::Owned(
            self.0.borrow_mut().tensor(name, max_bytes)?.into_owned(),
        ))
    }
    fn asset(&mut self, name: &str, max_bytes: usize) -> Result<Cow<'_, [u8]>, QuantEngineError> {
        Ok(Cow::Owned(
            self.0.borrow_mut().asset(name, max_bytes)?.into_owned(),
        ))
    }
    fn input_gram(
        &mut self,
        name: &str,
        max_entries: usize,
    ) -> Result<Cow<'_, [f64]>, QuantEngineError> {
        Ok(Cow::Owned(
            self.0
                .borrow_mut()
                .input_gram(name, max_entries)?
                .into_owned(),
        ))
    }
}

#[test]
fn observed_source_checks_real_memory_and_seekable_shards_without_changing_identity() {
    for precision in [
        ScalePrecision::F32,
        ScalePrecision::F16,
        ScalePrecision::Bf16,
    ] {
        let mut input = source(precision);
        let request = plan(&input);
        let expected = Engine::observe_source(&request, &mut input).unwrap();
        let payloads = input.clone();
        let norm = BTreeMap::from([("norm".into(), input.tensors["norm"].clone())]);
        let weight = BTreeMap::from([("weight".into(), input.tensors["weight"].clone())]);
        let mut shards = SafeTensorsWeightSource::new(
            [Cursor::new(shard(&norm)), Cursor::new(shard(&weight))],
            input.assets.clone(),
            BTreeMap::new(),
        )
        .unwrap();
        for actual in [&mut input as &mut dyn WeightSource, &mut shards] {
            let mut guarded = ObservedWeightSource::new(&request, actual).unwrap();
            assert_eq!(guarded.source_id(), expected);
            assert_eq!(guarded.architecture(), "fixture");
            assert_eq!(guarded.tensor_bindings().len(), 2);
            assert_eq!(guarded.asset_bindings().len(), 2);
            for binding in guarded.tensor_bindings().to_vec() {
                let name = binding.source.name;
                let bytes = &payloads.tensors[&name].bytes;
                assert!(guarded.tensor(&name, bytes.len() - 1).is_err());
                assert_eq!(guarded.tensor(&name, bytes.len()).unwrap().as_ref(), bytes);
            }
            for asset in guarded.asset_bindings().to_vec() {
                let bytes = &payloads.assets[&asset.name];
                assert!(guarded.asset(&asset.name, bytes.len() - 1).is_err());
                assert_eq!(
                    guarded.asset(&asset.name, bytes.len()).unwrap().as_ref(),
                    bytes
                );
            }
            assert!(guarded.tensor("absent", 100).is_err());
            assert!(guarded.asset("absent", 100).is_err());
            assert!(guarded.input_gram("weight", 100).is_err());
            guarded.revalidate().unwrap();
            let mut sink = MemoryTensorSink::default();
            let fitted = Engine::run_with_source_guard(
                &request,
                &mut guarded,
                &mut sink,
                recorder(),
                expected,
            )
            .unwrap();
            assert_eq!(fitted.source_id, expected);
            assert!(sink.artifact().is_some());
        }
    }
}

#[test]
fn observed_source_rejects_mutation_including_signed_zero_assets_and_inventory() {
    for change in 0..8 {
        let state = Rc::new(RefCell::new(source(ScalePrecision::F32)));
        let mut input = SharedSource(state.clone());
        let request = plan(&input);
        let mut guarded = ObservedWeightSource::new(&request, &mut input).unwrap();
        let identity = guarded.source_id();
        {
            let mut input = state.borrow_mut();
            match change {
                0 => input.tensors.get_mut("weight").unwrap().bytes[..4]
                    .copy_from_slice(&0.25f32.to_le_bytes()),
                1 => input.tensors.get_mut("norm").unwrap().bytes[..4]
                    .copy_from_slice(&0.0f32.to_le_bytes()),
                2 => {
                    input.assets.insert("config.json".into(), b"{ }".to_vec());
                }
                3 => {
                    input.tensors.get_mut("norm").unwrap().shape = vec![1, 2];
                }
                4 => {
                    input.tensors.remove("norm");
                }
                5 => {
                    let extra = input.tensors["norm"].clone();
                    input.tensors.insert("extra".into(), extra);
                }
                6 => {
                    input.assets.insert("extra.json".into(), b"{}".to_vec());
                }
                _ => {
                    input.tensors.get_mut("norm").unwrap().precision = ScalePrecision::F16;
                }
            }
        }
        match change {
            0 => assert!(guarded.tensor("weight", 128).is_err()),
            1 | 4 => assert!(guarded.tensor("norm", 8).is_err()),
            2 => assert!(guarded.asset("config.json", 100).is_err()),
            _ => assert!(guarded.inventory().is_err()),
        }
        assert!(guarded.revalidate().is_err(), "mutation {change}");
        assert_eq!(guarded.source_id(), identity); // Frozen identity is not current-file certification.
    }
}

#[test]
fn observed_source_does_not_claim_gram_capture_or_substitute_missing_metrics() {
    let mut input = source(ScalePrecision::F32);
    input.grams.insert("weight".into(), vec![1.0, 2.0]);
    let request = plan(&input);
    let mut guarded = ObservedWeightSource::new(&request, &mut input).unwrap();
    assert!(guarded.input_gram("weight", 1).is_err());
    assert_eq!(
        guarded.input_gram("weight", 2).unwrap().as_ref(),
        &[1.0, 2.0]
    );
    assert!(guarded.input_gram("norm", 100).is_err());
    assert!(guarded.input_gram("absent", 100).is_err());
}
