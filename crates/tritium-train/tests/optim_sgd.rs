//! Portable SGD behavior required by current training manifest.

use tritium_train::{Optimizer, Sgd, checkpoint};

#[test]
fn sgd_step_matches_plain_update_literal() {
    let optimizer = Sgd { lr: 0.1 };
    let mut parameter = [1.0_f32, -2.0];
    let gradient = [0.5_f32, -0.25];
    let mut state = optimizer.init_state(parameter.len());
    optimizer.step(1, &mut parameter, &gradient, &mut state);
    assert_eq!(parameter, [0.95, -1.975]);
}

#[test]
fn sgd_zero_state_checkpoint_roundtrips() {
    let optimizer = Sgd::new(0.01);
    let checkpoint = checkpoint::Checkpoint {
        step: 7,
        leaves: vec![checkpoint::LeafCheckpoint {
            param: vec![1.0, -2.0, 3.0],
            state: optimizer.init_state(3),
        }],
    };
    let bytes = checkpoint::write_checkpoint(&optimizer, &checkpoint);
    assert_eq!(
        checkpoint::read_checkpoint(&optimizer, &bytes),
        Ok(checkpoint)
    );
}
