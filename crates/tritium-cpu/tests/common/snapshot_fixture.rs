//! Shared complete execution checks for scale and full packed versions.
use tritium_core::{
    AdditiveView, PerTilePlan, Trit, UniformPlan, accumulate_embed_scale_vjp,
    accumulate_embed_weight_vjp, accumulate_ternary_matmul_scale_vjp,
    accumulate_ternary_matmul_weight_vjp, reference_embed, reference_ternary_matmul,
    reference_ternary_matmul_input_vjp,
};
use tritium_cpu::CpuBackend;
use tritium_format::AdditiveTensor;
use tritium_schema::{
    ADMITTED_LAWS, AdditiveLayout, Basis, PlaneAllocation, PlaneCodec, Transport,
};
use tritium_spec::{DeviceBuffer, TensorMatmul, TernaryBackend};

pub(super) fn tensor(
    law: usize,
    codec: PlaneCodec,
    basis: Basis,
    mode: u8,
    group: u16,
    planes: u8,
) -> AdditiveTensor {
    let admitted = ADMITTED_LAWS[law];
    let layout = AdditiveLayout {
        rows: 3,
        cols: if basis == Basis::Identity { 260 } else { 512 },
        tile: 256,
        group,
        max_planes: admitted.max_planes,
        allocation: if mode == 0 {
            PlaneAllocation::Uniform
        } else {
            PlaneAllocation::PerTile
        },
        codec,
        law: admitted.law,
        basis,
        transport: Transport::Raw,
    };
    let counts = if mode == 2 {
        vec![0; 6]
    } else {
        [0, 1, planes, 2, 2, 0].map(|n| n.min(planes)).to_vec()
    };
    let (n, s) = if mode == 0 {
        let p = UniformPlan::new(layout, planes).unwrap();
        (p.trit_count(), p.scale_count())
    } else {
        let p = PerTilePlan::new(layout, &counts).unwrap();
        (p.trit_count(), p.scale_count())
    };
    let trits = (0..n)
        .map(|i| [Trit::ZERO, Trit::POS, Trit::NEG, Trit::POS][i % 4])
        .collect();
    if mode == 0 {
        AdditiveTensor::new(layout, planes, trits, vec![0.5; s]).unwrap()
    } else {
        AdditiveTensor::new_per_tile(layout, counts, trits, vec![0.5; s]).unwrap()
    }
}
fn values(n: usize) -> Vec<f32> {
    (0..n).map(|i| (i % 7) as f32 * 0.125 - 0.375).collect()
}
fn bits32(values: &[f32]) -> Vec<u32> {
    values.iter().map(|v| v.to_bits()).collect()
}
fn bits64(values: &[f64]) -> Vec<u64> {
    values.iter().map(|v| v.to_bits()).collect()
}

pub(super) fn check(handle: &dyn DeviceBuffer, view: AdditiveView<'_>) {
    let cpu = CpuBackend::new();
    let rows = view.layout().rows as usize;
    let cols = view.layout().cols as usize;
    assert!(cpu.additive_equals_view(handle, view).unwrap());
    assert_eq!(
        cpu.additive_stored_scale_count(handle).unwrap(),
        view.scales().len()
    );
    for batch in [0, 2] {
        let x = values(batch * cols);
        let dy = values(batch * rows);
        let ids = if batch == 0 { vec![] } else { vec![2, 0, 2] };
        let de = values(ids.len() * cols);
        let mut actual = vec![19.0; dy.len()];
        let mut expected = actual.clone();
        let mut arow = vec![17.0; x.len()];
        let mut erow = arow.clone();
        cpu.matmul(TensorMatmul {
            act: &x,
            tensor: handle,
            batch,
            transformed_act: &mut arow,
            out: &mut actual,
        })
        .unwrap();
        reference_ternary_matmul(&x, &view.bind_matmul(), batch, &mut erow, &mut expected).unwrap();
        assert_eq!(bits32(&actual), bits32(&expected));
        assert_eq!(bits32(&arow), bits32(&erow));
        let mut actual = vec![19.0; de.len()];
        let mut expected = actual.clone();
        cpu.embed_rows(handle, &ids, &mut actual).unwrap();
        reference_embed(&view.bind_gather(), &ids, &mut expected).unwrap();
        assert_eq!(bits32(&actual), bits32(&expected));
        let mut actual = vec![19.0; x.len()];
        let mut expected = actual.clone();
        let mut arow = vec![17.0; cols];
        let mut erow = arow.clone();
        cpu.additive_matmul_input_vjp(handle, &dy, batch, &mut arow, &mut actual)
            .unwrap();
        reference_ternary_matmul_input_vjp(
            &dy,
            &view.bind_matmul(),
            batch,
            &mut erow,
            &mut expected,
        )
        .unwrap();
        assert_eq!(bits32(&actual), bits32(&expected));
        assert_eq!(bits32(&arow), bits32(&erow));
        let mut actual = vec![0.25; view.scales().len()];
        let mut expected = actual.clone();
        cpu.accumulate_additive_matmul_scale_vjp(handle, &x, &dy, batch, &mut arow, &mut actual)
            .unwrap();
        accumulate_ternary_matmul_scale_vjp(
            &x,
            &view.bind_matmul(),
            &dy,
            batch,
            &mut erow,
            &mut expected,
        )
        .unwrap();
        cpu.accumulate_additive_embed_scale_vjp(handle, &ids, &de, &mut arow, &mut actual)
            .unwrap();
        accumulate_embed_scale_vjp(&view.bind_gather(), &ids, &de, &mut erow, &mut expected)
            .unwrap();
        assert_eq!(bits64(&actual), bits64(&expected));
        assert_eq!(bits32(&arow), bits32(&erow));
        let count = cpu.additive_present_weight_count(handle).unwrap();
        assert_eq!(
            count,
            if view.plane_count() == 0 {
                0
            } else {
                view.plane_trits(0).unwrap().len()
            }
        );
        let mut actual = vec![0.25; count];
        let mut expected = actual.clone();
        cpu.accumulate_additive_matmul_weight_vjp(handle, &x, &dy, batch, &mut arow, &mut actual)
            .unwrap();
        accumulate_ternary_matmul_weight_vjp(
            &x,
            &view.bind_matmul(),
            &dy,
            batch,
            &mut erow,
            &mut expected,
        )
        .unwrap();
        cpu.accumulate_additive_embed_weight_vjp(handle, &ids, &de, &mut arow, &mut actual)
            .unwrap();
        accumulate_embed_weight_vjp(&view.bind_gather(), &ids, &de, &mut erow, &mut expected)
            .unwrap();
        assert_eq!(bits64(&actual), bits64(&expected));
        assert_eq!(bits32(&arow), bits32(&erow));
    }
}
