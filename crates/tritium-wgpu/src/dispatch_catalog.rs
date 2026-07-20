//! Shared portable-training WebGPU execution catalog.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PortableDispatchStageV1 {
    pub(crate) module_id: &'static str,
    pub(crate) entry_point: &'static str,
    pub(crate) selector: Option<u32>,
    pub(crate) dispatch: &'static str,
    pub(crate) repeat: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PortableDispatchFormV1 {
    pub(crate) operation: &'static str,
    pub(crate) execution: &'static str,
    pub(crate) stages: &'static [PortableDispatchStageV1],
}

include!(concat!(env!("OUT_DIR"), "/portable_dispatch_catalog.rs"));

pub(crate) fn portable_dispatch_form_v1(
    operation: &str,
    execution: &str,
) -> Option<&'static PortableDispatchFormV1> {
    PORTABLE_DISPATCH_FORMS_V1
        .iter()
        .find(|form| form.operation == operation && form.execution == execution)
}

pub(crate) fn portable_pointwise_selector_v1(
    operation: &str,
    execution: &str,
    stage: usize,
) -> Option<u32> {
    portable_dispatch_form_v1(operation, execution)?
        .stages
        .get(stage)?
        .selector
}

#[cfg(test)]
mod tests {
    use super::{PORTABLE_DISPATCH_FORMS_V1, portable_dispatch_form_v1};

    #[test]
    fn shared_catalog_covers_fifty_seven_forms_and_phase_specific_stages() {
        assert_eq!(PORTABLE_DISPATCH_FORMS_V1.len(), 57);
        let salt_vjp = portable_dispatch_form_v1("graph.salt_ste", "vjp").unwrap();
        assert_eq!(salt_vjp.stages[0].module_id, "pointwise");
        assert_eq!(salt_vjp.stages[0].selector, Some(0));
        assert_eq!(
            portable_dispatch_form_v1("graph.add", "vjp")
                .unwrap()
                .stages
                .len(),
            2
        );
        assert_eq!(
            portable_dispatch_form_v1("graph.concat_cols", "vjp")
                .unwrap()
                .stages[0]
                .repeat,
            "per_output"
        );
        let int8 = portable_dispatch_form_v1("optimizer.int8_adamw", "step").unwrap();
        assert_eq!(int8.stages.len(), 8);
        assert_eq!(int8.stages[6].entry_point, "reduce_scales");
        assert_eq!(int8.stages[6].dispatch, "optimizer_blocks_256");
    }
}
