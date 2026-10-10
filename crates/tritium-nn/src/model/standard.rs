//! One standard-transformer inventory drives both preflight and assembly.

use super::{
    ModelWeights,
    hf::{DenseTensorRequest, NameSchema},
};
use crate::{
    ArchSpec, Mlp, MlpKind, ModelConfig, NnError, Projection, Relu2Mlp, SwiGluMlp, TokenEmbedding,
    TransformerBlock,
};

#[derive(Debug)]
pub(super) struct TensorRequest {
    pub(super) name: String,
    pub(super) shape: Vec<usize>,
}

impl TensorRequest {
    fn vector(name: String, len: usize) -> Self {
        Self {
            name,
            shape: vec![len],
        }
    }
    fn matrix(name: String, rows: usize, cols: usize) -> Self {
        Self {
            name,
            shape: vec![rows, cols],
        }
    }

    fn vector_values(
        &self,
        dense: &impl Fn(&str, DenseTensorRequest) -> Result<Vec<f32>, NnError>,
    ) -> Result<Vec<f32>, NnError> {
        let len = self.shape[0];
        let values = dense(&self.name, DenseTensorRequest::Vector { len })?;
        if values.len() != len {
            return Err(NnError::Shape {
                expected: len,
                got: values.len(),
            });
        }
        Ok(values)
    }

    fn projection(
        &self,
        proj: &mut impl FnMut(&str, usize, usize) -> Result<Projection, NnError>,
    ) -> Result<Projection, NnError> {
        proj(&self.name, self.shape[0], self.shape[1])
    }
}

#[derive(Debug)]
struct LayerPlan {
    gate: TensorRequest,
    up: TensorRequest,
    down: TensorRequest,
    attn_norm: TensorRequest,
    q: TensorRequest,
    k: TensorRequest,
    v: TensorRequest,
    o: TensorRequest,
    ffn_norm: TensorRequest,
    q_bias: Option<TensorRequest>,
    k_bias: Option<TensorRequest>,
    v_bias: Option<TensorRequest>,
    q_norm: Option<TensorRequest>,
    k_norm: Option<TensorRequest>,
    attn_sub_norm: Option<TensorRequest>,
    ffn_sub_norm: Option<TensorRequest>,
}

impl LayerPlan {
    fn tensors(&self) -> impl Iterator<Item = &TensorRequest> {
        [
            &self.gate,
            &self.up,
            &self.down,
            &self.attn_norm,
            &self.q,
            &self.k,
            &self.v,
            &self.o,
            &self.ffn_norm,
        ]
        .into_iter()
        .chain(
            [
                &self.q_bias,
                &self.k_bias,
                &self.v_bias,
                &self.q_norm,
                &self.k_norm,
                &self.attn_sub_norm,
                &self.ffn_sub_norm,
            ]
            .into_iter()
            .filter_map(Option::as_ref),
        )
    }
}

#[derive(Debug)]
pub(super) struct StandardModelPlan {
    pub(super) embedding: TensorRequest,
    output_norm: TensorRequest,
    lm_head: Option<TensorRequest>,
    layers: Vec<LayerPlan>,
}

impl StandardModelPlan {
    pub(super) fn new(
        config: &ModelConfig,
        spec: &ArchSpec,
        schema: NameSchema,
        vocab: usize,
    ) -> Result<Self, NnError> {
        let n_embd = config.n_embd as usize;
        if n_embd == 0 || vocab == 0 {
            return Err(NnError::Shape {
                expected: n_embd.max(1),
                got: 0,
            });
        }
        let head_dim = config.head_dim() as usize;
        if head_dim == 0 || !head_dim.is_multiple_of(2) {
            return Err(NnError::Backend(
                "attention head dimension must be nonzero and even".to_owned(),
            ));
        }
        let q_width = (config.n_head as usize)
            .checked_mul(head_dim)
            .ok_or_else(|| NnError::Backend("query projection width overflows usize".to_owned()))?;
        let kv_width = (config.n_head_kv as usize)
            .checked_mul(head_dim)
            .ok_or_else(|| NnError::Backend("KV projection width overflows usize".to_owned()))?;
        let n_ff = config.n_ff as usize;
        let mut layers = Vec::new();
        let layer_count = config.n_layers as usize;
        layers.try_reserve_exact(layer_count).map_err(|_| {
            NnError::Backend(format!(
                "allocate metadata for {layer_count} transformer layers"
            ))
        })?;
        for i in 0..layer_count {
            let vector = |key, len| TensorRequest::vector(schema.layer(i, key), len);
            let matrix = |key, rows, cols| TensorRequest::matrix(schema.layer(i, key), rows, cols);
            layers.push(LayerPlan {
                gate: matrix("gate", n_ff, n_embd),
                up: matrix("up", n_ff, n_embd),
                down: matrix("down", n_embd, n_ff),
                attn_norm: vector("attn_norm", n_embd),
                q: matrix("q", q_width, n_embd),
                k: matrix("k", kv_width, n_embd),
                v: matrix("v", kv_width, n_embd),
                o: matrix("o", n_embd, q_width),
                ffn_norm: vector("ffn_norm", n_embd),
                q_bias: spec.qkv_bias.then(|| vector("q_bias", q_width)),
                k_bias: spec.qkv_bias.then(|| vector("k_bias", kv_width)),
                v_bias: spec.qkv_bias.then(|| vector("v_bias", kv_width)),
                q_norm: spec.qk_norm.then(|| vector("q_norm", head_dim)),
                k_norm: spec.qk_norm.then(|| vector("k_norm", head_dim)),
                attn_sub_norm: spec.attn_sub_norm.then(|| vector("attn_sub_norm", q_width)),
                ffn_sub_norm: (spec.mlp == MlpKind::Relu2 && spec.ffn_sub_norm)
                    .then(|| vector("ffn_sub_norm", n_ff)),
            });
        }
        Ok(Self {
            embedding: TensorRequest::matrix(schema.top("token_embd").into(), vocab, n_embd),
            output_norm: TensorRequest::vector(schema.top("output_norm").into(), n_embd),
            lm_head: (!spec.tied_embeddings)
                .then(|| TensorRequest::matrix(schema.top("lm_head").into(), vocab, n_embd)),
            layers,
        })
    }

    pub(super) fn tensors(&self) -> impl Iterator<Item = &TensorRequest> {
        [&self.embedding, &self.output_norm]
            .into_iter()
            .chain(self.layers.iter().flat_map(LayerPlan::tensors))
            .chain(self.lm_head.iter())
    }

    pub(super) fn vocab(&self) -> usize {
        self.embedding.shape[0]
    }

    pub(super) fn assemble(
        &self,
        config: &ModelConfig,
        spec: &ArchSpec,
        token_embd: TokenEmbedding,
        dense: impl Fn(&str, DenseTensorRequest) -> Result<Vec<f32>, NnError>,
        mut proj: impl FnMut(&str, usize, usize) -> Result<Projection, NnError>,
    ) -> Result<ModelWeights, NnError> {
        let n_embd = self.embedding.shape[1];
        if token_embd.cols() != n_embd || token_embd.rows() != self.vocab() {
            return Err(NnError::Shape {
                expected: n_embd,
                got: token_embd.cols(),
            });
        }
        let optional_vector = |request: &Option<TensorRequest>| {
            request
                .as_ref()
                .map(|r| r.vector_values(&dense))
                .transpose()
                .map(Option::unwrap_or_default)
        };
        let output_norm = self.output_norm.vector_values(&dense)?;
        let mut layers = Vec::new();
        layers.try_reserve_exact(self.layers.len()).map_err(|_| {
            NnError::Backend(format!(
                "allocate metadata for {} transformer layers",
                self.layers.len()
            ))
        })?;
        for layer in &self.layers {
            let (gate, up, down) = (
                layer.gate.projection(&mut proj)?,
                layer.up.projection(&mut proj)?,
                layer.down.projection(&mut proj)?,
            );
            let mlp = match spec.mlp {
                MlpKind::SwiGlu => Mlp::SwiGlu(SwiGluMlp::new(gate, up, down)?),
                MlpKind::Relu2 => Mlp::Relu2(Relu2Mlp {
                    gate,
                    up,
                    down,
                    ffn_sub_norm: optional_vector(&layer.ffn_sub_norm)?,
                    rms_eps: config.rms_eps,
                }),
            };
            let (q_bias, k_bias, v_bias) = (
                optional_vector(&layer.q_bias)?,
                optional_vector(&layer.k_bias)?,
                optional_vector(&layer.v_bias)?,
            );
            let (q_norm, k_norm) = (
                optional_vector(&layer.q_norm)?,
                optional_vector(&layer.k_norm)?,
            );
            layers.push(TransformerBlock {
                attn_norm: layer.attn_norm.vector_values(&dense)?,
                q_proj: layer.q.projection(&mut proj)?,
                k_proj: layer.k.projection(&mut proj)?,
                v_proj: layer.v.projection(&mut proj)?,
                o_proj: layer.o.projection(&mut proj)?,
                attn_sub_norm: optional_vector(&layer.attn_sub_norm)?,
                q_bias,
                k_bias,
                v_bias,
                q_norm,
                k_norm,
                ffn_norm: layer.ffn_norm.vector_values(&dense)?,
                mlp,
            });
        }
        let lm_head = self
            .lm_head
            .as_ref()
            .map(|r| r.projection(&mut proj))
            .transpose()?;
        Ok(ModelWeights {
            token_embd,
            vocab: self.vocab(),
            n_embd,
            layers,
            output_norm,
            lm_head,
        })
    }
}
