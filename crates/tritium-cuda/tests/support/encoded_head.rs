// Authored repeating canonical bytes, never a converted model or dense master.
use tritium_core::UniformPlan;
use tritium_schema::{
    AdditiveLayout, Basis, PlaneAllocation, PlaneCodec, PlaneRelation, ScaleAnchor, ScaleLaw,
    ScalePrecision, Transport,
};
use tritium_spec::{
    AdditiveMetadata, BackendError, TENSOR_STREAM_CHUNK_BYTES, TensorMetadata,
    TensorStreamEncoding, TensorStreamSource,
};

pub(super) const SCALE_BITS: [u16; 5] = [0x0001, 0x0400, 0x2e66, 0x3800, 0x4200];

#[derive(Debug)]
pub(super) struct EncodedHead {
    pub(super) layout: AdditiveLayout,
    pub(super) planes: u8,
    pub(super) seed: Option<u64>,
    plane_trits: usize,
    plane_bytes: usize,
    scales: usize,
    bytes: usize,
}

impl EncodedHead {
    pub(super) fn new(rows: usize, cols: usize, planes: u8) -> Self {
        assert!(rows > 0 && cols > 0 && (1..=3).contains(&planes));
        let layout = AdditiveLayout {
            rows: rows as u64,
            cols: cols as u64,
            tile: 256,
            group: 128,
            max_planes: planes,
            allocation: PlaneAllocation::Uniform,
            codec: PlaneCodec::B3,
            law: ScaleLaw {
                anchor: ScaleAnchor::Group,
                relation: PlaneRelation::Free,
                precision: ScalePrecision::F16,
            },
            basis: Basis::Identity,
            transport: Transport::Raw,
        };
        let plan = UniformPlan::new(layout, planes).unwrap();
        let plane_trits = rows.checked_mul(cols).unwrap();
        let plane_bytes = plane_trits.div_ceil(5);
        let scales = plan.scale_count();
        let bytes = plane_bytes
            .checked_mul(planes as usize)
            .unwrap()
            .checked_add(scales.checked_mul(2).unwrap())
            .unwrap();
        Self {
            layout,
            planes,
            seed: None,
            plane_trits,
            plane_bytes,
            scales,
            bytes,
        }
    }

    pub(super) fn counter_mixed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    pub(super) fn code_at(&self, plane: usize, byte: usize) -> u8 {
        if let Some(seed) = self.seed {
            let mut value = (byte as u64)
                .wrapping_add((plane as u64).wrapping_mul(0x9e3779b97f4a7c15))
                .wrapping_add(seed);
            value = (value ^ (value >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            value = (value ^ (value >> 27)).wrapping_mul(0x94d049bb133111eb);
            ((value ^ (value >> 31)) % 243) as u8
        } else {
            (((byte % 243) * 73 + plane * 19) % 243) as u8
        }
    }
}

impl TensorStreamSource for EncodedHead {
    fn metadata(&self) -> TensorMetadata<'_> {
        TensorMetadata::Additive(AdditiveMetadata::uniform(self.layout, self.planes).unwrap())
    }
    fn encoding(&self) -> TensorStreamEncoding {
        TensorStreamEncoding::CanonicalRawAdditive
    }
    fn encoded_bytes(&self) -> u64 {
        self.bytes as u64
    }
    fn visit(
        &self,
        visitor: &mut dyn FnMut(&[u8]) -> Result<(), BackendError>,
    ) -> Result<(), BackendError> {
        // Fixed scratch, regardless of matrix dimensions. The closure is
        // monomorphic per byte; the external visitor runs only per window.
        let mut window = [0u8; TENSOR_STREAM_CHUNK_BYTES];
        let mut used = 0;
        {
            let mut emit = |byte: u8| -> Result<(), BackendError> {
                window[used] = byte;
                used += 1;
                if used == window.len() {
                    visitor(&window)?;
                    used = 0;
                }
                Ok(())
            };
            for plane in 0..self.planes as usize {
                let mut code = (plane * 19) % 243;
                for index in 0..self.plane_bytes {
                    let mut byte = if self.seed.is_some() {
                        self.code_at(plane, index) as usize
                    } else {
                        code
                    };
                    let remainder = self.plane_trits % 5;
                    if index + 1 == self.plane_bytes && remainder != 0 {
                        let radix = 3usize.pow(remainder as u32);
                        // Unused B3 digits are canonical ZERO (digit one).
                        byte = byte % radix + (243 - radix) / 2;
                    }
                    emit(byte as u8)?;
                    code += 73;
                    if code >= 243 {
                        code -= 243;
                    }
                }
            }
            for index in 0..self.scales {
                for byte in SCALE_BITS[index % SCALE_BITS.len()].to_le_bytes() {
                    emit(byte)?;
                }
            }
        }
        if used > 0 {
            visitor(&window[..used])?;
        }
        Ok(())
    }
}
