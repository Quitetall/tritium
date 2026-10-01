# Generated from tritium-schema Rust types; do not edit.
from __future__ import annotations

from typing import Any, Literal, Union
from typing_extensions import NotRequired, TypeAlias, TypedDict

class BasisVariant1(TypedDict):
    Hadamard: BasisVariant1Hadamard

class BasisVariant1Hadamard(TypedDict):
    block: int

class BasisVariant2(TypedDict):
    SignedRht: BasisVariant2SignedRht

class BasisVariant2SignedRht(TypedDict):
    block: int
    domain: int
    seed: int

class LevelProvenanceVariant0(TypedDict):
    Direct: LevelProvenanceVariant0Direct

class LevelProvenanceVariant0Direct(TypedDict):
    evidence: str

class LevelProvenanceVariant1(TypedDict):
    ImportedPrefixOf: LevelProvenanceVariant1ImportedPrefixOf

class LevelProvenanceVariant1ImportedPrefixOf(TypedDict):
    level: str

class PlaneRelationVariant1(TypedDict):
    Tied: PlaneRelationVariant1Tied

class PlaneRelationVariant1Tied(TypedDict):
    den: int
    num: int

class TensorMetaVariant0(TypedDict):
    Additive: TensorMetaVariant0Additive

class TensorMetaVariant0Additive(TypedDict):
    blob: BlobId
    layout: AdditiveLayout
    semantic_digest: SemanticTensorDigest

class TensorMetaVariant1(TypedDict):
    Dense: TensorMetaVariant1Dense

class TensorMetaVariant1Dense(TypedDict):
    blob: BlobId
    dtype: DenseDType
    shape: list[int]

class TransportVariant1(TypedDict):
    JointHuffman: TransportVariant1JointHuffman

class TransportVariant1JointHuffman(TypedDict):
    block: int

class VerdictVariant2(TypedDict):
    Unknown: VerdictVariant2Unknown

class VerdictVariant2Unknown(TypedDict):
    reason: UnknownReason

class AdditiveLayout(TypedDict):
    allocation: PlaneAllocation
    basis: Basis
    codec: PlaneCodec
    cols: int
    group: int
    law: ScaleLaw
    max_planes: int
    rows: int
    tile: int
    transport: Transport

class AdmittedLaw(TypedDict):
    law: ScaleLaw
    max_planes: int

Basis: TypeAlias = Union[BasisVariant1, BasisVariant2, Literal["Identity"]]

BlobId: TypeAlias = tuple[int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int]

DenseDType: TypeAlias = Union[Literal["Bf16"], Literal["F16"], Literal["F32"]]

class EvidenceEnvelope(TypedDict):
    digest: str
    parent: NotRequired[Union[None, str]]
    payload: Any
    run: str
    schema: str
    seq: int
    span: str
    t: int
    v: int

LayoutError: TypeAlias = Union[Literal["BasisBlockDoesNotDivideColumns"], Literal["EmptyDimensions"], Literal["EmptyPlaneStack"], Literal["EmptyTransportBlock"], Literal["InvalidBasisBlock"], Literal["InvalidScaleRatio"], Literal["PlaneCountExceedsLaw"], Literal["UnsupportedGroup"], Literal["UnsupportedLaw"], Literal["UnsupportedTileSize"]]

class Level(TypedDict):
    name: str
    provenance: LevelProvenance
    tensors: dict[str, BlobId]

LevelProvenance: TypeAlias = Union[LevelProvenanceVariant0, LevelProvenanceVariant1, Literal["Imported"]]

ModelId: TypeAlias = tuple[int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int]

class ModelManifest(TypedDict):
    architecture: str
    assets: dict[str, BlobId]
    legacy_ids: dict[str, str]
    levels: list[Level]
    schema: SchemaId
    tensors: dict[str, TensorMeta]

PackageId: TypeAlias = tuple[int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int]

PlaneAllocation: TypeAlias = Union[Literal["PerTile"], Literal["Uniform"]]

PlaneCodec: TypeAlias = Union[Literal["B3"], Literal["D2"], Literal["S34"]]

PlaneRelation: TypeAlias = Union[Literal["Free"], PlaneRelationVariant1]

ScaleAnchor: TypeAlias = Union[Literal["Channel"], Literal["Group"], Literal["Row"], Literal["Tensor"]]

class ScaleLaw(TypedDict):
    anchor: ScaleAnchor
    precision: ScalePrecision
    relation: PlaneRelation

ScalePrecision: TypeAlias = Union[Literal["Bf16"], Literal["F16"], Literal["F32"]]

class SchemaId(TypedDict):
    major: int
    minor: int

SemanticTensorDigest: TypeAlias = tuple[int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int]

TensorMeta: TypeAlias = Union[TensorMetaVariant0, TensorMetaVariant1]

Transport: TypeAlias = Union[Literal["Raw"], TransportVariant1]

UnknownReason: TypeAlias = Union[Literal["AwaitingIndependentReview"], Literal["MissingCapability"], Literal["MissingEvidence"], Literal["NotEvaluated"], Literal["Other"]]

Verdict: TypeAlias = Union[Literal["Fail"], Literal["Pass"], VerdictVariant2]
