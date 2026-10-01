// Generated from tritium-schema Rust types; do not edit.
// JavaScript numbers cannot exactly represent every uint64 JSON integer.
// Use a lossless JSON parser when consuming values above Number.MAX_SAFE_INTEGER.

export interface AdditiveLayout {
  readonly allocation: PlaneAllocation;
  readonly basis: Basis;
  readonly codec: PlaneCodec;
  readonly cols: number;
  readonly group: number;
  readonly law: ScaleLaw;
  readonly max_planes: number;
  readonly rows: number;
  readonly tile: number;
  readonly transport: Transport;
}

export interface AdmittedLaw {
  readonly law: ScaleLaw;
  readonly max_planes: number;
}

export type Basis = "Identity" | { readonly Hadamard: { readonly block: number } } | { readonly SignedRht: { readonly block: number; readonly domain: number; readonly seed: number } };

export type BlobId = [number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number];

export type DenseDType = "Bf16" | "F16" | "F32";

export interface EvidenceEnvelope {
  readonly digest: string;
  readonly parent?: null | string;
  readonly payload: unknown;
  readonly run: string;
  readonly schema: string;
  readonly seq: number;
  readonly span: string;
  readonly t: number;
  readonly v: number;
}

export type LayoutError = "BasisBlockDoesNotDivideColumns" | "EmptyDimensions" | "EmptyPlaneStack" | "EmptyTransportBlock" | "InvalidBasisBlock" | "InvalidScaleRatio" | "PlaneCountExceedsLaw" | "UnsupportedGroup" | "UnsupportedLaw" | "UnsupportedTileSize";

export interface Level {
  readonly name: string;
  readonly provenance: LevelProvenance;
  readonly tensors: Readonly<Record<string, BlobId>>;
}

export type LevelProvenance = "Imported" | { readonly Direct: { readonly evidence: string } } | { readonly ImportedPrefixOf: { readonly level: string } };

export type ModelId = [number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number];

export interface ModelManifest {
  readonly architecture: string;
  readonly assets: Readonly<Record<string, BlobId>>;
  readonly legacy_ids: Readonly<Record<string, string>>;
  readonly levels: ReadonlyArray<Level>;
  readonly schema: SchemaId;
  readonly tensors: Readonly<Record<string, TensorMeta>>;
}

export type PackageId = [number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number];

export type PlaneAllocation = "PerTile" | "Uniform";

export type PlaneCodec = "B3" | "D2" | "S34";

export type PlaneRelation = "Free" | { readonly Tied: { readonly den: number; readonly num: number } };

export type ScaleAnchor = "Channel" | "Group" | "Row" | "Tensor";

export interface ScaleLaw {
  readonly anchor: ScaleAnchor;
  readonly precision: ScalePrecision;
  readonly relation: PlaneRelation;
}

export type ScalePrecision = "Bf16" | "F16" | "F32";

export interface SchemaId {
  readonly major: number;
  readonly minor: number;
}

export type SemanticTensorDigest = [number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number];

export type TensorMeta = { readonly Additive: { readonly blob: BlobId; readonly layout: AdditiveLayout; readonly semantic_digest: SemanticTensorDigest } } | { readonly Dense: { readonly blob: BlobId; readonly dtype: DenseDType; readonly shape: ReadonlyArray<number> } };

export type Transport = "Raw" | { readonly JointHuffman: { readonly block: number } };

export type UnknownReason = "AwaitingIndependentReview" | "MissingCapability" | "MissingEvidence" | "NotEvaluated" | "Other";

export type Verdict = "Fail" | "Pass" | { readonly Unknown: { readonly reason: UnknownReason } };
