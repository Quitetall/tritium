export type TrainingOpCategoryV1 =
  | "graph"
  | "loss"
  | "optimizer"
  | "lifecycle";
export type TrainingVjpV1 = "none" | "first_order";

export interface TrainingOpDescriptorV1 {
  readonly id: string;
  readonly category: TrainingOpCategoryV1;
  readonly forward: boolean;
  readonly vjp: TrainingVjpV1;
  readonly mutates: boolean;
  readonly checkpoint_planes: readonly string[];
}

export interface TrainingOpManifestV1 {
  readonly schema_id: "tritium.training_op_manifest";
  readonly schema_version: 1;
  readonly dtype: "f32";
  readonly operations: readonly TrainingOpDescriptorV1[];
}

export declare class TrainingManifestError extends Error {
  constructor(message: string);
}

export declare function canonicalTrainingManifestJson(): Uint8Array;
export declare function parseTrainingManifest(
  data: string | Uint8Array,
): TrainingOpManifestV1;

export declare const TRAINING_MANIFEST_DIGEST_V1:
  "aefb352d04db145e48394b392a106ab0ad831e09e62d8c76ceddedb36a564083";
export declare const TRAINING_VECTOR_DIGEST_V1:
  "fcb250733b991aac165871f8c54b0b063337a3ed01bd1da02de220916887fbd6";
