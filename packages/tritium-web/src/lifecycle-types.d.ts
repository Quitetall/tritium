export type PortableCheckpointOptimizerV1 =
  | "sgd"
  | "adamw"
  | "cautious_adamw"
  | "int8_adamw"
  | "muon";

export interface PortableSgdLeafV1 {
  readonly parameter: readonly number[];
}

export interface PortableAdamLeafV1 extends PortableSgdLeafV1 {
  readonly moment1: readonly number[];
  readonly moment2: readonly number[];
}

export interface PortableInt8AdamLeafV1 extends PortableSgdLeafV1 {
  readonly moment1Q8: readonly number[];
  readonly moment2Q8: readonly number[];
  readonly moment1Scale: readonly number[];
  readonly moment2Scale: readonly number[];
}

export interface PortableMuonLeafV1 extends PortableSgdLeafV1 {
  readonly momentum: readonly number[];
}

export type PortableCheckpointStateV1 =
  | {
      readonly optimizer: "sgd";
      readonly step: number;
      readonly leaves: readonly PortableSgdLeafV1[];
    }
  | {
      readonly optimizer: "adamw" | "cautious_adamw";
      readonly step: number;
      readonly leaves: readonly PortableAdamLeafV1[];
    }
  | {
      readonly optimizer: "int8_adamw";
      readonly step: number;
      readonly leaves: readonly PortableInt8AdamLeafV1[];
    }
  | {
      readonly optimizer: "muon";
      readonly step: number;
      readonly leaves: readonly PortableMuonLeafV1[];
    };
