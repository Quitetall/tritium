import init, {
  tritium_portable_build_id,
  tritium_portable_conformance_case_count,
  tritium_portable_max_caller_bytes,
  tritium_portable_manifest_digest,
  tritium_portable_max_linear_memory_bytes,
  tritium_portable_operation_count,
  tritium_portable_report_digest,
  tritium_portable_vector_digest,
} from "../.generated/tritium_wasm.js";
import { WASM_GUEST_DIGEST_V1 } from "../.generated/wasm_identity.ts";
import { blake3 } from "@noble/hashes/blake3.js";
import { bytesToHex } from "@noble/hashes/utils.js";

import {
  TRAINING_MANIFEST_DIGEST_V1,
  TRAINING_VECTOR_DIGEST_V1,
} from "./identity.ts";

export interface PortableWasmConformanceReceiptV1 {
  readonly schemaId: "tritium.portable_wasm_conformance_receipt";
  readonly schemaVersion: 1;
  readonly implementation: "wasm-fallback";
  readonly engine: "wasm32-unknown-unknown";
  readonly buildId: string;
  readonly guestDigest: typeof WASM_GUEST_DIGEST_V1;
  readonly executionDigest: string;
  readonly manifestDigest: typeof TRAINING_MANIFEST_DIGEST_V1;
  readonly vectorDigest: typeof TRAINING_VECTOR_DIGEST_V1;
  readonly operationCount: number;
  readonly caseCount: number;
  readonly maxCallerBytes: number;
  readonly maxLinearMemoryBytes: number;
  readonly repeatedExecutions: 2;
}

let initialized: Promise<void> | null = null;

export type PortableWasmSourceV1 =
  | RequestInfo
  | URL
  | Response
  | BufferSource;

async function readGuestBytes(source: PortableWasmSourceV1): Promise<Uint8Array> {
  if (source instanceof Response) {
    return new Uint8Array(await source.arrayBuffer());
  }
  if (ArrayBuffer.isView(source)) {
    return new Uint8Array(
      source.buffer.slice(source.byteOffset, source.byteOffset + source.byteLength),
    );
  }
  if (source instanceof ArrayBuffer) return new Uint8Array(source.slice(0));
  const response = await fetch(source);
  if (!response.ok) {
    throw new Error(`portable WASM fetch failed with HTTP ${response.status}`);
  }
  return new Uint8Array(await response.arrayBuffer());
}

/** Execute the complete canonical vector corpus twice inside the bundled guest. */
export async function runPortableWasmConformance(
  source: PortableWasmSourceV1 = new URL(
    "./tritium_wasm_bg.wasm",
    import.meta.url,
  ),
): Promise<PortableWasmConformanceReceiptV1> {
  const guestBytes = await readGuestBytes(source);
  const guestDigest = bytesToHex(blake3(guestBytes));
  if (guestDigest !== WASM_GUEST_DIGEST_V1) {
    throw new Error(
      `portable WASM guest digest mismatch: expected ${WASM_GUEST_DIGEST_V1}, got ${guestDigest}`,
    );
  }
  if (initialized === null) {
    const attempt = init({ module_or_path: guestBytes }).then(() => undefined);
    initialized = attempt;
    try {
      await attempt;
    } catch (error) {
      if (initialized === attempt) initialized = null;
      throw error;
    }
  } else {
    await initialized;
  }
  const firstExecutionDigest = tritium_portable_report_digest();
  const secondExecutionDigest = tritium_portable_report_digest();
  const operationCount = tritium_portable_operation_count();
  const caseCount = tritium_portable_conformance_case_count();
  const maxCallerBytes = tritium_portable_max_caller_bytes();
  const maxLinearMemoryBytes = tritium_portable_max_linear_memory_bytes();
  const guestManifestDigest = tritium_portable_manifest_digest();
  const guestVectorDigest = tritium_portable_vector_digest();
  if (
    operationCount !== 35 ||
    caseCount !== 114 ||
    maxCallerBytes !== 64 * 1024 * 1024 ||
    maxLinearMemoryBytes !== 192 * 1024 * 1024 ||
    guestManifestDigest !== TRAINING_MANIFEST_DIGEST_V1 ||
    guestVectorDigest !== TRAINING_VECTOR_DIGEST_V1 ||
    !/^[0-9a-f]{64}$/.test(firstExecutionDigest) ||
    secondExecutionDigest !== firstExecutionDigest
  ) {
    throw new Error(
      `portable WASM conformance failed: operations=${operationCount}, cases=${caseCount}, maxCallerBytes=${maxCallerBytes}, maxLinearMemoryBytes=${maxLinearMemoryBytes}, manifest=${guestManifestDigest}, vectors=${guestVectorDigest}, execution=${firstExecutionDigest}/${secondExecutionDigest}`,
    );
  }
  return Object.freeze({
    schemaId: "tritium.portable_wasm_conformance_receipt",
    schemaVersion: 1,
    implementation: "wasm-fallback",
    engine: "wasm32-unknown-unknown",
    buildId: tritium_portable_build_id(),
    guestDigest: WASM_GUEST_DIGEST_V1,
    executionDigest: firstExecutionDigest,
    manifestDigest: TRAINING_MANIFEST_DIGEST_V1,
    vectorDigest: TRAINING_VECTOR_DIGEST_V1,
    operationCount,
    caseCount,
    maxCallerBytes,
    maxLinearMemoryBytes,
    repeatedExecutions: 2,
  });
}
