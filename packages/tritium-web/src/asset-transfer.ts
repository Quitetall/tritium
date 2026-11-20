import { blake3 } from "@noble/hashes/blake3.js";
import { bytesToHex } from "@noble/hashes/utils.js";

// Internal implementation candidate for proposed ADR 0086. Not a public wire
// format, package export, model admission, or physical-memory qualification.
export interface AssetDescription {
  readonly byteLength: number;
  readonly blake3: string;
}

export interface AssetReader {
  read(offset: number, byteLength: number, signal?: AbortSignal): Promise<Uint8Array>;
}

export interface AssetStage {
  write(offset: number, bytes: Uint8Array): Promise<void>;
  /** Atomic publication after verification; no cancellation after this point. */
  complete(): Promise<void>;
  abort(): Promise<void>;
}

export interface AssetReceiver {
  /** Must reserve/admit receiver resources before constructing private state. */
  begin(description: AssetDescription): Promise<AssetStage>;
}

export interface AssetTransferLimits {
  readonly maxAssetBytes: number;
  readonly maxWindowBytes: number;
}

export class AssetTransferError extends Error {
  readonly code: "invalid_metadata" | "capacity" | "read_size" | "integrity" | "cancelled";

  constructor(code: AssetTransferError["code"], message: string) {
    super(message);
    this.name = "AssetTransferError";
    this.code = code;
  }
}

function intrinsicByteLengthGetter(): (this: Uint8Array) => number {
  const getter = Object.getOwnPropertyDescriptor(
    Object.getPrototypeOf(Uint8Array.prototype),
    "byteLength",
  )?.get;
  if (getter === undefined) throw new Error("typed-array byte-length getter unavailable");
  return getter;
}
const byteLengthGetter = intrinsicByteLengthGetter();

function cancelled(signal?: AbortSignal): void {
  if (signal?.aborted) throw new AssetTransferError("cancelled", "asset transfer cancelled");
}

/** Stream exact immutable bytes into unpublished receiver state. */
export async function transferAsset(
  input: AssetDescription,
  reader: AssetReader,
  receiver: AssetReceiver,
  limits: AssetTransferLimits,
  signal?: AbortSignal,
): Promise<
  Readonly<{ byteLength: number; windows: number; maxSnapshotBytes: number; blake3: string }>
> {
  // Capture caller metadata once; do not retain a mutable description object.
  const byteLength = input.byteLength;
  const expectedDigest = input.blake3;
  const maxAssetBytes = limits.maxAssetBytes;
  const maxWindowBytes = limits.maxWindowBytes;
  if (
    !Number.isSafeInteger(byteLength) ||
    byteLength <= 0 ||
    !Number.isSafeInteger(maxAssetBytes) ||
    maxAssetBytes <= 0 ||
    !Number.isSafeInteger(maxWindowBytes) ||
    maxWindowBytes <= 0 ||
    !/^[0-9a-f]{64}$/.test(expectedDigest)
  )
    throw new AssetTransferError("invalid_metadata", "asset metadata or limits invalid");
  if (byteLength > maxAssetBytes || maxWindowBytes > maxAssetBytes) {
    throw new AssetTransferError("capacity", "asset exceeds admitted transfer limits");
  }
  cancelled(signal);
  const description = Object.freeze({ byteLength, blake3: expectedDigest });
  const stage = await receiver.begin(description);
  let windows = 0;
  let maxSnapshotBytes = 0;
  try {
    const hash = blake3.create();
    for (let offset = 0; offset < byteLength; ) {
      cancelled(signal);
      const length = Math.min(maxWindowBytes, byteLength - offset);
      const value = await reader.read(offset, length, signal);
      cancelled(signal);
      if (
        !(value instanceof Uint8Array) ||
        !ArrayBuffer.isView(value) ||
        byteLengthGetter.call(value) !== length
      ) {
        throw new AssetTransferError(
          "read_size",
          "reader did not return the exact requested window",
        );
      }
      // Own one bounded raw-byte snapshot. Ignore caller iterators/byteLength
      // overrides; the receiver may borrow it only until write resolves.
      const snapshot = new Uint8Array(length);
      snapshot.set(value);
      hash.update(snapshot);
      await stage.write(offset, snapshot);
      cancelled(signal);
      windows += 1;
      maxSnapshotBytes = Math.max(maxSnapshotBytes, length);
      offset += length;
    }
    const observed = bytesToHex(hash.digest());
    if (observed !== expectedDigest)
      throw new AssetTransferError("integrity", "asset digest differs");
    cancelled(signal);
    // This is the linearization point. A trusted receiver must publish
    // atomically; cancellation during complete does not revoke success.
    await stage.complete();
    return Object.freeze({ byteLength, windows, maxSnapshotBytes, blake3: observed });
  } catch (error) {
    try {
      await stage.abort();
    } catch (cleanupError) {
      throw new AggregateError([error, cleanupError], "asset transfer and abort failed");
    }
    throw error;
  }
}
