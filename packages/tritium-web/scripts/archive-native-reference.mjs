import { constants } from "node:fs";
import { open } from "node:fs/promises";
import { resolve } from "node:path";
import { validateNativeReferenceV1 } from "../../../scripts/run-browser-training-lane.mjs";

// Internal packaging input, not a new release receipt or public package API.
export async function readBoundedOrdinaryFile(path, maximumBytes) {
  const file = await open(path, constants.O_RDONLY | constants.O_NOFOLLOW | constants.O_NONBLOCK);
  try {
    const metadata = await file.stat();
    if (!metadata.isFile() || metadata.size <= 0 || metadata.size > maximumBytes) {
      throw new Error("native parity input must be a bounded nonempty ordinary file");
    }
    const bytes = Buffer.alloc(maximumBytes + 1);
    let length = 0;
    while (length < bytes.byteLength) {
      const { bytesRead } = await file.read(bytes, length, bytes.byteLength - length, length);
      if (bytesRead === 0) break;
      length += bytesRead;
    }
    if (length !== metadata.size || length > maximumBytes) {
      throw new Error("native parity input changed size or exceeded its bound");
    }
    return bytes.subarray(0, length);
  } finally {
    await file.close();
  }
}

export async function admitArchiveNativeReference(directory, revision) {
  if (typeof directory !== "string" || directory.length === 0 || directory.includes("\0")) {
    throw new Error("native parity reference directory is invalid");
  }
  const bytes = await readBoundedOrdinaryFile(resolve(directory, "native.salt"), 8 * 1024 * 1024);
  const receiptBytes = await readBoundedOrdinaryFile(
    resolve(directory, "receipt.json"),
    1024 * 1024,
  );
  const receipt = JSON.parse(receiptBytes.toString("utf8"));
  const admitted = validateNativeReferenceV1(receipt, bytes, "native.salt", revision);
  return { bytes, receiptBytes, admitted };
}
