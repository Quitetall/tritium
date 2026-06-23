/* roundtrip.c — minimal C consumer of the Tritium C ABI.
 *
 * Loads a GGUF model on the CPU backend, greedily generates token IDs from a
 * fixed prompt, prints them, and frees the model. Demonstrates the full
 * lifecycle: version probe -> load -> size-then-fill generate -> free.
 *
 * Build (against the staticlib; from the workspace root):
 *
 *   cargo build -p tritium-ffi --release
 *   cc -std=c11 -I crates/tritium-ffi/include crates/tritium-ffi/examples/roundtrip.c \
 *      target/release/libtritium_ffi.a -lpthread -ldl -lm -o /tmp/roundtrip
 *   /tmp/roundtrip path/to/model.gguf
 *
 * Or against the cdylib (shared library): link `-Ltarget/release -ltritium_ffi`
 * and set LD_LIBRARY_PATH (Linux) / DYLD_LIBRARY_PATH (macOS) accordingly.
 */

#include <tritium.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <model.gguf>\n", argv[0]);
        return 2;
    }

    printf("tritium %s (C ABI v%u)\n", tritium_version(), tritium_abi_version());

    enum TritiumStatus status = TritiumStatus_Panic;
    struct TritiumModel *model = tritium_model_load_file(argv[1], &status);
    if (model == NULL || status != TritiumStatus_Ok) {
        fprintf(stderr, "load failed: status=%d\n", (int)status);
        return 1;
    }

    /* A toy prompt of token IDs. A real caller tokenizes text first. */
    const uint32_t prompt[] = {1, 2, 3};
#define MAX_NEW 16
    const uint32_t eos = 128001; /* TRITIUM default EOS */

    /* The generated count never exceeds max_new, so size the buffer to max_new
     * and generate in a single pass — no separate sizing call (which would
     * re-run generation). Read the actual count from `produced`. */
    uint32_t out[MAX_NEW];
    size_t produced = 0;
    status = tritium_generate(model, prompt, sizeof(prompt) / sizeof(prompt[0]),
                              MAX_NEW, eos, out, MAX_NEW, &produced);
    if (status != TritiumStatus_Ok) {
        fprintf(stderr, "generate failed: status=%d\n", (int)status);
        tritium_model_free(model);
        return 1;
    }

    /* (If you must learn the length without a buffer, pass out=NULL, out_cap=0:
     * the call returns TritiumStatus_BufferTooSmall with *out_len set to the
     * required count — but it re-runs generation, so prefer the pass above.) */

    printf("generated %zu tokens:", produced);
    for (size_t i = 0; i < produced; i++) {
        printf(" %u", out[i]);
    }
    printf("\n");

    tritium_model_free(model);
    return 0;
}
