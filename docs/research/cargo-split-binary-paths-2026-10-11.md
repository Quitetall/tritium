# Cargo binary-path test adapter repair

At clean72fa46ed, exact invocation9161f05c3a95444f88416ee8f74c5548 failed
four typed conversion child deadlines. The unchanged isolated plain fixture
subsequently passed in0.406 seconds; the unchanged complete parallel target
passed all5 cases in2.414 seconds (invocations e5d547ec35ed4d28ba9a262fc126ebe3
and b120448cb04745a893f34f446ce0fc6b). Recent host I/O stalls were severe and
then declined. Transient contention is supported, not causally proved. No
production change, deadline increase or test serialization was used.

Exact repeat712a94da542c404984d82603b4ade3f3 reached another real failure:
all3 generated-directory dispatch tests could not launch the CLI (OS NotFound).
They walked current_exe out of deps to guess the executable. The workstation's
normal Cargo build-dir puts intermediate tests in /mnt/4tb/build/cargo while
finished binaries live in the project target directory. The convert and ladder
roundtrip helpers made the same invalid layout assumption.

The three test adapters now use Cargo's CARGO_BIN_EXE_tritium, like the existing
typed conversion tests. No source/wire/numerical/CLI or model gate changes, no
binary symlink, shared intermediate override or hidden fallback.

Scoped invocationdf7c6e2ebf7a47a2af295b9337fd361f is terminal0:the actual
generate_converted_dir, ladder_bundle_roundtrip and convert_roundtrip targets
passed their nonignored cases under normal split build-dir; real-model ignores
remain unexecuted. Strict CLI all-target Clippy, fmt and diff/parent guards passed.
Original failure/control/source and corrected scoped evidence are retained in
the project's durable verification archives. Exact committed combined/canonical
checks and ordinary publication remain required. This is software validation,
not model quality, runtime performance, physical backend or release qualification.
