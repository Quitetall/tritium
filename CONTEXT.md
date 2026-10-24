# Tritium tensor and training language

Tritium represents additive ternary models, their trainable state, and the
evidence needed to interpret and qualify them. These terms distinguish logical
model ownership from stored representations and measured execution.

## Language

**Code unit**:
A jointly projected set of additive ternary coefficients for one logical weight,
or one four-weight sparse 3:4 block, across its present planes.
_Avoid_: optimizer step, scale group, tensor owner

**Present weight coordinate**:
A stored-basis logical weight position whose tile retains a base ternary plane,
even when its coefficient is zero. It is distinct from an individual plane coefficient.
_Avoid_: nonzero weight, dense slot, trit index

**Source-bound chat prompt**:
A conversation rendered using the template and token vocabulary belonging to
the loaded model source. It is distinct from an architecture-based approximation.
_Avoid_: generic chat wrapper, qualified model response

**Source chat controls**:
Explicit choices supplied to a source-bound chat template. They are inputs to
that template, not guarantees about the model's reasoning or generated output.
_Avoid_: reasoning enforcement, prompt rewrite

**Tensor owner**:
The single owner of a logical tensor's storage. Tied names and aliases identify
that owner rather than independent copies.
_Avoid_: parameter copy, alias owner

**Training recipe**:
A declared computation and update rule over named tensor owners. It is distinct
from the state used to execute that computation.
_Avoid_: checkpoint, model weights

**Recorded resume point**:
A training snapshot bound to the exact recorded update prefix that produced it.
It is distinct from a hard ternary artifact or permission to erase later work.
_Avoid_: completed model, truncation authorization

**Recorded resume bundle**:
A self-contained recorded resume point and its recovery history. It is training
recovery input, not a refined child or qualification receipt.
_Avoid_: deployable model, qualified checkpoint

**Owner-paged recovery**:
Recovery that retains committed training state outside resident owner payloads
and materializes one tensor owner at a time. It is distinct from within-owner
paging, a resident-memory measurement, or a qualified trained model.
_Avoid_: GPU training, memory-mapped model, completed refinement

**Staged recorded update**:
A private candidate training snapshot and its exact gradient observations,
awaiting admission into the current recorded history. It is distinct from a
committed resume point, a refined child or qualification evidence.
_Avoid_: completed training, published checkpoint, verified model

**Packed training cohort**:
The complete named set of immutable packed owner versions corresponding to one
committed training snapshot. It is distinct from an individual tensor version or
permission to replace a live graph before its recorded update commits.
_Avoid_: trained model, qualified checkpoint, partial owner update

**Pending gradient window**:
Uncommitted owner adjoints associated with one current training revision. It is
distinct from a committed training snapshot or authenticated objective execution.
_Avoid_: trained checkpoint, recovery bundle, qualification receipt

**Owner calibration recipe**:
An ordered, weighted set of declared scalar objectives and observation limits
for tensor owners. It is distinct from captured observations and qualification.
_Avoid_: Fisher estimate, qualified profile

**Owner window capture**:
The observed contribution of every supported use of a tensor owner to one
scalar objective over a declared input window.
_Avoid_: independent alias gradient, qualification receipt

**Training payload**:
The initial tensor state supplied to a training recipe. It is not a trained
artifact or evidence that the recipe has executed.
_Avoid_: trained model, qualification

**Basis-restored input adjoint**:
The loss contribution to a matmul consumer's input in its original coordinates,
after undoing the stored tensor basis. It is distinct from a scale or code update.
_Avoid_: ternary weight gradient, hard-code refinement

**Training snapshot**:
The parameter and optimizer state associated with one completed training step.
It is distinct from a hard ternary export.
_Avoid_: ternary artifact, tensor master

**Hard ternary artifact**:
A discrete additive-ternary representation intended for loading or inference.
It does not contain the latent training state needed to resume optimization.
_Avoid_: QAT checkpoint, dense master

**Recovery replay**:
Re-execution of a ternary training recipe from its original training payload,
reproducing the same training snapshot. It is distinct from authenticating the
objective inputs or qualifying the hard ternary artifact.
_Avoid_: execution attestation, qualification

**Refined child**:
A complete immutable hard artifact descended from one explicitly selected parent
level through a declared recovery run. Its ancestry is distinct from quality qualification.
_Avoid_: direct PTQ fit, overwritten parent, multilevel refinement

**Immediate refinement parent**:
The exact hard artifact and selected level supplying the current run's starting
values. Its admission does not verify earlier refinement edges.
_Avoid_: original source, verified entire ancestry

**Model shell**:
A model structure supplied to receive the stored state of a hard artifact.
Its tensor shapes, dtypes and aliases are not evidence of source tensor values.
_Avoid_: original model, dense checkpoint, observed source

**Tensor work record**:
An immutable, content-bound piece of intermediate conversion work for one tensor.
It is distinct from a deployable model, a training snapshot and a qualification receipt.
_Avoid_: model checkpoint, completion proof

**Qualification receipt**:
Evidence bound to a declared artifact, source, workload, and execution scope.
It is distinct from a structural plan or a claim of human authorization.
_Avoid_: capability declaration, approval

**Release candidate cohort**:
The exact source-, version- and byte-bound Tritium artifact inventory awaiting
qualification and public activation. Its version may be final or prerelease;
its existence is not publication authorization.
_Avoid_: prerelease label, published release, release approval

**Resident payload envelope**:
A reserved logical ceiling covering a resident model's matrices and execution
state. It is distinct from actual payload consumption and measured physical memory.
_Avoid_: peak VRAM, exact allocation plan, physical fit

**Resident payload plan**:
The checked logical matrix, coefficient, state and workspace sizes for one
resident graph and stream policy. It is not a reserved ceiling or physical memory.
_Avoid_: resident envelope, measured VRAM, qualification receipt
