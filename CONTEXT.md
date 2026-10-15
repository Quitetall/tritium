# Tritium tensor and training language

Tritium represents additive ternary models, their trainable state, and the
evidence needed to interpret and qualify them. These terms distinguish logical
model ownership from stored representations and measured execution.

## Language

**Tensor owner**:
The single owner of a logical tensor's storage. Tied names and aliases identify
that owner rather than independent copies.
_Avoid_: parameter copy, alias owner

**Training recipe**:
A declared computation and update rule over named tensor owners. It is distinct
from the state used to execute that computation.
_Avoid_: checkpoint, model weights

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

**Training snapshot**:
The parameter and optimizer state associated with one completed training step.
It is distinct from a hard ternary export.
_Avoid_: ternary artifact, tensor master

**Hard ternary artifact**:
A discrete additive-ternary representation intended for loading or inference.
It does not contain the latent training state needed to resume optimization.
_Avoid_: QAT checkpoint, dense master

**Qualification receipt**:
Evidence bound to a declared artifact, source, workload, and execution scope.
It is distinct from a structural plan or a claim of human authorization.
_Avoid_: capability declaration, approval
