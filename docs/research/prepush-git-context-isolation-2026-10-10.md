# Pre-push Git context isolation

During the distributed-admission milestone, the normal pre-push hook detached
the originating managed checkout at the pushed source. The user checkout's
foreign staged diff remained unchanged. This exposed an unsafe scratch boundary,
not a distributed-model or compiler failure.

Git exports repository-local environment selectors to hooks. Inherited
`GIT_DIR` overrides `git -C` repository selection. Explicit `GIT_WORK_TREE`,
`GIT_INDEX_FILE` and `GIT_COMMON_DIR` can additionally redirect scratch checkout
and cleanup toward the originating files/index. The hook now resolves its
origin once, clears variables reported by `git rev-parse --local-env-vars`, and
anchors root operations there. Scratch commands then discover their own
registered worktree normally. Gate scope and bypass policy are unchanged.

The original hook was reproduced in a disposable real-Git fixture with a
warm scratch worktree and a candidate commit distinct from the owner branch.
The minimal case detached the owner branch. The expanded baseline failed
three of four cases: clean/staged origins crossed with basic/full inherited
selectors. The full-selector staged case changed the owner's staged diff.
Only fixture data was affected. No user data was deleted or reset.

The repaired fixture checks branch/HEAD, staged diff, tracked bytes and foreign
untracked data for all four contexts. Cargo is stubbed in this test: it proves
Git isolation, not formatting, compilation or release qualification. Normal
pre-commit and pre-push gates remain separate and must run without a bypass.
Canonical script-test discovery runs this regression in CI.

Red/green fixture logs and the explicitly unsafe baseline hook snapshot are
retained with the distributed-admission evidence under
`/home/brianklam/Projects/Tritium/archive/verification/distributed-binding-d92f7124-20261010`.
Historical campaigns, shared caches and foreign worktrees are outside cleanup.
