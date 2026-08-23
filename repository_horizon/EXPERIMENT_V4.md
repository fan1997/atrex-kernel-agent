# Repository Horizon v4 architecture-escape experiment

## Purpose

This experiment evaluates a general repository-optimization policy, not a known solution. Exact
evaluator cases, external winning implementations, private traces, Wiki content, and post-R0 source
history remain unavailable to the coding agent.

The policy addresses a common failure mode: repeated local improvements can make the incumbent look
structurally final even though a higher-upside architecture requires several non-promotable episodes
to become feature-complete.

## Implementation

- Development repository: `/data/ai-users/fanzheng/home/reproduce_mudi_trace_with_aka/atrex-kernel-agent`
- Previous mechanism branch: `feat/repository-horizon-v3`
- Previous mechanism commit: `33b446f`
- Branch: `feat/repository-horizon-v4-architecture-escape`
- Architecture-escape implementation commit: `f1eb8e7`
- Private-label isolation hardening commit: `cddcce2`
- Locked source revision: `b54df166ebb69b896892826014759d09b9c3c9c6`
- Editable source remains limited to the manifest-declared `flash_attn/cute` tree.

The previous version is the thin current-main Repository Horizon v3 implementation at `33b446f`.
The next version under test is v4 at `f1eb8e7`, plus the launch-identity hardening at `cddcce2`:
it retains v3's evaluator, workspace, prompt, promotion, and isolation contracts and changes only
how the campaign reacts to architectural stagnation. The hardening commit lets the public campaign
identity be set independently from a private evaluator directory name, so private or historical
labels cannot leak through generated workspace paths. These are implementation commits; document
commits do not change runtime behavior.

V4 keeps the v3 isolation model and current-main supervisor. It adds only:

1. Promotion-stall and periodic architecture-escape triggers. Repository `max-stall` activates an
   escape instead of terminating the campaign.
2. A persistent, generic architecture map built from R0, the bounded pre-R0 corpus, canonical memory,
   and first-principles reasoning.
3. A protected multi-episode commitment budget. Intermediate architecture versions may be slower,
   incomplete, or temporarily unable to run.
4. A distinction between implementation refutation and architecture refutation. Permanent refutation
   requires feature parity, two materially different implementations, and independent review.
5. A supervisor-owned WIP patch carried through `last_trial_commit`, allowing the next isolated
   episode to continue an architecture rewrite without changing the production incumbent.

State is stored outside candidate Git history in:

- `.repository_horizon_runtime/strategy_state.json`
- `.repository_horizon_runtime/architecture_map.json`
- `.repository_horizon_runtime/architecture_wip.patch`

## Previous isolated run

- Mechanism version: Repository Horizon v3 at `33b446f`
- Campaign root: `/data/ai-users/fanzheng/home/reproduce_mudi_trace_with_aka/campaigns/mudi5-v3-private`
- Paused after starting episode 49; latest canonical memory is version 48.
- Preserved incumbent source commit: `54b9e1cbf725188748552a6a68aa71a46324d1d1`

The v3 supervisor and its active coding session were terminated without deleting the incumbent,
episode worktree, journal, canonical memory, or gateway records. This makes the pause auditable and
recoverable while preventing the old optimizer from competing with the new experiment for GPU 4.

## New isolated run

- Campaign label: `fa4-hd256-v4-private`
- Agent-visible operator label: `fa4-hd256-private`
- Campaign root: `/data/ai-users/fanzheng/home/reproduce_mudi_trace_with_aka/campaigns/fa4-hd256-v4-private`
- Launch wrapper: `/data/ai-users/fanzheng/home/reproduce_mudi_trace_with_aka/run_fa4_hd256_v4_private.sh`
- Restart watchdog: `/data/ai-users/fanzheng/home/reproduce_mudi_trace_with_aka/watch_fa4_hd256_v4_private.sh`
- GPU gateway: `http://127.0.0.1:8004`, physically pinned to GPU 4
- Agent backend: Claude through Bailian
- Optional plan reviewer: Codex `gpt-5.6-sol`, effort `high`
- Maximum versions: 100
- Architecture escape: 4 consecutive unpromoted episodes
- Periodic architecture review: every 6 episodes
- Protected architecture commitment: 4 episodes

The launch wrapper retries non-zero supervisor exits after a short delay while preserving the same
campaign workspace. A clean exit at the configured version budget is not restarted.

## Acceptance and isolation

- The first real probe remains R0 even if it cannot execute.
- The first fully correct version is the performance baseline.
- Exact evaluator shapes remain private and are visible only to supervisor verification.
- Wiki/KernelWiki and external solution history are not installed.
- The source corpus remains `replay_strict` and excludes explicitly forbidden post-R0 commits.
- Only complete, correct candidates passing same-allocation ABBA and the promotion threshold can
  replace the incumbent.
- Architecture WIP is never promoted merely because its protected budget exists.

## Next version candidates

V5 should be driven by evidence from this run. Likely additions are:

1. Separate device-kernel and end-to-end stagnation counters, so a host-only promotion cannot reset
   evidence that the kernel architecture is flat.
2. A portfolio scheduler that allocates episodes among multiple architecture theses according to
   measured information gain rather than selecting one indefinitely.
3. Machine-verifiable reviewer attestations instead of the current review artifact/status contract.
4. Anonymous evaluator regime feedback, if it can be exposed without leaking exact private cases.

These are deliberately deferred: V4 first tests whether durable WIP plus valid-refutation rules are
sufficient to prevent premature convergence.
