# graphistry/opencode — benchmark build manifest

This fork carries small patches that make Amazon Bedrock work reliably under opencode's
**headless** `run` (`opencode run --format json`) for benchmarking. Build experimental
runs from an **immutable tag** (e.g. `bench-bedrock-v1`), never a moving branch, so runs
are reproducible. Record the build identity in every result (see Provenance below).

## Tags

### bench-bedrock-v3
- Upstream base: `anomalyco/opencode @ 07808be` (branch `dev`)
- Patches (each a separate commit):
  - `fix(cli): await event-drain loop in non-interactive run` → upstream `anomalyco/opencode#29132`/#31389
  - `feat(bedrock): honor streaming:false for tool use ...`    → upstream `anomalyco/opencode#31357`
  - `feat(bedrock): recover Gemma tool_code text blocks into native tool calls` (v2)
  - `fix(bedrock): serialize Gemma tool history as text so multi-turn Converse roles alternate` (NEW in v3) —
    Gemma has no native tool channel, so replaying a recovered tool-call as a native Converse `toolUse`/`toolResult`
    block makes the Gemma serverless endpoint's chat template collapse the `toolUse`-only assistant turn, leaving
    `user/user` and a 400 `"Conversation roles must alternate ..."` after the FIRST tool call. Adds
    `GemmaToolCode.rewritePromptForGemma` (called from the Gemma `transformParams` path, same gating) which re-renders
    prior tool-calls as assistant ` ```tool_code ` text and tool results as user ` ```tool_output ` text, then merges
    adjacent same-role turns so alternation is exact. Gemma-gated; Claude/Nova/Llama/Nemotron untouched.
- Toolchain: `bun 1.3.14`
- Validated 2026-06-08/09 on real Bedrock: Gemma multi-turn 2-turn (`echo first`→`echo second`) and 4-command
  (`alpha/bravo/charlie/delta`) runs complete with NO role-alternation 400 and all tool calls executed in order;
  `typecheck` (tsgo --noEmit) exit 0; `gemma-tool-code.test.ts` 18/18 (12 + 6 new). Supersedes v2 for any
  multi-turn tool task (v2 only works for a single tool call).

### bench-bedrock-v2
- Upstream base: `anomalyco/opencode @ 07808be` (branch `dev`)
- Patches (each a separate commit):
  - `fix(cli): await event-drain loop in non-interactive run` → upstream `anomalyco/opencode#29132`/#31389
  - `feat(bedrock): honor streaming:false for tool use ...`    → upstream `anomalyco/opencode#31357`
  - `feat(bedrock): recover Gemma tool_code text blocks into native tool calls` (NEW in v2) — Gemma 3 has
    no native tool-use tokens; routes Gemma via non-streaming `doGenerate` + a `wrapGenerate` middleware that
    parses ` ```tool_code ` blocks into native tool calls. Gemma-gated; Claude/Nova/Llama/Nemotron untouched.
- Toolchain: `bun 1.3.14`
- Validated 2026-06-08: Gemma tool_use_count 0→3/3 on real Bedrock; Sonnet/Nemotron native tool use intact;
  `llm.test.ts` 26/26 + new `gemma-tool-code.test.ts`.

### bench-bedrock-v1
- Upstream base: `anomalyco/opencode @ 07808be` (branch `dev`)
- Patches (each a separate commit, attributable to an upstream PR):
  - `fix(cli): await event-drain loop in non-interactive run` → upstream `anomalyco/opencode#29132`
  - `feat(bedrock): honor streaming:false for tool use ...`    → upstream `anomalyco/opencode#31357` (+ Bedrock message-transform guard)
- Toolchain: `bun 1.3.14`
- Validated 2026-06-08 on ARM64 (DGX): Sonnet 4.5 9/9, Nova Pro 6/6, Nemotron Super 120B 6/6,
  Gemma 3 27B 6/6 (all `129`, deterministic); Llama 4 Maverick answers with tools via `streaming:false`.

## Build (from source, wrapper — no compile needed)
```
bun install
# wrapper script (point BENCH_OPENCODE_BIN at it):
#   exec bun run --conditions=node "<this-repo>/packages/opencode/src/index.ts" "$@"
```

## Config (harness `BENCH_OPENCODE_CONFIG_STYLE=v116`)
- provider key `amazon-bedrock`, model field `id`, no `npm`
- top-level `model` + `small_model` (accessible model, e.g. `sonnet-4-5`) to avoid the inaccessible
  default-haiku title-gen call
- Llama 4: `options.streaming=false` + `limit.output ≤ 8192`
- Nova / Nemotron / Gemma: `limit.output ≤ 8192`

## Provenance (record in every run record)
- `opencode_source`     = `graphistry/opencode` (from `git remote get-url origin`)
- `opencode_ref`        = `git describe --tags --always --dirty` (e.g. `bench-bedrock-v1`; a `-dirty`
  suffix means the tree was modified and the run is NOT reproducible)
- `opencode_sha`        = full HEAD sha
- `opencode_upstream_base` = `anomalyco/opencode@07808be`

Note: three opencode repos exist — `anomalyco/opencode` (upstream), `opencode-ai/opencode` (a
different/older project), and this fork. Always record the full `org/repo`, never just `opencode`.
