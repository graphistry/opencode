// Gemma `tool_code` → native tool-call emulation.
//
// Gemma 3 (incl. on Amazon Bedrock via the Converse `bedrock-runtime` endpoint)
// has NO native tool-use tokens. Per Google's model card, function calling is
// done purely by prompting: the model emits its intended call as TEXT inside a
//   ```tool_code
//   ...
//   ```
// fenced block (Python-call syntax, or — when the agent's tools map to shell —
// a bare command). Bedrock accepts opencode's Converse `toolConfig` but Gemma
// ignores it, so opencode/ai-sdk never sees a native tool-call and the command
// is dropped (tool_use_count = 0). See AWS Gemma 3 model cards (which recommend
// the `bedrock-mantle` OpenAI-compatible endpoint over Converse) and
// https://ai.google.dev/gemma/docs/capabilities/function-calling.
//
// This module parses those `tool_code` blocks back into real tool calls so the
// rest of opencode's tool pipeline runs unchanged. It is intentionally narrow:
// it only fires for models flagged as needing it, only touches text content,
// and only emits calls whose tool name is actually offered for the request.

import type {
  LanguageModelV3Content,
  LanguageModelV3FunctionTool,
  LanguageModelV3Message,
  LanguageModelV3Prompt,
  LanguageModelV3ToolResultOutput,
} from "@ai-sdk/provider"

export type OfferedTool = {
  readonly name: string
  // The single "primary" string parameter to fill when Gemma emits a bare
  // command (e.g. `bash` -> "command"). Undefined when the tool has no obvious
  // single string field.
  readonly bareArg?: string
}

const FENCE_RE = /```(?:tool_code|python|tool_call|json)?\s*\n?([\s\S]*?)```/gi

// Tools whose single primary string argument should receive a bare command/text
// when Gemma emits one without kwargs. Conservative allow-list of opencode's
// shell-shaped tools.
const BARE_ARG_BY_TOOL: Record<string, string> = {
  bash: "command",
  shell: "command",
}

export function offeredToolsFrom(
  tools: ReadonlyArray<LanguageModelV3FunctionTool | { type: string; name?: string; inputSchema?: unknown }> | undefined,
): OfferedTool[] {
  if (!tools) return []
  const out: OfferedTool[] = []
  for (const t of tools) {
    if (!t || (t as any).type !== "function") continue
    const name = (t as LanguageModelV3FunctionTool).name
    if (!name) continue
    out.push({ name, bareArg: bareArgFor(name, (t as LanguageModelV3FunctionTool).inputSchema) })
  }
  return out
}

function bareArgFor(name: string, schema: unknown): string | undefined {
  const known = BARE_ARG_BY_TOOL[name.toLowerCase()]
  if (known) return known
  // Otherwise: a tool with exactly one required string property can take a bare value.
  const s = schema as { properties?: Record<string, { type?: string }>; required?: string[] } | undefined
  const props = s?.properties
  if (!props) return undefined
  const stringKeys = Object.keys(props).filter((k) => props[k]?.type === "string")
  if (stringKeys.length === 1) return stringKeys[0]
  const required = s?.required
  if (required && required.length === 1 && props[required[0]]?.type === "string") return required[0]
  return undefined
}

export type ParsedToolCall = {
  readonly toolName: string
  readonly input: Record<string, unknown>
}

const DEFAULT_DESC = "Run command (recovered from Gemma tool_code)"

// Parse the inner text of a single tool_code block into one or more tool calls,
// given the offered tools. Returns [] when nothing maps cleanly. Gemma emits a
// few shapes in the wild (it has no canonical tool format): Python-call syntax,
// a single JSON object, and a JSON array of {tool, ...args} objects.
export function parseToolCodeBlock(raw: string, offered: ReadonlyArray<OfferedTool>): ParsedToolCall[] {
  const byName = new Map(offered.map((t) => [t.name.toLowerCase(), t]))
  let text = raw.trim()
  if (!text) return []

  // Strip a leading `print(...)` wrapper Gemma sometimes adds.
  const printWrap = text.match(/^print\(\s*([\s\S]*)\)\s*$/)
  if (printWrap) text = printWrap[1].trim()

  // 1) JSON form: a single object or an array of call objects. Each may use
  //    {"name"|"tool"|"function": X, "arguments"|"args"|"parameters": {...}} or
  //    flatten the args as siblings: {"tool": "bash", "command": "..."}.
  if (text.startsWith("{") || text.startsWith("[")) {
    try {
      const parsed = JSON.parse(text)
      const items = Array.isArray(parsed) ? parsed : [parsed]
      const calls: ParsedToolCall[] = []
      for (const obj of items) {
        const call = coerceJsonCall(obj, byName)
        if (call) calls.push(call)
      }
      if (calls.length) return calls
    } catch {
      // fall through
    }
  }

  // 2) Python-call form: tool_name(arg=value, arg2="...") possibly multi-line.
  const call = text.match(/^([A-Za-z_][A-Za-z0-9_.]*)\s*\(([\s\S]*)\)\s*;?\s*$/)
  if (call) {
    const fnRaw = call[1]
    const fn = fnRaw.includes(".") ? fnRaw.split(".").pop()! : fnRaw
    const t = byName.get(fn.toLowerCase())
    if (t) {
      const input = parseKwargs(call[2])
      if (input) return [{ toolName: t.name, input: withRequiredDesc(t.name, input) }]
    }
  }

  // 3) Bare command form: the whole block is a shell command. Route to a shell
  // tool if one is offered.
  const shell = byName.get("bash") ?? byName.get("shell")
  if (shell && shell.bareArg && looksLikeShellCommand(text)) {
    const input: Record<string, unknown> = { [shell.bareArg]: text }
    return [{ toolName: shell.name, input: withRequiredDesc(shell.name, input) }]
  }

  return []
}

// Coerce one JSON object into a tool call. Handles both nested-args and
// flat-sibling-args shapes.
function coerceJsonCall(
  obj: unknown,
  byName: Map<string, OfferedTool>,
): ParsedToolCall | undefined {
  if (!obj || typeof obj !== "object") return undefined
  const o = obj as Record<string, unknown>
  const nameRaw = o["name"] ?? o["tool"] ?? o["tool_name"] ?? o["function"]
  if (typeof nameRaw !== "string") return undefined
  const t = byName.get(nameRaw.toLowerCase())
  if (!t) return undefined
  const nested = o["arguments"] ?? o["args"] ?? o["parameters"] ?? o["input"]
  let input: Record<string, unknown>
  if (nested && typeof nested === "object" && !Array.isArray(nested)) {
    input = { ...(nested as Record<string, unknown>) }
  } else {
    // Flat shape: args are siblings of the name key. Drop the name fields.
    input = {}
    for (const [k, v] of Object.entries(o)) {
      if (k === "name" || k === "tool" || k === "tool_name" || k === "function") continue
      input[k] = v
    }
  }
  return { toolName: t.name, input: withRequiredDesc(t.name, input) }
}

// opencode's bash/shell tool requires a non-empty `description`; supply one when
// Gemma omits it so schema validation passes.
function withRequiredDesc(toolName: string, input: Record<string, unknown>): Record<string, unknown> {
  const lc = toolName.toLowerCase()
  if ((lc === "bash" || lc === "shell") && typeof input["command"] === "string" && !input["description"]) {
    return { ...input, description: DEFAULT_DESC }
  }
  return input
}

function looksLikeShellCommand(text: string): boolean {
  // Accept things that look like commands; reject obvious natural-language prose.
  // The block was explicitly fenced as tool_code so we are fairly lenient, but
  // text that reads like an English sentence (ends in . ? !, several words, and
  // carries no shell metacharacters) is almost certainly a note, not a command.
  if (text.includes("\n\n")) return false
  if (!/^[\w./~$-]/.test(text)) return false
  const hasShellSignal = /[|/=><&;${}*]|--|\s-\w|\bhttps?:\/\//.test(text)
  const endsLikeSentence = /[.!?]$/.test(text)
  const wordCount = text.split(/\s+/).length
  if (endsLikeSentence && wordCount >= 3 && !hasShellSignal) return false
  return true
}

// Minimal Python-kwargs parser: handles key=value pairs with string / number /
// bool / None values. Tolerant of nested quotes via a small scanner.
function parseKwargs(s: string): Record<string, unknown> | undefined {
  const out: Record<string, unknown> = {}
  let i = 0
  const n = s.length
  const skipWs = () => {
    while (i < n && /\s/.test(s[i])) i++
  }
  while (i < n) {
    skipWs()
    if (i >= n) break
    // key
    const keyMatch = /^[A-Za-z_][A-Za-z0-9_]*/.exec(s.slice(i))
    if (!keyMatch) {
      // positional or unparseable -> bail so caller can try bare-command path
      return Object.keys(out).length ? out : undefined
    }
    const key = keyMatch[0]
    i += key.length
    skipWs()
    if (s[i] !== "=") return Object.keys(out).length ? out : undefined
    i++
    skipWs()
    const val = readValue(s, i)
    if (!val) return Object.keys(out).length ? out : undefined
    out[key] = val.value
    i = val.next
    skipWs()
    if (s[i] === ",") i++
  }
  return out
}

function readValue(s: string, i: number): { value: unknown; next: number } | undefined {
  const n = s.length
  const ch = s[i]
  if (ch === '"' || ch === "'") {
    const quote = ch
    i++
    let buf = ""
    while (i < n) {
      const c = s[i]
      if (c === "\\" && i + 1 < n) {
        const next = s[i + 1]
        buf += next === "n" ? "\n" : next === "t" ? "\t" : next
        i += 2
        continue
      }
      if (c === quote) {
        i++
        return { value: buf, next: i }
      }
      buf += c
      i++
    }
    return undefined
  }
  // unquoted token until , or end
  let j = i
  let depth = 0
  while (j < n) {
    const c = s[j]
    if (c === "[" || c === "{" || c === "(") depth++
    else if (c === "]" || c === "}" || c === ")") depth--
    else if (c === "," && depth === 0) break
    j++
  }
  const token = s.slice(i, j).trim()
  if (token === "") return undefined
  if (token === "True") return { value: true, next: j }
  if (token === "False") return { value: false, next: j }
  if (token === "None" || token === "null") return { value: null, next: j }
  const num = Number(token)
  if (!Number.isNaN(num) && /^-?\d/.test(token)) return { value: num, next: j }
  return { value: token, next: j }
}

let counter = 0
function nextCallId(): string {
  counter = (counter + 1) % 1_000_000
  return `gemma_tc_${Date.now().toString(36)}_${counter}`
}

// Rewrite a doGenerate content array: replace tool_code text blocks with native
// tool-call parts. Returns the new content plus whether any tool call was
// extracted (so the caller can flip finishReason to "tool-calls").
export function rewriteContent(
  content: ReadonlyArray<LanguageModelV3Content>,
  offered: ReadonlyArray<OfferedTool>,
): { content: LanguageModelV3Content[]; toolCalls: number } {
  if (offered.length === 0) return { content: [...content], toolCalls: 0 }
  const out: LanguageModelV3Content[] = []
  let toolCalls = 0

  for (const part of content) {
    if (part.type !== "text" || !part.text || !part.text.includes("```")) {
      out.push(part)
      continue
    }
    const text = part.text
    let lastIndex = 0
    let matched = false
    FENCE_RE.lastIndex = 0
    let m: RegExpExecArray | null
    while ((m = FENCE_RE.exec(text)) !== null) {
      const inner = m[1]
      const parsedCalls = parseToolCodeBlock(inner, offered)
      if (parsedCalls.length === 0) continue
      matched = true
      const before = text.slice(lastIndex, m.index)
      if (before.trim()) out.push({ type: "text", text: before })
      for (const parsed of parsedCalls) {
        out.push({
          type: "tool-call",
          toolCallId: nextCallId(),
          toolName: parsed.toolName,
          input: JSON.stringify(parsed.input),
        })
        toolCalls++
      }
      lastIndex = FENCE_RE.lastIndex
    }
    if (!matched) {
      out.push(part)
      continue
    }
    const after = text.slice(lastIndex)
    if (after.trim()) out.push({ type: "text", text: after })
  }

  return { content: out, toolCalls }
}

// Re-serialize a prior tool call the Gemma way: a ```tool_code``` block holding
// the bare command (for shell tools) or a `name(kwargs)` Python call. This is the
// inverse of parseToolCodeBlock above — the model originally emitted exactly this
// shape, so replaying it as text keeps the transcript in Gemma's own idiom.
function renderToolCallAsText(toolName: string, input: unknown): string {
  const obj = input && typeof input === "object" ? (input as Record<string, unknown>) : {}
  const bare = BARE_ARG_BY_TOOL[toolName.toLowerCase()]
  const command = bare ? obj[bare] : undefined
  if (typeof command === "string") {
    return "```tool_code\n" + command + "\n```"
  }
  // Non-shell tool: render as a Python-style call so a round-trip through
  // parseToolCodeBlock would recover the same arguments.
  const args = Object.entries(obj)
    .filter(([k]) => k !== "description")
    .map(([k, v]) => `${k}=${JSON.stringify(v)}`)
    .join(", ")
  return "```tool_code\n" + `${toolName}(${args})` + "\n```"
}

// Flatten an ai-sdk tool-result output into the plain text Gemma understands.
function renderToolOutputAsText(output: LanguageModelV3ToolResultOutput): string {
  switch (output.type) {
    case "text":
    case "error-text":
      return output.value
    case "json":
    case "error-json":
      return JSON.stringify(output.value)
    case "execution-denied":
      return output.reason ?? "Tool execution denied."
    case "content":
      return output.value
        .map((part) => (part.type === "text" ? part.text : `[${part.type}]`))
        .join("\n")
    default:
      return ""
  }
}

// Rewrite a replayed V3 prompt so a Gemma turn never carries native tool blocks.
//
// Gemma has no native tool channel (see header). On Bedrock the request still
// converts cleanly to Converse — a recovered tool-call becomes an `assistant`
// `toolUse` block and its result a `user` `toolResult` block — but the Gemma
// serverless endpoint re-templates Converse into its OWN chat format, where a
// `toolUse`-only assistant turn renders to no text and collapses. The surviving
// turns then read user/user, so Gemma's chat template rejects the request with
// "Conversation roles must alternate user/assistant/user/assistant/...".
//
// The fix mirrors how Gemma emits calls in the first place: render each prior
// tool-call as assistant TEXT holding a ```tool_code``` block, and each tool
// result as USER text holding a ```tool_output``` block. Every turn then carries
// real text and the user/assistant alternation holds. Consecutive same-role
// messages produced by this rewrite (e.g. an assistant text part next to a
// rewritten tool-call) are merged so the alternation is exact. Narrow by design:
// only the assistant tool-call and tool-result parts change; text, file, system,
// and reasoning parts pass through untouched. Gated by the caller (isGemma +
// OPENCODE_DISABLE_GEMMA_TOOLCODE off).
export function rewritePromptForGemma(prompt: LanguageModelV3Prompt): LanguageModelV3Prompt {
  const lowered: LanguageModelV3Message[] = []
  for (const msg of prompt) {
    if (msg.role === "assistant") {
      const parts: Array<{ type: "text"; text: string }> = []
      for (const part of msg.content) {
        switch (part.type) {
          case "tool-call":
            parts.push({ type: "text", text: renderToolCallAsText(part.toolName, part.input) })
            break
          case "text":
          case "reasoning":
            if (part.text.trim()) parts.push({ type: "text", text: part.text })
            break
          // Drop tool-result parts that occasionally ride on assistant messages;
          // their content is replayed via the dedicated tool message below.
          default:
            break
        }
      }
      lowered.push({ role: "assistant", content: parts.length ? parts : [{ type: "text", text: "" }] })
      continue
    }
    if (msg.role === "tool") {
      const text = msg.content
        .map((part) => (part.type === "tool-result" ? renderToolOutputAsText(part.output) : ""))
        .filter(Boolean)
        .join("\n")
      // A tool result is the user's turn for Gemma (it follows an assistant call).
      lowered.push({ role: "user", content: [{ type: "text", text: "```tool_output\n" + text + "\n```" }] })
      continue
    }
    lowered.push(msg)
  }
  return mergeAdjacentRoles(lowered)
}

// Merge consecutive messages that share a role (other than system) into one, so
// the lowering above never produces two assistant or two user turns in a row.
function mergeAdjacentRoles(msgs: LanguageModelV3Message[]): LanguageModelV3Message[] {
  const out: LanguageModelV3Message[] = []
  for (const msg of msgs) {
    const prev = out[out.length - 1]
    if (
      prev &&
      prev.role === msg.role &&
      (msg.role === "user" || msg.role === "assistant") &&
      Array.isArray(prev.content) &&
      Array.isArray(msg.content)
    ) {
      ;(prev.content as unknown[]).push(...(msg.content as unknown[]))
      continue
    }
    out.push(msg)
  }
  return out
}

export * as GemmaToolCode from "./gemma-tool-code"
