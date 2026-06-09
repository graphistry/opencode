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

import type { LanguageModelV3Content, LanguageModelV3FunctionTool } from "@ai-sdk/provider"

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

export * as GemmaToolCode from "./gemma-tool-code"
