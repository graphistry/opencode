import { describe, expect, test } from "bun:test"
import { GemmaToolCode } from "@/session/llm/gemma-tool-code"

const offered = GemmaToolCode.offeredToolsFrom([
  { type: "function", name: "bash" },
  { type: "function", name: "read" },
])

describe("GemmaToolCode.parseToolCodeBlock", () => {
  test("bare shell command -> bash", () => {
    expect(GemmaToolCode.parseToolCodeBlock("echo HELLO", offered)).toEqual([
      { toolName: "bash", input: { command: "echo HELLO", description: expect.any(String) } },
    ])
  })

  test("python-call kwargs -> tool call", () => {
    expect(GemmaToolCode.parseToolCodeBlock('bash(command="ls -la", description="list")', offered)).toEqual([
      { toolName: "bash", input: { command: "ls -la", description: "list" } },
    ])
  })

  test("print() wrapper is stripped", () => {
    expect(GemmaToolCode.parseToolCodeBlock('print(bash(command="pwd"))', offered)).toEqual([
      { toolName: "bash", input: { command: "pwd", description: expect.any(String) } },
    ])
  })

  test("single JSON object with nested arguments", () => {
    const out = GemmaToolCode.parseToolCodeBlock('{"name":"bash","arguments":{"command":"id","description":"d"}}', offered)
    expect(out).toEqual([{ toolName: "bash", input: { command: "id", description: "d" } }])
  })

  test("flat JSON object {tool, ...args}", () => {
    const out = GemmaToolCode.parseToolCodeBlock('{"tool":"bash","command":"whoami"}', offered)
    expect(out).toEqual([{ toolName: "bash", input: { command: "whoami", description: expect.any(String) } }])
  })

  test("JSON array of calls (real Gemma 3 27B Bedrock shape)", () => {
    const out = GemmaToolCode.parseToolCodeBlock('[{"tool": "Bash", "command": "echo HELLO_FROM_TOOL"}]', offered)
    expect(out).toEqual([{ toolName: "bash", input: { command: "echo HELLO_FROM_TOOL", description: expect.any(String) } }])
  })

  test("multiple calls in one array", () => {
    const out = GemmaToolCode.parseToolCodeBlock('[{"tool":"bash","command":"echo a"},{"tool":"bash","command":"echo b"}]', offered)
    expect(out.map((c) => (c.input as any).command)).toEqual(["echo a", "echo b"])
  })

  test("unknown tool name -> no call", () => {
    expect(GemmaToolCode.parseToolCodeBlock('{"tool":"definitely_not_a_tool","x":1}', offered)).toEqual([])
  })

  test("natural-language sentence is not treated as a command", () => {
    expect(GemmaToolCode.parseToolCodeBlock("I will now run the command.", offered)).toEqual([])
  })
})

describe("GemmaToolCode.rewriteContent", () => {
  test("replaces a tool_code text block with a native tool-call part", () => {
    const content = [{ type: "text" as const, text: "```tool_code\necho HELLO_FROM_TOOL\n```" }]
    const { content: out, toolCalls } = GemmaToolCode.rewriteContent(content, offered)
    expect(toolCalls).toBe(1)
    expect(out).toHaveLength(1)
    expect(out[0].type).toBe("tool-call")
    const call = out[0] as { toolName: string; input: string }
    expect(call.toolName).toBe("bash")
    expect(JSON.parse(call.input)).toEqual({ command: "echo HELLO_FROM_TOOL", description: expect.any(String) })
  })

  test("preserves surrounding prose around a tool_code block", () => {
    const content = [
      { type: "text" as const, text: "Let me run it.\n```tool_code\necho hi\n```\nDone." },
    ]
    const { content: out, toolCalls } = GemmaToolCode.rewriteContent(content, offered)
    expect(toolCalls).toBe(1)
    expect(out.map((p) => p.type)).toEqual(["text", "tool-call", "text"])
  })

  test("no tool_code -> content unchanged", () => {
    const content = [{ type: "text" as const, text: "just a plain answer" }]
    const { content: out, toolCalls } = GemmaToolCode.rewriteContent(content, offered)
    expect(toolCalls).toBe(0)
    expect(out).toEqual(content)
  })

  test("no offered tools -> content unchanged", () => {
    const content = [{ type: "text" as const, text: "```tool_code\necho hi\n```" }]
    const { content: out, toolCalls } = GemmaToolCode.rewriteContent(content, [])
    expect(toolCalls).toBe(0)
    expect(out).toEqual(content)
  })
})
