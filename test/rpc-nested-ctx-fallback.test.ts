/**
 * rpc-nested-ctx-fallback.test.ts — the cross-extension RPC spawn handler's
 * ExtensionContext resolution across multiple activations in one process.
 *
 * Subagent sessions re-activate this extension in the same process
 * (session.bindExtensions in agent-runner.ts), but lifecycle events
 * (session_start / turn_start) never reach the child activation — so its
 * per-activation `currentCtx` stays undefined forever. Before the fix, any
 * cross-extension RPC spawn answered by a child activation (e.g. an
 * orchestrator subagent spawning a sub-subagent through another extension's
 * tool) was refused with "No active session" even while the root session was
 * actively running turns.
 *
 * The fix mirrors the Symbol.for manager-registry pattern: activations that DO
 * receive lifecycle events publish the last-known ctx under
 * Symbol.for("pi-subagents:last-ctx"), and getCtx falls back to it when the
 * answering activation never captured one of its own.
 */
import { afterEach, describe, expect, it, vi } from "vitest";

vi.mock("../src/agent-runner.js", async () => {
  const actual = await vi.importActual<typeof import("../src/agent-runner.js")>("../src/agent-runner.js");
  return { ...actual, runAgent: vi.fn() };
});

import { runAgent } from "../src/agent-runner.js";
import subagentsExtension from "../src/index.js";

const MANAGER_KEY = Symbol.for("pi-subagents:manager");
const LAST_CTX_KEY = Symbol.for("pi-subagents:last-ctx");

/** A minimal working event bus — each activation gets its own, mirroring the
 *  per-resource-loader buses subagent sessions really run on. */
function makeBus() {
  const handlers = new Map<string, Set<(data: unknown) => void>>();
  return {
    on: vi.fn((event: string, handler: (data: unknown) => void) => {
      const set = handlers.get(event) ?? new Set();
      set.add(handler);
      handlers.set(event, set);
      return () => set.delete(handler);
    }),
    emit: vi.fn((event: string, data: unknown) => {
      for (const handler of [...(handlers.get(event) ?? [])]) handler(data);
    }),
  };
}

function makePi() {
  const tools = new Map<string, any>();
  const lifecycle = new Map<string, any>();
  const pi = {
    registerMessageRenderer: vi.fn(),
    registerTool: vi.fn((t: any) => tools.set(t.name, t)),
    registerCommand: vi.fn(),
    on: vi.fn((event: string, handler: any) => lifecycle.set(event, handler)),
    events: makeBus(),
    appendEntry: vi.fn(),
    sendMessage: vi.fn(),
  } as any;
  return { pi, tools, lifecycle };
}

function ctx() {
  return {
    hasUI: false,
    ui: { setStatus: vi.fn(), setWidget: vi.fn(), notify: vi.fn() },
    cwd: process.cwd(),
    model: undefined,
    modelRegistry: { find: vi.fn(), getAvailable: vi.fn(() => []) },
    sessionManager: { getSessionId: vi.fn(() => "s1"), getBranch: vi.fn(() => []) },
    getSystemPrompt: vi.fn(() => "parent"),
  } as any;
}

/** Emit a spawn RPC on the given bus and return the reply envelope. */
function rpcSpawn(bus: ReturnType<typeof makeBus>, requestId: string): Promise<any> {
  return new Promise((resolve) => {
    bus.on(`subagents:rpc:spawn:reply:${requestId}`, resolve);
    bus.emit("subagents:rpc:spawn", {
      requestId,
      type: "general-purpose",
      prompt: "go",
      options: { description: "nested spawn" },
    });
  });
}

// Restore the global slots around every test.
const priorManager = (globalThis as any)[MANAGER_KEY];
const priorCtx = (globalThis as any)[LAST_CTX_KEY];
afterEach(() => {
  if (priorManager === undefined) delete (globalThis as any)[MANAGER_KEY];
  else (globalThis as any)[MANAGER_KEY] = priorManager;
  if (priorCtx === undefined) delete (globalThis as any)[LAST_CTX_KEY];
  else (globalThis as any)[LAST_CTX_KEY] = priorCtx;
  vi.mocked(runAgent).mockReset();
});

describe("cross-extension RPC spawn ctx fallback across activations", () => {
  it("a child activation that never saw lifecycle events spawns via the last-known ctx", async () => {
    delete (globalThis as any)[MANAGER_KEY];
    delete (globalThis as any)[LAST_CTX_KEY];
    vi.mocked(runAgent).mockImplementation(() => new Promise(() => {}) as any); // never resolves

    // Root session activates and runs: session_start delivers its ctx.
    const root = makePi();
    subagentsExtension(root.pi);
    await root.lifecycle.get("session_start")?.({ type: "session_start", reason: "startup" }, ctx());

    // A child (subagent) session re-activates the extension on its own bus.
    // Deliberately fire NO lifecycle events on it — that is the real-world
    // condition for subagent sessions.
    const child = makePi();
    subagentsExtension(child.pi);

    const reply = await rpcSpawn(child.pi.events, "req-1");
    expect(reply.success).toBe(true);
    expect(reply.data.id).toBeDefined();
  });

  it("turn_start keeps the fallback fresh when session_start never fired", async () => {
    delete (globalThis as any)[MANAGER_KEY];
    delete (globalThis as any)[LAST_CTX_KEY];
    vi.mocked(runAgent).mockImplementation(() => new Promise(() => {}) as any);

    // The root only ever sees turn_start (e.g. its session_start predates this
    // activation) — the fallback must still populate.
    const root = makePi();
    subagentsExtension(root.pi);
    await root.lifecycle.get("turn_start")?.({ type: "turn_start" }, ctx());

    const child = makePi();
    subagentsExtension(child.pi);

    const reply = await rpcSpawn(child.pi.events, "req-2");
    expect(reply.success).toBe(true);
  });

  it("with no ctx anywhere the spawn still fails loud with 'No active session'", async () => {
    delete (globalThis as any)[MANAGER_KEY];
    delete (globalThis as any)[LAST_CTX_KEY];

    const lone = makePi();
    subagentsExtension(lone.pi);

    const reply = await rpcSpawn(lone.pi.events, "req-3");
    expect(reply.success).toBe(false);
    expect(reply.error).toBe("No active session");
  });
});
