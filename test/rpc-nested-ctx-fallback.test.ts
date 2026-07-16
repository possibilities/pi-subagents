/**
 * rpc-nested-ctx-fallback.test.ts — pins how the ctx-resolution chain composes
 * with the #142 lifecycle gate. The gate is authoritative for WHO answers: an
 * activation that never saw session_start registers nothing and stays silent.
 * The chain (active scope → activation scope → captured ctx → last stamped
 * ctx) only refines WHICH ctx a bound registration spawns under — e.g. a
 * turn_start refresh — and must never resurrect the filtered-factory zombie.
 */
import { afterEach, describe, expect, it, vi } from "vitest";
import { PROTOCOL_VERSION } from "../src/cross-extension-rpc.js";

vi.mock("../src/agent-runner.js", async () => {
  const actual = await vi.importActual<typeof import("../src/agent-runner.js")>("../src/agent-runner.js");
  return { ...actual, runAgent: vi.fn() };
});

import { runAgent } from "../src/agent-runner.js";
import subagentsExtension from "../src/index.js";

const MANAGER_KEY = Symbol.for("pi-subagents:manager");
const LAST_CTX_KEY = Symbol.for("pi-subagents:last-ctx");

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
  const lifecycle = new Map<string, (event: unknown, ctx: ReturnType<typeof makeCtx>) => Promise<void>>();
  const pi = {
    registerMessageRenderer: vi.fn(),
    registerTool: vi.fn(),
    registerCommand: vi.fn(),
    on: vi.fn((event: string, handler: (event: unknown, ctx: ReturnType<typeof makeCtx>) => Promise<void>) => {
      lifecycle.set(event, handler);
    }),
    events: makeBus(),
    appendEntry: vi.fn(),
    sendMessage: vi.fn(),
  };
  return { pi, lifecycle };
}

function makeCtx(label: string) {
  return {
    label,
    hasUI: false,
    ui: { setStatus: vi.fn(), setWidget: vi.fn(), notify: vi.fn() },
    cwd: process.cwd(),
    model: undefined,
    modelRegistry: { find: vi.fn(), getAvailable: vi.fn(() => []), getAll: vi.fn(() => []) },
    sessionManager: { getSessionId: vi.fn(() => label), getBranch: vi.fn(() => []) },
    getSystemPrompt: vi.fn(() => "parent"),
  };
}

function rpcSpawn(bus: ReturnType<typeof makeBus>, requestId: string) {
  return new Promise<{ success: boolean; data?: { id: string; handle: string }; error?: string }>((resolve) => {
    bus.on(`subagents:rpc:spawn:reply:${requestId}`, (data) => resolve(data as { success: boolean; data?: { id: string; handle: string }; error?: string }));
    bus.emit("subagents:rpc:spawn", {
      requestId,
      version: PROTOCOL_VERSION,
      type: "general-purpose",
      prompt: "go",
      options: { description: "nested spawn" },
    });
  });
}

function rpcSubscriptions(bus: ReturnType<typeof makeBus>) {
  return bus.on.mock.calls.filter(([event]) => String(event).startsWith("subagents:rpc:") && !String(event).includes(":reply:"));
}

const priorManager = (globalThis as Record<PropertyKey, unknown>)[MANAGER_KEY];
const priorCtx = (globalThis as Record<PropertyKey, unknown>)[LAST_CTX_KEY];

afterEach(() => {
  const globals = globalThis as Record<PropertyKey, unknown>;
  if (priorManager === undefined) delete globals[MANAGER_KEY];
  else globals[MANAGER_KEY] = priorManager;
  if (priorCtx === undefined) delete globals[LAST_CTX_KEY];
  else globals[LAST_CTX_KEY] = priorCtx;
  vi.mocked(runAgent).mockReset();
});

describe("nested RPC ctx resolution under the session_start gate", () => {
  it("a lifecycle-less activation registers nothing — only the bound root serves", async () => {
    const globals = globalThis as Record<PropertyKey, unknown>;
    delete globals[MANAGER_KEY];
    delete globals[LAST_CTX_KEY];
    vi.mocked(runAgent).mockResolvedValue({
      responseText: "done",
      session: { dispose: vi.fn() } as never,
      aborted: false,
      steered: false,
    });

    const root = makePi();
    subagentsExtension(root.pi as never);
    // Factory time wires nothing (#142): no handlers, no readiness broadcast.
    expect(rpcSubscriptions(root.pi.events)).toHaveLength(0);
    await root.lifecycle.get("session_start")?.({}, makeCtx("root"));
    expect(rpcSubscriptions(root.pi.events).length).toBeGreaterThan(0);

    const child = makePi();
    subagentsExtension(child.pi as never);
    // A filtered-out/nested activation never reaches session_start: its bus
    // stays fully silent — no handlers to answer, no subagents:ready. The
    // last-ctx fallback must not soften this back into an answering zombie.
    expect(rpcSubscriptions(child.pi.events)).toHaveLength(0);
    expect(child.pi.events.emit.mock.calls.filter(([event]) => event === "subagents:ready")).toHaveLength(0);

    // The bound root still serves spawns through the shared manager.
    const reply = await rpcSpawn(root.pi.events, "root-spawn");
    expect(reply.success).toBe(true);
    expect(reply.data?.handle).toMatch(/^[0-9a-f-]{36}$/);
    await child.lifecycle.get("session_shutdown")?.({}, makeCtx("child"));
    await root.lifecycle.get("session_shutdown")?.({}, makeCtx("root"));
  });

  it("turn_start refreshes the ctx a bound registration spawns under", async () => {
    const globals = globalThis as Record<PropertyKey, unknown>;
    delete globals[MANAGER_KEY];
    delete globals[LAST_CTX_KEY];
    vi.mocked(runAgent).mockResolvedValue({
      responseText: "done",
      session: { dispose: vi.fn() } as never,
      aborted: false,
      steered: false,
    });

    const root = makePi();
    subagentsExtension(root.pi as never);
    await root.lifecycle.get("session_start")?.({}, makeCtx("session"));
    await root.lifecycle.get("turn_start")?.({}, makeCtx("turn"));

    const reply = await rpcSpawn(root.pi.events, "fresh-ctx");
    expect(reply.success).toBe(true);
    // The spawn ran under the turn_start ctx, not the stale session_start one.
    expect((vi.mocked(runAgent).mock.calls[0]?.[0] as { label?: string } | undefined)?.label).toBe("turn");
    await root.lifecycle.get("session_shutdown")?.({}, makeCtx("session"));
  });
});
