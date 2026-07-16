import { beforeEach, describe, expect, it, vi } from "vitest";
import {
  type EventBus,
  PROTOCOL_VERSION,
  type RpcDeps,
  registerRpcHandlers,
  type SpawnCapable,
} from "../src/cross-extension-rpc.js";

function createEventBus(): EventBus {
  const listeners = new Map<string, Set<(data: unknown) => void>>();
  return {
    on(event, handler) {
      const handlers = listeners.get(event) ?? new Set();
      handlers.add(handler);
      listeners.set(event, handlers);
      return () => handlers.delete(handler);
    },
    emit(event, data) {
      for (const handler of listeners.get(event) ?? []) handler(data);
    },
  };
}

function request<T>(events: EventBus, channel: string, params: object): Promise<T> {
  const requestId = (params as { requestId: string }).requestId;
  return new Promise((resolve) => {
    const unsub = events.on(`${channel}:reply:${requestId}`, (reply) => {
      unsub();
      resolve(reply as T);
    });
    events.emit(channel, params);
  });
}

describe("cross-extension RPC v3", () => {
  let events: EventBus;
  let manager: SpawnCapable;
  let ctx: RpcDeps["getCtx"] extends () => infer T ? T : never;
  let deps: RpcDeps;

  beforeEach(() => {
    events = createEventBus();
    manager = {
      spawn: vi.fn().mockReturnValue("agent-42"),
      getScopeHandle: vi.fn().mockReturnValue("opaque-owner-handle"),
      cancelScope: vi.fn().mockResolvedValue({ settled: true, failures: [] }),
    };
    ctx = { session: true } as never;
    deps = {
      events,
      pi: { events } as never,
      getCtx: () => ctx,
      resolveType: (type) => ["general-purpose", "Explore", "Plan"].includes(type) ? type : undefined,
      manager,
    };
  });

  it("reports the current protocol version", async () => {
    registerRpcHandlers(deps);
    await expect(request(events, "subagents:rpc:ping", { requestId: "ping" }))
      .resolves.toEqual({ success: true, data: { version: PROTOCOL_VERSION } });
    expect(PROTOCOL_VERSION).toBe(3);
  });

  it("fails loudly when an older facade omits the protocol version", async () => {
    registerRpcHandlers(deps);
    await expect(request(events, "subagents:rpc:spawn", {
      requestId: "old",
      type: "general-purpose",
      prompt: "work",
    })).resolves.toEqual({
      success: false,
      error: "RPC protocol mismatch: expected 3, received missing",
    });
    expect(manager.spawn).not.toHaveBeenCalled();
  });

  it("strictly resolves a named type and returns an opaque owner handle", async () => {
    registerRpcHandlers(deps);
    await expect(request(events, "subagents:rpc:spawn", {
      requestId: "spawn",
      version: PROTOCOL_VERSION,
      type: "Explore",
      prompt: "find it",
      options: { description: "search" },
    })).resolves.toEqual({
      success: true,
      data: { id: "agent-42", handle: "opaque-owner-handle" },
    });
    expect(manager.spawn).toHaveBeenCalledWith(
      deps.pi,
      ctx,
      "Explore",
      "find it",
      { description: "search" },
    );
  });

  it("rejects unknown types instead of falling back to general-purpose", async () => {
    registerRpcHandlers(deps);
    await expect(request(events, "subagents:rpc:spawn", {
      requestId: "missing",
      version: PROTOCOL_VERSION,
      type: "missing-agent",
      prompt: "work",
    })).resolves.toEqual({
      success: false,
      error: "Unknown or disabled agent type: \"missing-agent\"",
    });
    expect(manager.spawn).not.toHaveBeenCalled();
  });

  it("requires an active context", async () => {
    ctx = undefined as never;
    registerRpcHandlers(deps);
    await expect(request(events, "subagents:rpc:spawn", {
      requestId: "no-context",
      version: PROTOCOL_VERSION,
      type: "general-purpose",
      prompt: "work",
    })).resolves.toEqual({ success: false, error: "No active session" });
  });

  it("resolves serializable model overrides", async () => {
    const model = { id: "gpt-5.5", provider: "openai-codex", name: "GPT 5.5" };
    ctx = {
      modelRegistry: {
        find: (provider: string, id: string) => provider === model.provider && id === model.id ? model : null,
        getAll: () => [model],
        getAvailable: () => [model],
      },
    } as never;
    registerRpcHandlers(deps);

    await request(events, "subagents:rpc:spawn", {
      requestId: "model",
      version: PROTOCOL_VERSION,
      type: "general-purpose",
      prompt: "work",
      options: { description: "work", model: "openai-codex/gpt-5.5" },
    });

    expect(manager.spawn).toHaveBeenCalledWith(
      deps.pi,
      ctx,
      "general-purpose",
      "work",
      { description: "work", model },
    );
  });

  it("acknowledges cancellation only after the manager finalizer settles", async () => {
    let settle!: (value: { settled: boolean; failures: string[] }) => void;
    vi.mocked(manager.cancelScope).mockReturnValue(new Promise((resolve) => { settle = resolve; }));
    registerRpcHandlers(deps);

    let replied = false;
    const response = request(events, "subagents:rpc:stop", {
      requestId: "stop",
      version: PROTOCOL_VERSION,
      handle: "opaque-owner-handle",
      reason: "panel cancelled",
    }).then((reply) => { replied = true; return reply; });
    await Promise.resolve();
    expect(replied).toBe(false);

    settle({ settled: true, failures: [] });
    await expect(response).resolves.toEqual({
      success: true,
      data: { settled: true, failures: [] },
    });
    expect(manager.cancelScope).toHaveBeenCalledWith("opaque-owner-handle", "panel cancelled");
  });

  it("reports a bounded cancellation failure", async () => {
    vi.mocked(manager.cancelScope).mockResolvedValue({
      settled: false,
      failures: ["Agent child did not settle before cancellation timeout"],
    });
    registerRpcHandlers(deps);

    await expect(request(events, "subagents:rpc:stop", {
      requestId: "timeout",
      version: PROTOCOL_VERSION,
      handle: "opaque-owner-handle",
    })).resolves.toEqual({
      success: true,
      data: {
        settled: false,
        failures: ["Agent child did not settle before cancellation timeout"],
      },
    });
  });

  it("rejects cancellation by an unscoped agent id", async () => {
    registerRpcHandlers(deps);
    await expect(request(events, "subagents:rpc:stop", {
      requestId: "legacy-stop",
      version: PROTOCOL_VERSION,
      agentId: "agent-42",
    })).resolves.toEqual({ success: false, error: "Ownership scope handle is required" });
  });
});
