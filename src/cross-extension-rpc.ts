/**
 * Cross-extension RPC handlers for the subagents extension.
 *
 * Exposes versioned ping, spawn, and stop RPCs over the pi.events event bus,
 * using per-request scoped reply channels.
 */

import type { ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent";
import type { ScopeCancellationResult, SpawnOptions } from "./agent-manager.js";
import { type ModelRegistry, resolveModel } from "./model-resolver.js";

export interface EventBus {
  on(event: string, handler: (data: unknown) => void): () => void;
  emit(event: string, data: unknown): void;
}

export type RpcReply<T = void> =
  | { success: true; data?: T }
  | { success: false; error: string };

export const PROTOCOL_VERSION = 3;

export interface SpawnCapable {
  spawn(pi: ExtensionAPI, ctx: ExtensionContext, type: string, prompt: string, options: SpawnOptions): string;
  getScopeHandle(id: string): string | undefined;
  cancelScope(handle: string, reason?: unknown): Promise<ScopeCancellationResult | undefined>;
}

export interface RpcDeps {
  events: EventBus;
  pi: ExtensionAPI;
  getCtx: () => ExtensionContext | undefined;
  resolveType: (type: string) => string | undefined;
  manager: SpawnCapable;
}

export interface RpcHandle {
  unsubPing: () => void;
  unsubSpawn: () => void;
  unsubStop: () => void;
}

function handleRpc<P extends { requestId: string }>(
  events: EventBus,
  channel: string,
  fn: (params: P) => unknown | Promise<unknown>,
): () => void {
  return events.on(channel, async (raw: unknown) => {
    const params = raw as P;
    try {
      const data = await fn(params);
      const reply: { success: true; data?: unknown } = { success: true };
      if (data !== undefined) reply.data = data;
      events.emit(`${channel}:reply:${params.requestId}`, reply);
    } catch (error) {
      events.emit(`${channel}:reply:${params.requestId}`, {
        success: false,
        error: error instanceof Error ? error.message : String(error),
      });
    }
  });
}

function assertProtocol(version: unknown): void {
  if (version !== PROTOCOL_VERSION) {
    throw new Error(
      `RPC protocol mismatch: expected ${PROTOCOL_VERSION}, received ${version === undefined ? "missing" : String(version)}`,
    );
  }
}

export function registerRpcHandlers(deps: RpcDeps): RpcHandle {
  const { events, pi, getCtx, resolveType, manager } = deps;

  const unsubPing = handleRpc(events, "subagents:rpc:ping", () => {
    return { version: PROTOCOL_VERSION };
  });

  const unsubSpawn = handleRpc<{
    requestId: string;
    version: number;
    type: string;
    prompt: string;
    options?: SpawnOptions;
  }>(events, "subagents:rpc:spawn", ({ version, type, prompt, options }) => {
    assertProtocol(version);
    const ctx = getCtx();
    if (!ctx) throw new Error("No active session");

    const canonicalType = resolveType(type);
    if (!canonicalType) throw new Error(`Unknown or disabled agent type: "${type}"`);

    let normalizedOptions: SpawnOptions = options ?? { description: type };
    if (typeof normalizedOptions.model === "string") {
      const registry = (ctx as { modelRegistry?: ModelRegistry }).modelRegistry;
      if (!registry) {
        throw new Error(
          `Model override "${normalizedOptions.model}" provided but ctx.modelRegistry is unavailable`,
        );
      }
      const resolved = resolveModel(normalizedOptions.model, registry);
      if (typeof resolved === "string") throw new Error(resolved);
      normalizedOptions = { ...normalizedOptions, model: resolved };
    }

    const id = manager.spawn(pi, ctx, canonicalType, prompt, normalizedOptions);
    const handle = manager.getScopeHandle(id);
    if (!handle) throw new Error("Spawned agent has no ownership scope");
    return { id, handle };
  });

  const unsubStop = handleRpc<{
    requestId: string;
    version: number;
    handle: string;
    reason?: string;
  }>(events, "subagents:rpc:stop", async ({ version, handle, reason }) => {
    assertProtocol(version);
    if (typeof handle !== "string" || handle.length === 0) {
      throw new Error("Ownership scope handle is required");
    }
    const result = await manager.cancelScope(handle, reason);
    if (!result) throw new Error("Ownership scope not found");
    return result;
  });

  return { unsubPing, unsubSpawn, unsubStop };
}
