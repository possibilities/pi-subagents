/**
 * agent-manager.ts — Tracks agents, background execution, resume support.
 *
 * Background agents are subject to a configurable concurrency limit (default: 4).
 * Excess agents are queued and auto-started as running agents complete.
 * Foreground agents bypass the queue (they block the parent anyway).
 */

import { AsyncLocalStorage } from "node:async_hooks";
import { randomUUID } from "node:crypto";
import { statSync } from "node:fs";
import { isAbsolute } from "node:path";
import type { Model } from "@earendil-works/pi-ai";
import type { AgentSession, ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent";
import { resumeAgent, runAgent, type ToolActivity } from "./agent-runner.js";
import type { AgentInvocation, AgentRecord, IsolationMode, SubagentType, ThinkingLevel } from "./types.js";
import { addUsage } from "./usage.js";
import { cleanupWorktree, createWorktree, pruneWorktrees, } from "./worktree.js";

export type OnAgentComplete = (record: AgentRecord) => void;
export type OnAgentStart = (record: AgentRecord) => void;
export type OnAgentCompact = (record: AgentRecord, info: CompactionInfo) => void;
export type CompactionInfo = { reason: "manual" | "threshold" | "overflow"; tokensBefore: number };

/** Default max concurrent background agents. */
const DEFAULT_MAX_CONCURRENT = 4;
const DEFAULT_CANCEL_TIMEOUT_MS = 5_000;

export interface ScopeCancellationResult {
  settled: boolean;
  failures: string[];
}

interface OwnershipScope {
  handle: string;
  agentId: string;
  manager: AgentManager;
  parent?: string;
  children: Set<string>;
  ctx: ExtensionContext;
  cancelReason?: unknown;
  cancelPromise?: Promise<ScopeCancellationResult>;
}

interface OwnershipRuntime {
  storage: AsyncLocalStorage<string>;
  scopes: Map<string, OwnershipScope>;
}

const OWNERSHIP_RUNTIME_KEY = Symbol.for("pi-subagents:ownership-runtime");
const ownershipRuntime = (() => {
  const globals = globalThis as typeof globalThis & { [OWNERSHIP_RUNTIME_KEY]?: OwnershipRuntime };
  globals[OWNERSHIP_RUNTIME_KEY] ??= {
    storage: new AsyncLocalStorage<string>(),
    scopes: new Map<string, OwnershipScope>(),
  };
  return globals[OWNERSHIP_RUNTIME_KEY];
})();

export function getActiveScopeContext(): ExtensionContext | undefined {
  const handle = ownershipRuntime.storage.getStore();
  return handle ? ownershipRuntime.scopes.get(handle)?.ctx : undefined;
}

export function captureActiveScopeContext(): () => ExtensionContext | undefined {
  const handle = ownershipRuntime.storage.getStore();
  return () => handle ? ownershipRuntime.scopes.get(handle)?.ctx : undefined;
}

export function setActiveScopeContext(ctx: ExtensionContext): void {
  const handle = ownershipRuntime.storage.getStore();
  const scope = handle ? ownershipRuntime.scopes.get(handle) : undefined;
  if (scope) scope.ctx = ctx;
}

export function getOwnedAgentRecord(id: string): AgentRecord | undefined {
  for (const scope of ownershipRuntime.scopes.values()) {
    if (scope.agentId === id) return scope.manager.getRecord(id);
  }
  return undefined;
}

export function abortOwnedAgent(id: string, reason?: unknown): boolean {
  for (const scope of ownershipRuntime.scopes.values()) {
    if (scope.agentId === id) return scope.manager.abort(id, reason);
  }
  return false;
}

function waitWithin<T>(promise: Promise<T>, deadline: number): Promise<boolean> {
  const remaining = Math.max(0, deadline - Date.now());
  return new Promise((resolve) => {
    const timer = setTimeout(() => resolve(false), remaining);
    timer.unref();
    promise.then(
      () => { clearTimeout(timer); resolve(true); },
      () => { clearTimeout(timer); resolve(true); },
    );
  });
}

/**
 * Validate a caller-supplied SpawnOptions.cwd. `undefined`/`null` mean "unset"
 * (parent cwd). Anything else must be an absolute path to an existing
 * directory — curated errors instead of TypeErrors from path/fs internals
 * (RPC callers send arbitrary JSON: null, numbers, file paths).
 */
function assertValidSpawnCwd(cwd: unknown): asserts cwd is string | undefined | null {
  if (cwd == null) return;
  if (typeof cwd !== "string" || !isAbsolute(cwd)) {
    throw new Error(`SpawnOptions.cwd must be an absolute path: "${String(cwd)}"`);
  }
  let isDirectory = false;
  try {
    isDirectory = statSync(cwd).isDirectory();
  } catch {
    throw new Error(`SpawnOptions.cwd does not exist: "${cwd}"`);
  }
  if (!isDirectory) {
    throw new Error(`SpawnOptions.cwd is not a directory: "${cwd}"`);
  }
}

interface SpawnArgs {
  pi: ExtensionAPI;
  ctx: ExtensionContext;
  type: SubagentType;
  prompt: string;
  options: SpawnOptions;
}

export interface SpawnOptions {
  description: string;
  model?: Model<any>;
  maxTurns?: number;
  isolated?: boolean;
  inheritContext?: boolean;
  thinkingLevel?: ThinkingLevel;
  isBackground?: boolean;
  /**
   * Skip the maxConcurrent queue check for this spawn — start immediately even
   * if the configured concurrency limit would otherwise queue it. Used by the
   * scheduler so a fired job can't be deferred past its trigger window.
   */
  bypassQueue?: boolean;
  /** Isolation mode — "worktree" creates a temp git worktree for the agent. */
  isolation?: IsolationMode;
  /**
   * Working directory for the agent (absolute path). Default: parent session
   * cwd. The agent's tools operate here, but .pi config (extensions, skills,
   * settings, memory) still loads from the parent session's project — the
   * target directory's `.pi` extensions never execute. With isolation:
   * "worktree", the worktree is created FROM this directory and the result
   * branch lands in that repo.
   */
  cwd?: string;
  /** Resolved invocation snapshot captured for UI display. */
  invocation?: AgentInvocation;
  /** Parent abort signal — when aborted, the subagent is also stopped. */
  signal?: AbortSignal;
  /** Called on tool start/end with activity info (for streaming progress to UI). */
  onToolActivity?: (activity: ToolActivity) => void;
  /** Called on streaming text deltas from the assistant response. */
  onTextDelta?: (delta: string, fullText: string) => void;
  /** Called when the agent session is created (for accessing session stats). */
  onSessionCreated?: (session: AgentSession) => void;
  /** Called at the end of each agentic turn with the cumulative count. */
  onTurnEnd?: (turnCount: number) => void;
  /** Called once per assistant message_end with that message's usage delta. */
  onAssistantUsage?: (usage: { input: number; output: number; cacheWrite: number }) => void;
  /** Called when the session successfully compacts. */
  onCompaction?: (info: CompactionInfo) => void;
}

export class AgentManager {
  private agents = new Map<string, AgentRecord>();
  private cleanupInterval: ReturnType<typeof setInterval>;
  private onComplete?: OnAgentComplete;
  private onStart?: OnAgentStart;
  private onCompact?: OnAgentCompact;
  private maxConcurrent: number;
  /** Base repos worktrees were created from — so dispose() can prune them all,
   *  not just the parent repo (caller-supplied cwd can target other repos). */
  private worktreeRepos = new Set<string>();
  private scopeByAgent = new Map<string, string>();
  private stopReasons = new Map<string, unknown>();
  private disposed = false;

  /** Queue of background agents waiting to start. */
  private queue: { id: string; args: SpawnArgs }[] = [];
  /** Number of currently running background agents. */
  private runningBackground = 0;

  constructor(
    onComplete?: OnAgentComplete,
    maxConcurrent = DEFAULT_MAX_CONCURRENT,
    onStart?: OnAgentStart,
    onCompact?: OnAgentCompact,
  ) {
    this.onComplete = onComplete;
    this.onStart = onStart;
    this.onCompact = onCompact;
    this.maxConcurrent = maxConcurrent;
    // Cleanup completed agents after 10 minutes (but keep sessions for resume)
    this.cleanupInterval = setInterval(() => this.cleanup(), 60_000);
    this.cleanupInterval.unref();
  }

  /** Update the max concurrent background agents limit. */
  setMaxConcurrent(n: number) {
    this.maxConcurrent = Math.max(1, n);
    // Start queued agents if the new limit allows
    this.drainQueue();
  }

  getMaxConcurrent(): number {
    return this.maxConcurrent;
  }

  /**
   * Spawn an agent and return its ID immediately (for background use).
   * If the concurrency limit is reached, the agent is queued.
   */
  spawn(
    pi: ExtensionAPI,
    ctx: ExtensionContext,
    type: SubagentType,
    prompt: string,
    options: SpawnOptions,
  ): string {
    // Validate before the queue branch — a queued spawn should fail at the
    // call, not minutes later at drain. Throw (not warn): programmatic callers
    // can fix and retry; the RPC layer converts throws into error envelopes.
    assertValidSpawnCwd(options.cwd);

    const id = randomUUID().slice(0, 17);
    const scopeHandle = randomUUID();
    const parentHandle = ownershipRuntime.storage.getStore();
    const abortController = new AbortController();
    const record: AgentRecord = {
      id,
      type,
      description: options.description,
      status: options.isBackground ? "queued" : "running",
      toolUses: 0,
      startedAt: Date.now(),
      abortController,
      lifetimeUsage: { input: 0, output: 0, cacheWrite: 0 },
      compactionCount: 0,
      // Raw tri-state (not coerced to a boolean): true = background, false =
      // foreground (has an inline tool-result surface), undefined = caller never
      // declared it (e.g. a cross-extension RPC spawn). The widget's background-
      // only filter excludes only explicit `false`, so undefined agents — which
      // have no inline surface — stay visible instead of vanishing.
      isBackground: options.isBackground,
      invocation: options.invocation,
    };
    this.agents.set(id, record);
    const scope: OwnershipScope = {
      handle: scopeHandle,
      agentId: id,
      manager: this,
      parent: parentHandle,
      children: new Set(),
      ctx,
      cancelReason: parentHandle
        ? ownershipRuntime.scopes.get(parentHandle)?.cancelReason
        : undefined,
    };
    ownershipRuntime.scopes.set(scopeHandle, scope);
    ownershipRuntime.scopes.get(parentHandle ?? "")?.children.add(scopeHandle);
    this.scopeByAgent.set(id, scopeHandle);

    const args: SpawnArgs = { pi, ctx, type, prompt, options };

    if (options.isBackground && !options.bypassQueue && this.runningBackground >= this.maxConcurrent) {
      // Queue it — will be started when a running agent completes
      this.queue.push({ id, args });
      return id;
    }

    // startAgent can throw (e.g. strict worktree-isolation failure) — clean
    // up the record so callers don't see an orphan in `listAgents()`.
    try {
      this.startAgent(id, record, args);
    } catch (err) {
      this.agents.delete(id);
      this.removeScope(id);
      throw err;
    }
    return id;
  }

  /** Actually start an agent (called immediately or from queue drain). */
  private startAgent(id: string, record: AgentRecord, { pi, ctx, type, prompt, options }: SpawnArgs) {
    // Re-validate a caller-supplied cwd: queued spawns can start minutes after
    // spawn()'s check, and the directory may be gone by then (TOCTOU). Same
    // curated errors; drainQueue parks a throw on the record as an error.
    assertValidSpawnCwd(options.cwd);
    // Single resolution point for the caller-supplied cwd — the worktree base
    // repo and both cleanup calls below MUST agree on this value forever.
    const customCwd = options.cwd ?? undefined; // null (RPC "unset") → undefined
    const baseCwd = customCwd ?? ctx.cwd;

    // Worktree isolation: try to create a temporary git worktree. Strict —
    // fail loud if not possible (no silent fallback to main tree). Done
    // BEFORE state mutation so a throw doesn't leave the record half-running.
    let worktreeCwd: string | undefined;
    if (options.isolation === "worktree") {
      const wt = createWorktree(baseCwd, id);
      if (!wt) {
        throw new Error(
          'Cannot run with isolation: "worktree" — not a git repo, no commits yet, or `git worktree add` failed. ' +
          'Initialize git and commit at least once, or omit `isolation`.',
        );
      }
      record.worktree = wt;
      // workPath preserves subdirectory scoping for caller-supplied cwds: a
      // cwd deep in a monorepo maps to the same subdir inside the copy, not
      // the copied repo's root. Plain worktree spawns keep the historical
      // behavior (agent at the copy's root) — moving them to workPath would
      // also move .pi config discovery when the parent session sits in a repo
      // subdirectory, silently dropping extensions/skills.
      worktreeCwd = customCwd !== undefined ? wt.workPath : wt.path;
      this.worktreeRepos.add(baseCwd);
    }

    record.status = "running";
    record.startedAt = Date.now();
    if (options.isBackground) this.runningBackground++;
    this.onStart?.(record);

    // Wire parent abort signal to stop the subagent when the parent is interrupted
    let detachParentSignal: (() => void) | undefined;
    if (options.signal) {
      const onParentAbort = () => this.abort(id, options.signal?.reason);
      options.signal.addEventListener("abort", onParentAbort, { once: true });
      detachParentSignal = () => options.signal!.removeEventListener("abort", onParentAbort);
    }
    const detach = () => { detachParentSignal?.(); detachParentSignal = undefined; };

    const scopeHandle = this.scopeByAgent.get(id)!;
    const promise = ownershipRuntime.storage.run(scopeHandle, () => runAgent(ctx, type, prompt, {
      pi,
      agentId: id,
      model: options.model,
      maxTurns: options.maxTurns,
      isolated: options.isolated,
      inheritContext: options.inheritContext,
      thinkingLevel: options.thinkingLevel,
      // Worktree wins for the working dir (the agent must run in the copy —
      // which, with a custom cwd, was created from that target). Config stays
      // with the parent project when a caller-supplied cwd is in play; it must
      // stay undefined otherwise so plain worktree runs keep resolving config
      // (incl. relative extension paths and memory) inside the worktree copy.
      cwd: worktreeCwd ?? customCwd,
      configCwd: customCwd !== undefined ? ctx.cwd : undefined,
      signal: record.abortController!.signal,
      onToolActivity: (activity) => {
        if (activity.type === "end") record.toolUses++;
        options.onToolActivity?.(activity);
      },
      onTurnEnd: options.onTurnEnd,
      onTextDelta: options.onTextDelta,
      onAssistantUsage: (usage) => {
        addUsage(record.lifetimeUsage, usage);
        options.onAssistantUsage?.(usage);
      },
      onCompaction: (info) => {
        record.compactionCount++;
        this.onCompact?.(record, info);
        options.onCompaction?.(info);
      },
      onSessionCreated: (session) => {
        record.session = session;
        // Flush any steers that arrived before the session was ready
        if (record.pendingSteers?.length) {
          for (const msg of record.pendingSteers) {
            session.steer(msg).catch(() => {});
          }
          record.pendingSteers = undefined;
        }
        options.onSessionCreated?.(session);
      },
    }))
      .then(({ responseText, session, aborted, steered, failure }) => {
        if (this.stopReasons.has(id)) {
          record.status = "stopped";
        } else if (record.status !== "stopped") {
          // An external stop that bypassed the scoped-stop registry still wins
          // over a late resolution. Then: a hard abort keeps "aborted"; a failed
          // final turn (provider error that pi resolved instead of rejecting,
          // #144) is an honest "error" — never a completion with an empty or
          // stale result.
          if (aborted) {
            record.status = "aborted";
          } else if (failure) {
            record.status = "error";
            record.error = failure;
          } else {
            record.status = steered ? "steered" : "completed";
          }
        }
        record.result = responseText;
        record.session = session;
        record.completedAt ??= Date.now();

        detach();

        // Final flush of streaming output file
        if (record.outputCleanup) {
          try { record.outputCleanup(); } catch { /* ignore */ }
          record.outputCleanup = undefined;
        }

        // Clean up worktree if used
        if (record.worktree) {
          const wtResult = cleanupWorktree(baseCwd, record.worktree, options.description);
          record.worktreeResult = wtResult;
          if (wtResult.hasChanges && wtResult.branch) {
            // With a caller-supplied cwd the branch lives in THAT repo, not the
            // parent session's — say so, or the orchestrator merges in the wrong repo.
            const repoNote = customCwd !== undefined ? ` in \`${baseCwd}\`` : "";
            record.result = (record.result ?? "") +
              `\n\n---\nChanges saved to branch \`${wtResult.branch}\`${repoNote}. Merge with: \`git merge ${wtResult.branch}\`${customCwd !== undefined ? ` (run in \`${baseCwd}\`)` : ""}`;
          }
        }

        // Fire onComplete for foreground agents too — lifecycle symmetry.
        // Mark resultConsumed so the callback skips notifications (result returned inline).
        if (!options.isBackground) {
          record.resultConsumed = true;
          try { this.onComplete?.(record); } catch { /* ignore completion side-effect errors */ }
        } else {
          this.runningBackground--;
          try { this.onComplete?.(record); } catch { /* ignore completion side-effect errors */ }
          this.drainQueue();
        }
        return responseText;
      })
      .catch((err) => {
        if (this.stopReasons.has(id)) {
          record.status = "stopped";
        } else {
          record.status = "error";
          record.error = err instanceof Error ? err.message : String(err);
        }
        record.completedAt ??= Date.now();

        detach();

        // Final flush of streaming output file on error
        if (record.outputCleanup) {
          try { record.outputCleanup(); } catch { /* ignore */ }
          record.outputCleanup = undefined;
        }

        // Best-effort worktree cleanup on error
        if (record.worktree) {
          try {
            const wtResult = cleanupWorktree(baseCwd, record.worktree, options.description);
            record.worktreeResult = wtResult;
          } catch { /* ignore cleanup errors */ }
        }

        // Fire onComplete for foreground agents too — lifecycle symmetry.
        // Mark resultConsumed so the callback skips notifications (result returned inline).
        if (!options.isBackground) {
          record.resultConsumed = true;
          this.onComplete?.(record);
        } else {
          this.runningBackground--;
          this.onComplete?.(record);
          this.drainQueue();
        }
        return "";
      })
      .finally(() => {
        this.stopReasons.delete(id);
      });

    record.promise = promise;
    const scope = ownershipRuntime.scopes.get(scopeHandle);
    if (scope?.cancelReason !== undefined) this.abort(id, scope.cancelReason);

    // Notify caller that spawn is complete (record is in the map, promise is set).
    // Called synchronously — onSessionCreated fires asynchronously inside runAgent.
    // Used by spawnAndWait to let the caller set up output files before streaming starts.
    this.onSpawned?.(id);
  }

  /** Start queued agents up to the concurrency limit. */
  private drainQueue() {
    while (this.queue.length > 0 && this.runningBackground < this.maxConcurrent) {
      const next = this.queue.shift()!;
      const record = this.agents.get(next.id);
      if (!record || record.status !== "queued") continue;
      try {
        this.startAgent(next.id, record, next.args);
      } catch (err) {
        // Late failure (e.g. strict worktree-isolation) — surface on the record
        // so the user/agent can see it via /agents, then keep draining.
        record.status = "error";
        record.error = err instanceof Error ? err.message : String(err);
        record.completedAt = Date.now();
        this.onComplete?.(record);
      }
    }
  }

  /**
   * Called synchronously right after spawn, before onSessionCreated fires.
   * Lets the caller set up the output file path on the record.
   * The record is guaranteed to be in this.agents at this point.
   */
  private onSpawned?: (id: string) => void;

  /**
   * Spawn an agent and wait for completion (foreground use).
   * Foreground agents bypass the concurrency queue.
   * Returns { id, record } so callers can access the agent ID.
   *
   * @param onSpawned - Called synchronously after spawn(), before onSessionCreated fires.
   *   Use this to set record.outputFile so streamToOutputFile can pick it up.
   */
  async spawnAndWait(
    pi: ExtensionAPI,
    ctx: ExtensionContext,
    type: SubagentType,
    prompt: string,
    options: Omit<SpawnOptions, "isBackground">,
    onSpawned?: (id: string) => void,
  ): Promise<{ id: string; record: AgentRecord }> {
    // Temporarily register the onSpawned hook so startAgent can call it.
    const prevOnSpawned = this.onSpawned;
    this.onSpawned = onSpawned;
    try {
      const id = this.spawn(pi, ctx, type, prompt, { ...options, isBackground: false });
      const record = this.agents.get(id)!;
      await record.promise;
      return { id, record };
    } finally {
      this.onSpawned = prevOnSpawned;
    }
  }

  /**
   * Resume an existing agent session with a new prompt.
   */
  async resume(
    id: string,
    prompt: string,
    signal?: AbortSignal,
  ): Promise<AgentRecord | undefined> {
    const record = this.agents.get(id);
    if (!record?.session) return undefined;

    record.status = "running";
    record.startedAt = Date.now();
    record.completedAt = undefined;
    record.result = undefined;
    record.error = undefined;

    try {
      const { text, failure } = await resumeAgent(record.session, prompt, {
        onToolActivity: (activity) => {
          if (activity.type === "end") record.toolUses++;
        },
        onAssistantUsage: (usage) => {
          addUsage(record.lifetimeUsage, usage);
        },
        onCompaction: (info) => {
          record.compactionCount++;
          this.onCompact?.(record, info);
        },
        signal,
      });
      // Same contract as the spawn path (#144): a failed final turn is an
      // error, not a completion — but the resumed text stays available.
      record.status = failure ? "error" : "completed";
      if (failure) record.error = failure;
      record.result = text;
      record.completedAt = Date.now();
    } catch (err) {
      record.status = "error";
      record.error = err instanceof Error ? err.message : String(err);
      record.completedAt = Date.now();
    }

    return record;
  }

  /**
   * Send a steering message to an agent from the UI (mirrors the steer_subagent
   * tool). A live session delivers it now — it interrupts the agent after its
   * current tool execution and appears as a user message. If the session isn't
   * ready yet, the message is queued on `pendingSteers` and flushed when the
   * session is created. Returns false if the agent can't accept steering
   * (unknown id, or no longer running/queued).
   */
  steer(id: string, message: string): boolean {
    const record = this.agents.get(id);
    if (!record) return false;
    if (record.status !== "running" && record.status !== "queued") return false;
    if (record.session) {
      record.session.steer(message).catch(() => {});
    } else {
      if (!record.pendingSteers) record.pendingSteers = [];
      record.pendingSteers.push(message);
    }
    return true;
  }

  getRecord(id: string): AgentRecord | undefined {
    return this.agents.get(id);
  }

  listAgents(): AgentRecord[] {
    return [...this.agents.values()].sort(
      (a, b) => b.startedAt - a.startedAt,
    );
  }

  abort(id: string, reason?: unknown): boolean {
    const record = this.agents.get(id);
    if (!record) return false;

    if (record.status === "queued") {
      this.queue = this.queue.filter(q => q.id !== id);
      this.stopReasons.set(id, reason);
      record.abortController?.abort(reason);
      record.status = "stopped";
      record.completedAt = Date.now();
      return true;
    }

    if (record.status === "stopping") return true;
    if (record.status !== "running") return false;
    this.stopReasons.set(id, reason);
    record.abortController?.abort(reason);
    record.status = "stopping";
    return true;
  }

  getScopeHandle(id: string): string | undefined {
    return this.scopeByAgent.get(id);
  }

  async cancelScope(
    handle: string,
    reason: unknown = "Cancelled by owner",
    timeoutMs = DEFAULT_CANCEL_TIMEOUT_MS,
  ): Promise<ScopeCancellationResult | undefined> {
    const scope = ownershipRuntime.scopes.get(handle);
    if (!scope) return undefined;
    if (scope.cancelPromise) return scope.cancelPromise;
    scope.cancelReason = reason;
    scope.cancelPromise = this.finalizeScope(scope, Date.now() + timeoutMs);
    return scope.cancelPromise;
  }

  private async finalizeScope(scope: OwnershipScope, deadline: number): Promise<ScopeCancellationResult> {
    const failures: string[] = [];
    const visited = new Set<string>();

    while (true) {
      const pendingChildren = [...scope.children]
        .filter((handle) => !visited.has(handle))
        .map((handle) => ownershipRuntime.scopes.get(handle))
        .filter((child): child is OwnershipScope => child !== undefined);
      if (pendingChildren.length === 0) break;
      for (const child of pendingChildren) {
        visited.add(child.handle);
        child.cancelReason ??= scope.cancelReason;
        child.cancelPromise ??= child.manager.finalizeScope(child, deadline);
      }
      const results = await Promise.all(pendingChildren.map((child) => child.cancelPromise!));
      for (const result of results) failures.push(...result.failures);
      if (Date.now() >= deadline) break;
    }

    ownershipRuntime.storage.run(scope.handle, () => {
      scope.manager.abort(scope.agentId, scope.cancelReason);
    });
    const record = scope.manager.getRecord(scope.agentId);
    if (record?.promise && !(await waitWithin(record.promise, deadline))) {
      failures.push(`Agent ${scope.agentId} did not settle before cancellation timeout`);
    }

    for (const handle of scope.children) {
      if (visited.has(handle)) continue;
      const child = ownershipRuntime.scopes.get(handle);
      if (!child) continue;
      child.cancelReason ??= scope.cancelReason;
      child.cancelPromise ??= child.manager.finalizeScope(child, deadline);
      const result = await child.cancelPromise;
      failures.push(...result.failures);
    }
    return { settled: failures.length === 0, failures };
  }

  /** Dispose a record's session and remove it from the map. */
  private removeRecord(id: string, record: AgentRecord): void {
    record.session?.dispose?.();
    record.session = undefined;
    this.agents.delete(id);
    this.removeScope(id);
  }

  private removeScope(id: string): void {
    const handle = this.scopeByAgent.get(id);
    if (!handle) return;
    const scope = ownershipRuntime.scopes.get(handle);
    if (scope) {
      ownershipRuntime.scopes.get(scope.parent ?? "")?.children.delete(handle);
      ownershipRuntime.scopes.delete(handle);
    }
    this.scopeByAgent.delete(id);
    this.stopReasons.delete(id);
  }

  private cleanup() {
    const cutoff = Date.now() - 10 * 60_000;
    for (const [id, record] of this.agents) {
      if (record.status === "running" || record.status === "stopping" || record.status === "queued") continue;
      if ((record.completedAt ?? 0) >= cutoff) continue;
      this.removeRecord(id, record);
    }
  }

  /**
   * Remove all completed/stopped/errored records immediately.
   * Called on session start/switch so tasks from a prior session don't persist.
   * Pass skipUnconsumed=true to preserve records the LLM hasn't read yet
   * (resultConsumed=false) — they will be evicted by the 10-minute cleanup timer instead.
   */
  clearCompleted(skipUnconsumed = false): void {
    for (const [id, record] of this.agents) {
      if (record.status === "running" || record.status === "stopping" || record.status === "queued") continue;
      if (skipUnconsumed && !record.resultConsumed) continue;
      this.removeRecord(id, record);
    }
  }

  /** Whether any agents are still running or queued. */
  hasRunning(): boolean {
    return [...this.agents.values()].some(
      r => r.status === "running" || r.status === "stopping" || r.status === "queued",
    );
  }

  /** Request cancellation for all running and queued agents. */
  abortAll(reason?: unknown): number {
    let count = 0;
    for (const record of this.agents.values()) {
      if (record.status === "running" || record.status === "queued") {
        if (this.abort(record.id, reason)) count++;
      }
    }
    return count;
  }

  /** Wait for all running and queued agents to complete (including queued ones). */
  async waitForAll(): Promise<void> {
    // Loop because drainQueue respects the concurrency limit — as running
    // agents finish they start queued ones, which need awaiting too.
    while (true) {
      this.drainQueue();
      const pending = [...this.agents.values()]
        .filter(r => r.status === "running" || r.status === "stopping" || r.status === "queued")
        .map(r => r.promise)
        .filter(Boolean);
      if (pending.length === 0) break;
      await Promise.allSettled(pending);
    }
  }

  async dispose(): Promise<void> {
    if (this.disposed) return;
    this.disposed = true;
    clearInterval(this.cleanupInterval);
    const roots = [...this.scopeByAgent.values()]
      .map((handle) => ownershipRuntime.scopes.get(handle))
      .filter((scope): scope is OwnershipScope => scope?.manager === this)
      .filter((scope) => {
        const parent = scope.parent ? ownershipRuntime.scopes.get(scope.parent) : undefined;
        return parent?.manager !== this;
      });
    await Promise.all(roots.map((scope) => this.cancelScope(scope.handle, "Agent manager disposed", 250)));
    this.queue = [];
    for (const [id, record] of this.agents) this.removeRecord(id, record);
    // Prune any orphaned git worktrees (crash recovery)
    try { pruneWorktrees(process.cwd()); } catch { /* ignore */ }
    // Also prune repos that caller-supplied cwds created worktrees in — a clean
    // exit with in-flight agents would otherwise leave stale registrations there.
    for (const repo of this.worktreeRepos) {
      try { pruneWorktrees(repo); } catch { /* ignore */ }
    }
  }
}
