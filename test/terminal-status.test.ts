/**
 * terminal-status.test.ts — a subagent run that ends without any usable output
 * must surface as a terminal failure, never as a "completed" run with an empty
 * result.
 *
 * pi's agent loop resolves an exhausted provider failure normally: the final
 * assistant message is empty and carries stopReason "error" / "aborted" /
 * "length" plus an errorMessage (see @earendil-works/pi-agent-core StreamFn).
 * The old manager mapped every non-aborted resolution straight to "completed",
 * so a failed run looked like a silent success with an empty result. Two
 * directions are pinned here:
 *   1. an empty terminal turn → status "error" carrying its stop reason;
 *   2. a non-empty textual result → status "completed".
 */
import { afterEach, describe, expect, it, vi } from "vitest";

// Keep the real classifyEmptyResult; stub only the run entrypoints so the
// manager tests can inject a terminal-failure RunResult.
vi.mock("../src/agent-runner.js", async () => {
  const actual = await vi.importActual<typeof import("../src/agent-runner.js")>("../src/agent-runner.js");
  return { ...actual, runAgent: vi.fn(), resumeAgent: vi.fn() };
});

vi.mock("../src/worktree.js", () => ({
  createWorktree: vi.fn(),
  cleanupWorktree: vi.fn(() => ({ hasChanges: false })),
  pruneWorktrees: vi.fn(),
}));

import { AgentManager } from "../src/agent-manager.js";
import { classifyEmptyResult, resumeAgent, runAgent } from "../src/agent-runner.js";

const mockPi = {} as any;
const mockCtx = { cwd: "/tmp" } as any;
const mockSession = () => ({ dispose: vi.fn() }) as any;

/** A fake session whose history ends with `msg` (plus a leading user turn). */
function sessionEndingWith(msg: Record<string, unknown> | undefined) {
  return { messages: msg ? [{ role: "user", content: "go" }, msg] : [] } as any;
}

describe("classifyEmptyResult — reads the final assistant turn's stop reason", () => {
  it("carries a provider-error stop reason and its errorMessage verbatim", () => {
    const failure = classifyEmptyResult(
      sessionEndingWith({ role: "assistant", content: [], stopReason: "error", errorMessage: "fetch failed" }),
    );
    expect(failure.stopReason).toBe("error");
    expect(failure.message).toBe("fetch failed");
  });

  it("reports an empty length stop", () => {
    const failure = classifyEmptyResult(
      sessionEndingWith({ role: "assistant", content: [], stopReason: "length" }),
    );
    expect(failure.stopReason).toBe("length");
    expect(failure.message).toMatch(/length/i);
  });

  it("reports an aborted turn", () => {
    const failure = classifyEmptyResult(
      sessionEndingWith({ role: "assistant", content: [], stopReason: "aborted" }),
    );
    expect(failure.stopReason).toBe("aborted");
    expect(failure.message).toMatch(/abort/i);
  });

  it("falls back to a generic message when no assistant turn was recorded", () => {
    const failure = classifyEmptyResult(sessionEndingWith(undefined));
    expect(failure.stopReason).toBeUndefined();
    expect(failure.message).toMatch(/no output/i);
  });

  it("inspects the FINAL assistant turn, not an earlier clean one", () => {
    const session = {
      messages: [
        { role: "assistant", content: [{ type: "text", text: "earlier" }], stopReason: "stop" },
        { role: "toolResult", content: [] },
        { role: "assistant", content: [], stopReason: "error", errorMessage: "boom" },
      ],
    } as any;
    expect(classifyEmptyResult(session).message).toBe("boom");
  });
});

describe("AgentManager — terminal-status wiring on spawn", () => {
  let manager: AgentManager;
  afterEach(() => manager?.dispose());

  it("maps an empty provider-error turn to status error carrying its stop reason", async () => {
    manager = new AgentManager();
    vi.mocked(runAgent).mockResolvedValue({
      responseText: "",
      session: mockSession(),
      aborted: false,
      steered: false,
      failure: { stopReason: "error", message: "fetch failed" },
    });

    const { record } = await manager.spawnAndWait(mockPi, mockCtx, "general-purpose", "t", { description: "t" });

    expect(record.status).toBe("error");
    expect(record.error).toBe("fetch failed");
    expect(record.result ?? "").toBe("");
  });

  it("maps an empty length stop to status error", async () => {
    manager = new AgentManager();
    vi.mocked(runAgent).mockResolvedValue({
      responseText: "",
      session: mockSession(),
      aborted: false,
      steered: false,
      failure: { stopReason: "length", message: "subagent hit the length limit before producing any output" },
    });

    const { record } = await manager.spawnAndWait(mockPi, mockCtx, "general-purpose", "t", { description: "t" });

    expect(record.status).toBe("error");
    expect(record.error).toMatch(/length/i);
  });

  it("still completes when a non-empty textual result comes back", async () => {
    manager = new AgentManager();
    vi.mocked(runAgent).mockResolvedValue({
      responseText: "the answer",
      session: mockSession(),
      aborted: false,
      steered: false,
    });

    const { record } = await manager.spawnAndWait(mockPi, mockCtx, "general-purpose", "t", { description: "t" });

    expect(record.status).toBe("completed");
    expect(record.result).toBe("the answer");
    expect(record.error).toBeUndefined();
  });

  it("does not reclassify a hard-aborted (max-turns) run as an error", async () => {
    manager = new AgentManager();
    vi.mocked(runAgent).mockResolvedValue({
      responseText: "",
      session: mockSession(),
      aborted: true,
      steered: false,
      failure: { stopReason: "aborted", message: "aborted before output" },
    });

    const { record } = await manager.spawnAndWait(mockPi, mockCtx, "general-purpose", "t", { description: "t" });

    expect(record.status).toBe("aborted");
  });
});

describe("AgentManager — terminal-status wiring on resume", () => {
  let manager: AgentManager;
  afterEach(() => manager?.dispose());

  it("reports an empty resumed turn as an error, never as a completion", async () => {
    manager = new AgentManager();
    // The first run answered; its session history then ends with an empty
    // provider-error turn from the resumed prompt. The resumed run's status
    // must come from that final turn — not from the earlier answer.
    const session = sessionEndingWith({
      role: "assistant",
      content: [],
      stopReason: "error",
      errorMessage: "socket hang up",
    });
    session.dispose = vi.fn();
    vi.mocked(runAgent).mockResolvedValue({
      responseText: "first response",
      session,
      aborted: false,
      steered: false,
    });
    const { record } = await manager.spawnAndWait(mockPi, mockCtx, "general-purpose", "t", { description: "t" });
    expect(record.status).toBe("completed");

    vi.mocked(resumeAgent).mockResolvedValue("");
    const resumed = await manager.resume(record.id, "continue");

    expect(resumed?.status).toBe("error");
    expect(resumed?.error).toBe("socket hang up");
    expect(resumed?.result).toBeUndefined();
  });
});
