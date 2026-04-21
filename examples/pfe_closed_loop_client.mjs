/**
 * JS/TS reference for the PFE closed loop.
 *
 * This file is intentionally framework-free so it can be dropped into a browser
 * app, Electron shell, or agent UI as the smallest possible state layer.
 */

let fallbackTurnCounter = 0;

function nowIso() {
  return new Date().toISOString();
}

function createId(prefix) {
  if (typeof globalThis.crypto?.randomUUID === "function") {
    return `${prefix}-${globalThis.crypto.randomUUID()}`;
  }
  fallbackTurnCounter += 1;
  return `${prefix}-${Date.now()}-${fallbackTurnCounter}`;
}

function compactObject(value) {
  const out = {};
  for (const [key, entry] of Object.entries(value ?? {})) {
    if (entry !== undefined) {
      out[key] = entry;
    }
  }
  return out;
}

function resolveAssistantText(response) {
  return response.choices?.[0]?.message?.content ?? "";
}

function feedbackStatusFromAction(action) {
  switch (action) {
    case "accept":
      return "accepted";
    case "edit":
      return "edited";
    case "regenerate":
      return "regenerated";
    case "reject":
      return "rejected";
    case "delete":
      return "deleted";
    default:
      return "pending";
  }
}

export class PFEClosedLoopWebClient {
  constructor({ baseUrl = "http://127.0.0.1:8921", apiKey = null, fetchImpl = globalThis.fetch } = {}) {
    if (typeof fetchImpl !== "function") {
      throw new Error("A fetch implementation is required.");
    }
    this.baseUrl = baseUrl.replace(/\/$/, "");
    this.apiKey = apiKey;
    this.fetch = fetchImpl;
  }

  headers() {
    const headers = {
      "Content-Type": "application/json",
    };
    if (this.apiKey) {
      headers.Authorization = `Bearer ${this.apiKey}`;
    }
    return headers;
  }

  async post(path, payload) {
    const response = await this.fetch(`${this.baseUrl}${path}`, {
      method: "POST",
      headers: this.headers(),
      body: JSON.stringify(payload),
    });
    const text = await response.text();
    let data = {};
    if (text) {
      try {
        data = JSON.parse(text);
      } catch (error) {
        throw new Error(`Non-JSON response calling ${path}: ${text}`);
      }
    }
    if (!response.ok) {
      throw new Error(`HTTP ${response.status} calling ${path}: ${text}`);
    }
    return data;
  }

  async sendUserMessage({
    message,
    sessionId,
    adapterVersion = "latest",
    model = "local",
    metadata = {},
    messages = null,
  }) {
    const payload = {
      model,
      adapter_version: adapterVersion,
      messages: messages ?? [{ role: "user", content: message }],
      metadata,
    };
    if (sessionId) {
      payload.session_id = sessionId;
    }

    const response = await this.post("/v1/chat/completions", payload);
    return {
      raw: response,
      assistantText: resolveAssistantText(response),
      sessionId: response.session_id ?? sessionId ?? null,
      requestId: response.request_id ?? null,
      adapterVersion: response.adapter_version ?? adapterVersion,
      signalCollection: response.metadata?.signal_collection ?? null,
      metadata: response.metadata ?? {},
    };
  }

  async submitFeedback({
    sessionId,
    requestId,
    action,
    responseTimeSeconds,
    editedText,
    nextMessage,
    userMessage,
    assistantMessage,
    metadata = {},
  }) {
    return this.post("/pfe/feedback", {
      session_id: sessionId,
      request_id: requestId,
      action,
      response_time_seconds: responseTimeSeconds,
      edited_text: editedText,
      next_message: nextMessage,
      user_message: userMessage,
      assistant_message: assistantMessage,
      metadata,
    });
  }
}

export class PFEMessageStore {
  constructor({ maxTurns = 200 } = {}) {
    this.maxTurns = maxTurns;
    this.turns = new Map();
    this.order = [];
  }

  upsertTurn({
    turnId,
    sessionId,
    requestId,
    userText,
    assistantText,
    adapterVersion,
    signalCollection,
    metadata = {},
    status,
    feedback,
  }) {
    const resolvedTurnId = turnId ?? requestId ?? createId("turn");
    const existing = this.turns.get(resolvedTurnId) ?? null;
    const nextTurn = {
      turnId: resolvedTurnId,
      sessionId: sessionId ?? existing?.sessionId ?? null,
      requestId: requestId ?? existing?.requestId ?? null,
      userText: userText ?? existing?.userText ?? "",
      assistantText: assistantText ?? existing?.assistantText ?? "",
      adapterVersion: adapterVersion ?? existing?.adapterVersion ?? "latest",
      signalCollection: signalCollection ?? existing?.signalCollection ?? null,
      metadata: {
        ...(existing?.metadata ?? {}),
        ...compactObject(metadata),
      },
      status: status ?? existing?.status ?? "draft",
      feedback: feedback ?? existing?.feedback ?? null,
      createdAt: existing?.createdAt ?? nowIso(),
      updatedAt: nowIso(),
    };

    this.turns.set(resolvedTurnId, nextTurn);
    if (!existing) {
      this.order.push(resolvedTurnId);
    }
    this.trim();
    return nextTurn;
  }

  trim() {
    while (this.order.length > this.maxTurns) {
      const removed = this.order.shift();
      if (removed) {
        this.turns.delete(removed);
      }
    }
  }

  getTurn(turnId) {
    return this.turns.get(turnId) ?? null;
  }

  latestTurn() {
    const turnId = this.order[this.order.length - 1];
    return turnId ? this.turns.get(turnId) ?? null : null;
  }

  latestPendingTurn() {
    for (let index = this.order.length - 1; index >= 0; index -= 1) {
      const turn = this.turns.get(this.order[index]);
      if (turn && (!turn.feedback || turn.feedback.dispatchStatus !== "sent")) {
        return turn;
      }
    }
    return null;
  }

  markFeedback(turnId, feedback, result = null) {
    const existing = this.getTurn(turnId);
    if (!existing) {
      throw new Error(`Unknown turn id: ${turnId}`);
    }
    return this.upsertTurn({
      turnId,
      status: feedbackStatusFromAction(feedback.action),
      feedback: {
        ...(existing.feedback ?? {}),
        ...compactObject(feedback),
        result,
        recordedAt: feedback.recordedAt ?? nowIso(),
      },
    });
  }

  listTurns() {
    return this.order.map((turnId) => this.turns.get(turnId)).filter(Boolean);
  }

  snapshot() {
    return {
      turnCount: this.order.length,
      latestTurn: this.latestTurn(),
      latestPendingTurn: this.latestPendingTurn(),
      turns: this.listTurns(),
    };
  }
}

export class PFEClosedLoopSessionTracker {
  constructor({ sessionId = null, adapterVersion = "latest" } = {}) {
    this.sessionId = sessionId;
    this.adapterVersion = adapterVersion;
    this.turnCount = 0;
    this.lastRequestId = null;
  }

  rememberSession(sessionId) {
    if (sessionId) {
      this.sessionId = sessionId;
    }
    return this.sessionId;
  }

  adoptReply(reply) {
    this.rememberSession(reply.sessionId ?? null);
    if (reply.requestId) {
      this.lastRequestId = reply.requestId;
    }
    if (reply.adapterVersion) {
      this.adapterVersion = reply.adapterVersion;
    }
    this.turnCount += 1;
    return this.snapshot();
  }

  snapshot() {
    return {
      sessionId: this.sessionId,
      adapterVersion: this.adapterVersion,
      lastRequestId: this.lastRequestId,
      turnCount: this.turnCount,
    };
  }
}

export class PFEFeedbackDispatcher {
  constructor(client, { dedupe = true } = {}) {
    if (!client) {
      throw new Error("A PFEClosedLoopWebClient instance is required.");
    }
    this.client = client;
    this.dedupe = dedupe;
    this.sentKeys = new Set();
  }

  async dispatch(turn, feedback) {
    if (!turn?.sessionId || !turn?.requestId) {
      throw new Error("A turn must include sessionId and requestId before feedback can be sent.");
    }
    const action = feedback.action;
    const dedupeKey = `${turn.sessionId}:${turn.requestId}:${action}`;
    if (this.dedupe && this.sentKeys.has(dedupeKey)) {
      return {
        skipped: true,
        dedupeKey,
      };
    }

    const result = await this.client.submitFeedback({
      sessionId: turn.sessionId,
      requestId: turn.requestId,
      action,
      responseTimeSeconds: feedback.responseTimeSeconds,
      editedText: feedback.editedText,
      nextMessage: feedback.nextMessage,
      userMessage: feedback.userMessage ?? turn.userText,
      assistantMessage: feedback.assistantMessage ?? turn.assistantText,
      metadata: {
        ...(turn.metadata ?? {}),
        ...(feedback.metadata ?? {}),
        turnId: turn.turnId,
        adapterVersion: turn.adapterVersion,
        source: "app_state",
      },
    });

    this.sentKeys.add(dedupeKey);
    return result;
  }
}

export class PFEClosedLoopConversation {
  constructor({
    client,
    store = new PFEMessageStore(),
    sessionTracker = new PFEClosedLoopSessionTracker(),
    feedbackDispatcher = null,
  } = {}) {
    if (!client) {
      throw new Error("A PFEClosedLoopWebClient instance is required.");
    }
    this.client = client;
    this.store = store;
    this.sessionTracker = sessionTracker;
    this.feedbackDispatcher = feedbackDispatcher ?? new PFEFeedbackDispatcher(client);
  }

  async send(message, { sessionId = null, adapterVersion = "latest", model = "local", metadata = {}, messages = null } = {}) {
    const reply = await this.client.sendUserMessage({
      message,
      sessionId: sessionId ?? this.sessionTracker.sessionId,
      adapterVersion,
      model,
      metadata,
      messages,
    });
    this.sessionTracker.adoptReply(reply);
    const turn = this.store.upsertTurn({
      turnId: reply.requestId ?? createId("turn"),
      sessionId: reply.sessionId ?? this.sessionTracker.sessionId,
      requestId: reply.requestId,
      userText: message,
      assistantText: reply.assistantText,
      adapterVersion: reply.adapterVersion ?? adapterVersion,
      signalCollection: reply.signalCollection,
      metadata: {
        ...compactObject(metadata),
        signalCollection: reply.signalCollection,
      },
      status: "completed",
    });

    return {
      reply,
      turn,
      state: this.snapshot(),
    };
  }

  async submitFeedbackForTurn(turnId, feedback) {
    const turn = this.store.getTurn(turnId);
    if (!turn) {
      throw new Error(`Unknown turn id: ${turnId}`);
    }
    const result = await this.feedbackDispatcher.dispatch(turn, feedback);
    const updatedTurn = this.store.markFeedback(
      turnId,
      {
        ...feedback,
        recordedAt: nowIso(),
        dispatchStatus: "sent",
      },
      result,
    );
    return {
      result,
      turn: updatedTurn,
      state: this.snapshot(),
    };
  }

  async submitFeedbackForLatest(feedback) {
    const turn = this.store.latestTurn();
    if (!turn) {
      throw new Error("There is no turn to attach feedback to.");
    }
    return this.submitFeedbackForTurn(turn.turnId, feedback);
  }

  async acceptLatest(feedback = {}) {
    return this.submitFeedbackForLatest({
      action: "accept",
      ...feedback,
    });
  }

  async editLatest(feedback = {}) {
    return this.submitFeedbackForLatest({
      action: "edit",
      ...feedback,
    });
  }

  async regenerateLatest(feedback = {}) {
    return this.submitFeedbackForLatest({
      action: "regenerate",
      ...feedback,
    });
  }

  async rejectLatest(feedback = {}) {
    return this.submitFeedbackForLatest({
      action: "reject",
      ...feedback,
    });
  }

  async deleteLatest(feedback = {}) {
    return this.submitFeedbackForLatest({
      action: "delete",
      ...feedback,
    });
  }

  snapshot() {
    return {
      session: this.sessionTracker.snapshot(),
      store: this.store.snapshot(),
    };
  }
}

export function createTrackedMessageRecord({ userText, assistantText, sessionId, requestId, adapterVersion }) {
  return {
    turnId: requestId ?? createId("turn"),
    userText,
    assistantText,
    sessionId,
    requestId,
    adapterVersion,
    status: "completed",
    feedback: null,
    createdAt: nowIso(),
    updatedAt: nowIso(),
  };
}

/**
 * Example UI flow:
 *
 * const client = new PFEClosedLoopWebClient({ apiKey: "your-secret-key" });
 * const conversation = new PFEClosedLoopConversation({ client });
 *
 * const { turn, state } = await conversation.send(
 *   "My name is Alex. Keep answers concise and structured.",
 * );
 *
 * console.log(state.session.sessionId);
 * console.log(turn.turnId);
 *
 * await conversation.acceptLatest({
 *   responseTimeSeconds: 2.4,
 * });
 *
 * await conversation.editLatest({
 *   editedText: "Please make it shorter and add bullets.",
 *   responseTimeSeconds: 5.1,
 * });
 *
 * const snapshot = conversation.snapshot();
 * console.log(snapshot.store.latestTurn);
 */
