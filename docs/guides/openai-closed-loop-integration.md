# OpenAI Compatible App Closed-Loop Integration

This document describes the smallest integration that keeps PFE's personalization loop alive after you connect an app or agent through the OpenAI-compatible API.

## Goal

If you only call `POST /v1/chat/completions`, PFE can serve local inference but it cannot reliably learn from user behavior.

To keep the loop closed, your app must do both:

1. Call `POST /v1/chat/completions`
2. Report user behavior back to `POST /pfe/feedback`

## Minimal flow

1. App sends a normal OpenAI-style chat request to `/v1/chat/completions`
2. PFE returns the assistant response plus `session_id` and `request_id`
3. App keeps those two ids with the rendered answer
4. User accepts, edits, regenerates, or deletes the answer
5. App sends that behavior to `/pfe/feedback`
6. PFE converts the feedback into signals and passes them into `signal -> curate -> train/eval/promote`

## Server startup

Start the local service:

```bash
python -m pfe_server --host 127.0.0.1 --port 8921 --api-key your-secret-key
```

Recommended client settings:

- Base URL: `http://127.0.0.1:8921`
- Chat endpoint: `POST /v1/chat/completions`
- Feedback endpoint: `POST /pfe/feedback`
- Auth header: `Authorization: Bearer your-secret-key`

## Chat request

```json
POST /v1/chat/completions
{
  "model": "local",
  "adapter_version": "latest",
  "messages": [
    {
      "role": "user",
      "content": "My name is Alex. Please keep answers concise."
    }
  ]
}
```

Example response fields that the app must keep:

```json
{
  "choices": [
    {
      "message": {
        "role": "assistant",
        "content": "Understood. I will keep replies concise."
      }
    }
  ],
  "session_id": "sess-abc12345",
  "request_id": "req-def67890",
  "metadata": {
    "signal_collection": {
      "session_id": "sess-abc12345",
      "request_id": "req-def67890",
      "interaction_stored": true
    }
  }
}
```

## Feedback request

When the user reacts to that answer, send the same ids back:

```json
POST /pfe/feedback
{
  "session_id": "sess-abc12345",
  "request_id": "req-def67890",
  "action": "accept",
  "response_time_seconds": 3.2
}
```

Supported `action` values:

- `accept`
- `edit`
- `regenerate`
- `reject`
- `delete`

Optional feedback fields:

- `edited_text`: required for useful `edit` signals
- `next_message`: helpful when the user continues after an accepted answer
- `response_time_seconds`: helps signal confidence

## Data handling guidance

Keep these fields per rendered answer:

- `session_id`
- `request_id`
- `adapter_version`
- Original user message
- Assistant message

The first two are required for the feedback loop. The rest improves signal quality and downstream pairing.

## Common integration patterns

### Standard app UI

- Render the answer from `/v1/chat/completions`
- Store `session_id/request_id` next to the message bubble
- On Accept button: call `/pfe/feedback` with `action=accept`
- On Edit and Send: call `/pfe/feedback` with `action=edit` and `edited_text`
- On Regenerate: call `/pfe/feedback` with `action=regenerate`
- On Delete or explicit thumbs down: call `/pfe/feedback` with `action=delete` or `action=reject`

### Agent wrapper

- Keep the upstream agent logic talking to `/v1/chat/completions`
- Add a thin wrapper that captures `session_id/request_id`
- Report tool-user outcomes or explicit operator actions to `/pfe/feedback`
- Do not assume OpenAI-compatible traffic alone is enough for training

## What this enables

With this loop in place, PFE can continue to:

- record explicit and implicit user preferences
- build SFT and DPO candidates when the event chain is complete
- update memory and profile routing from user messages
- raise auto-train readiness based on reinforced preference samples

## Reference implementation

Use the minimal client example here:

`examples/pfe_closed_loop_client.py`

For a browser or Electron-style app wrapper, use:

`examples/pfe_closed_loop_client.mjs`

Run it like this:

```bash
python examples/pfe_closed_loop_client.py \
  --base-url http://127.0.0.1:8921 \
  --api-key your-secret-key \
  --message "My name is Alex. I prefer structured answers." \
  --feedback-action accept
```

## Integration boundary

The OpenAI-compatible path is still only the inference surface. The personalization loop depends on the app preserving response ids and reporting user behavior. If the app drops `session_id/request_id`, PFE can still answer, but the training loop becomes incomplete or unreliable.

## JS/TS wrapper pattern

The recommended app-side shape is:

1. Send the user message to `/v1/chat/completions`
2. Store `sessionId/requestId` with the rendered assistant message
3. Keep a light message store so accept/edit/regenerate/reject actions can find the right turn
4. Dispatch feedback back to `/pfe/feedback`

The reference example exports a small state layer:

- `PFEClosedLoopWebClient`
- `PFEMessageStore`
- `PFEClosedLoopSessionTracker`
- `PFEFeedbackDispatcher`
- `PFEClosedLoopConversation`

```js
import {
  PFEClosedLoopConversation,
  PFEClosedLoopWebClient,
} from "../examples/pfe_closed_loop_client.mjs";

const client = new PFEClosedLoopWebClient({
  baseUrl: "http://127.0.0.1:8921",
  apiKey: "your-secret-key",
});

const conversation = new PFEClosedLoopConversation({ client });

const { turn, state } = await conversation.send(
  "My name is Alex. Keep answers concise and structured.",
);

console.log(state.session.sessionId);
console.log(turn.turnId);

await conversation.acceptLatest({
  responseTimeSeconds: 2.1,
});

await conversation.editLatest({
  editedText: "Please make it shorter and use bullet points.",
  responseTimeSeconds: 4.8,
});
```

For a real UI, treat the store as the source of truth for the rendered timeline:

```ts
type ClosedLoopTurn = {
  turnId: string;
  sessionId: string | null;
  requestId: string | null;
  userText: string;
  assistantText: string;
  adapterVersion: string;
  status: "draft" | "completed" | "accepted" | "edited" | "regenerated" | "rejected" | "deleted";
  feedback: null | {
    action: "accept" | "edit" | "regenerate" | "reject" | "delete";
    responseTimeSeconds?: number;
    editedText?: string;
    nextMessage?: string;
    recordedAt: string;
  };
  createdAt: string;
  updatedAt: string;
};
```

That shape keeps the UI layer simple while preserving the ids PFE needs for signal pairing and downstream training.

## Built-in local chat page

The bundled local chat page at `pfe-server/pfe_server/static/chat.html` now follows the same closed-loop rules:

- it sends chat requests through `/v1/chat/completions`
- it stores the returned `session_id/request_id` on each assistant turn
- its `accept / reject / regenerate / edit` buttons report feedback against the original turn ids
- `regenerate` records feedback first, then automatically re-runs the same user turn with preserved context
- it exposes an optional API key field so the same page can be used against remote or `api_key_required` setups
- regenerated replies keep a visible `regenerated from <request_id>` trace in the message meta
- it surfaces connection/auth state so `401 / 403 / 503` failures are visible in-page instead of only in the console
- it can copy the current integration snapshot, including base URL, endpoints, session, adapter, and auth state
- it generates copyable `curl / Python / JavaScript` closed-loop templates from the current page state, including both chat and feedback requests
- it shows the most recent chat request id and feedback action/status so the closed loop is visible during demos
- it can export the current session as a compact chat+feedback JSON summary for debugging or handoff
- it surfaces the latest `signal_collection` metadata, curator result, trigger-threshold summary, sample ids / train type, blocker / next action, and a compact `chat -> feedback -> curator -> status` timeline so you can see both why the gate is blocked and how the loop progressed

That makes the built-in page a usable browser-side reference for a real app, not just a mock shell.
