from __future__ import annotations

import argparse
import json
from typing import Any, Optional
from urllib import error, request


class PFEClosedLoopClient:
    def __init__(self, base_url: str, api_key: Optional[str] = None, timeout: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = request.Request(
            url=f"{self.base_url}{path}",
            data=body,
            headers=self._headers(),
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code} calling {path}: {detail}") from exc

    def chat_completion(
        self,
        *,
        message: str,
        session_id: Optional[str] = None,
        adapter_version: str = "latest",
        model: str = "local",
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model,
            "adapter_version": adapter_version,
            "messages": [{"role": "user", "content": message}],
        }
        if session_id:
            payload["session_id"] = session_id
        return self._post("/v1/chat/completions", payload)

    def submit_feedback(
        self,
        *,
        session_id: str,
        request_id: str,
        action: str,
        response_time_seconds: Optional[float] = None,
        edited_text: Optional[str] = None,
        next_message: Optional[str] = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "session_id": session_id,
            "request_id": request_id,
            "action": action,
        }
        if response_time_seconds is not None:
            payload["response_time_seconds"] = response_time_seconds
        if edited_text:
            payload["edited_text"] = edited_text
        if next_message:
            payload["next_message"] = next_message
        return self._post("/pfe/feedback", payload)


def _print_json(title: str, payload: dict[str, Any]) -> None:
    print(f"\n[{title}]")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description="Minimal closed-loop PFE client example.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8921", help="PFE server base URL")
    parser.add_argument("--api-key", default=None, help="Optional API key for management endpoints")
    parser.add_argument("--message", default="Remember that I prefer concise answers.", help="User message to send")
    parser.add_argument("--session-id", default=None, help="Optional session id to reuse across turns")
    parser.add_argument(
        "--feedback-action",
        default="accept",
        choices=["accept", "edit", "regenerate", "reject", "delete"],
        help="Feedback action to report after the chat call",
    )
    parser.add_argument("--edited-text", default=None, help="Edited text when feedback action is edit")
    parser.add_argument("--next-message", default=None, help="Next user message when feedback implies continue/regenerate")
    parser.add_argument("--response-time-seconds", type=float, default=3.0, help="Observed user response time in seconds")
    args = parser.parse_args()

    client = PFEClosedLoopClient(base_url=args.base_url, api_key=args.api_key)

    chat = client.chat_completion(
        message=args.message,
        session_id=args.session_id,
    )
    _print_json("chat", chat)

    session_id = chat.get("session_id")
    request_id = chat.get("request_id")
    if not session_id or not request_id:
        raise RuntimeError("PFE chat response did not include session_id/request_id.")

    feedback = client.submit_feedback(
        session_id=session_id,
        request_id=request_id,
        action=args.feedback_action,
        response_time_seconds=args.response_time_seconds,
        edited_text=args.edited_text,
        next_message=args.next_message,
    )
    _print_json("feedback", feedback)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
