# app/services/line_io.py
import base64, hashlib, hmac, json, requests
from typing import Dict, Any
from flask import Request
from ..config import AppConfig

LINE_REPLY_URL = "https://api.line.me/v2/bot/message/reply"
LINE_PUSH_URL  = "https://api.line.me/v2/bot/message/push"

def _auth_header() -> Dict[str, str]:
    if not AppConfig.LINE_CHANNEL_ACCESS_TOKEN:
        raise RuntimeError("Missing LINE_CHANNEL_ACCESS_TOKEN")
    return {"Authorization": f"Bearer {AppConfig.LINE_CHANNEL_ACCESS_TOKEN}"}

def validate_line_signature(flask_request: Request) -> bool:
    """Validate X-Line-Signature using channel secret (HMAC-SHA256 of raw body)."""
    if not AppConfig.LINE_VALIDATE_SIGNATURE:
        return True  # dev bypass for Postman tests

    body = flask_request.get_data() or b""
    received = flask_request.headers.get("X-Line-Signature")
    if not received:
        return False

    secret = (AppConfig.LINE_CHANNEL_SECRET or "").encode("utf-8")
    digest = hmac.new(secret, body, hashlib.sha256).digest()
    expected = base64.b64encode(digest).decode("utf-8")
    return hmac.compare_digest(received, expected)

def reply_text(reply_token: str, text: str) -> Dict[str, Any]:
    payload = {"replyToken": reply_token, "messages": [{"type": "text", "text": text[:5000]}]}
    resp = requests.post(LINE_REPLY_URL,
                         headers={**_auth_header(), "Content-Type": "application/json"},
                         data=json.dumps(payload))
    return {"status_code": resp.status_code, "body": resp.text}

def push_text(user_id: str, text: str) -> Dict[str, Any]:
    payload = {"to": user_id, "messages": [{"type": "text", "text": text[:5000]}]}
    resp = requests.post(LINE_PUSH_URL,
                         headers={**_auth_header(), "Content-Type": "application/json"},
                         data=json.dumps(payload))
    return {"status_code": resp.status_code, "body": resp.text}
