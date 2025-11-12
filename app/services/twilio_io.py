# app/services/twilio_io.py
import os
import re
import time
import logging
from typing import List, Optional
import random
from twilio.rest import Client
from twilio.base.exceptions import TwilioRestException

log = logging.getLogger(__name__)

# Required env
TWILIO_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")

# EITHER provide a Messaging Service (preferred) ...
TWILIO_MESSAGING_SERVICE_SID = os.getenv("TWILIO_MESSAGING_SERVICE_SID")

# ... OR provide WhatsApp/SMS senders explicitly
TWILIO_WHATSAPP_FROM = os.getenv("TWILIO_WHATSAPP_FROM")  # e.g., 'whatsapp:+14155238886' (sandbox) OR 'whatsapp:+<your_bsp_number>'
TWILIO_SMS_FROM = os.getenv("TWILIO_SMS_FROM")            # e.g., '+1XXXXXXXXXX'

# chunking config
_MAX_CHUNK_LEN = int(os.getenv("TWILIO_MAX_CHUNK_LEN", "1500"))  # keep under Twilio 1600 char limit
_SLEEP_BETWEEN = float(os.getenv("TWILIO_SLEEP_BETWEEN", "0.18"))

_MAX_CHUNK_LEN = int(os.getenv("TWILIO_MAX_CHUNK_LEN", "1500"))  # keep under 1600
_SLEEP_BETWEEN = float(os.getenv("TWILIO_SLEEP_BETWEEN", "0.18"))

# retry config (NEW)
_MAX_ATTEMPTS = int(os.getenv("TWILIO_MAX_ATTEMPTS", "5"))
_BASE_BACKOFF = float(os.getenv("TWILIO_BASE_BACKOFF", "0.6"))  # seconds
_RETRYABLE_CODES = {20503, 20429}  # service unavailable, rate limit

# --- Helpers ----------------------------------------------------------------
def _ensure_client() -> Client:
    if not TWILIO_SID or not TWILIO_TOKEN:
        raise RuntimeError("Missing TWILIO_ACCOUNT_SID / TWILIO_AUTH_TOKEN")
    return Client(TWILIO_SID, TWILIO_TOKEN)

def _wa_prefix(num: str) -> str:
    """Ensure whatsapp: prefix for WhatsApp addresses."""
    num = (num or "").strip()
    if not num:
        raise ValueError("Empty WhatsApp number")
    if num.startswith("whatsapp:"):
        return num
    if num.startswith("+"):
        return f"whatsapp:{num}"
    digits = "".join(ch for ch in num if ch.isdigit() or ch == "+")
    if digits and not digits.startswith("+"):
        digits = "+" + digits
    return f"whatsapp:{digits}"


def _validate_wa_from(from_num: Optional[str]) -> str:
    """
    Choose a valid WhatsApp 'from' identity.
    If TWILIO_MESSAGING_SERVICE_SID is set we prefer that (return empty string),
    otherwise return a whatsapp: prefixed sender.
    """
    if TWILIO_MESSAGING_SERVICE_SID:
        return ""  # caller will use messaging_service_sid
    from_env = from_num or TWILIO_WHATSAPP_FROM
    if not from_env:
        raise RuntimeError(
            "No WhatsApp sender configured. Set TWILIO_MESSAGING_SERVICE_SID "
            "or TWILIO_WHATSAPP_FROM='whatsapp:+14155238886' (sandbox) / your WA BSP number."
        )
    return _wa_prefix(from_env)


def _split_text_preserve_paragraphs(text: str, max_len: int = _MAX_CHUNK_LEN) -> List[str]:
    """
    Split `text` into chunks <= max_len. Prefer paragraph boundaries, then sentences,
    finally whitespace if necessary.
    """
    if not text:
        return []
    import re as _re
    text = text.replace("\r\n", "\n").strip()
    paragraphs = [p.strip() for p in _re.split(r"\n\s*\n", text) if p.strip()]
    chunks: List[str] = []
    current = ""
    sentence_split_re = _re.compile(r'(?<=[。.!?！？])\s+|(?<=\.)\s+|(?<=\!)\s+|(?<=\?)\s+')
    for p in paragraphs:
        if len(p) <= max_len:
            cand = (current + "\n\n" + p).strip() if current else p
            if len(cand) <= max_len:
                current = cand
            else:
                if current: chunks.append(current)
                current = p
        else:
            for s in [s.strip() for s in sentence_split_re.split(p) if s.strip()]:
                cand = (current + " " + s).strip() if current else s
                if len(cand) <= max_len:
                    current = cand
                else:
                    if current: chunks.append(current); current = ""
                    if len(s) <= max_len:
                        current = s
                    else:
                        parts, cur = [], ""
                        for w in s.split(" "):
                            if not cur: cur = w
                            elif len(cur) + 1 + len(w) <= max_len:
                                cur += " " + w
                            else:
                                parts.append(cur); cur = w
                        if cur: parts.append(cur)
                        for j, sc in enumerate(parts):
                            if j < len(parts) - 1:
                                chunks.append(sc)
                            else:
                                current = sc
    if current:
        chunks.append(current)
    return chunks

def _is_retryable_twilio_error(e: Exception) -> bool:
    if isinstance(e, TwilioRestException):
        try:
            if e.code in _RETRYABLE_CODES:
                return True
            if getattr(e, "status", 0) and int(e.status) >= 500:
                return True
        except Exception:
            pass
        msg = (e.msg or "") + " " + (e.detail or "")
        if "Service is unavailable" in msg or "rate limit" in msg.lower():
            return True
    else:
        # generic HTTP 5xx bubbles as text sometimes
        s = str(e)
        if "HTTP 5" in s or "Service is unavailable" in s:
            return True
    return False
def _send_with_retry(create_call) -> Optional[str]:
    """
    create_call: lambda client -> Message
    returns sid or None
    """
    client = _ensure_client()
    last_err = None
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            msg = create_call(client)
            sid = getattr(msg, "sid", None)
            if sid:
                log.info("Twilio send OK sid=%s attempt=%d", sid, attempt)
            return sid or ""
        except Exception as e:
            last_err = e
            if not _is_retryable_twilio_error(e) or attempt == _MAX_ATTEMPTS:
                log.error("Twilio send failed (final) attempt=%d: %s", attempt, e)
                return None
            sleep_s = _BASE_BACKOFF * (2 ** (attempt - 1)) + random.uniform(0, 0.4)
            log.warning("Twilio send retry %d/%d in %.2fs: %s", attempt, _MAX_ATTEMPTS, sleep_s, e)
            time.sleep(sleep_s)
    return None
# --- Public API --------------------------------------------------------------


def send_whatsapp(to: str, body: Optional[str] = None, media_url: Optional[str] = None) -> List[str]:
    """
    Send a WhatsApp message using Twilio, splitting large text into multiple messages.
    Media (if provided) is attached to the final chunk only.
    Returns list of Twilio message SIDs (may be empty on failure).
    """
    to_formatted = _wa_prefix(to)
    from_formatted = _validate_wa_from(TWILIO_WHATSAPP_FROM)
    body = (body or "").strip()

    if not body and not media_url:
        log.info("Nothing to send: empty body and no media")
        return []

    # Build chunks (text) or single empty chunk (media-only)
    chunks = _split_text_preserve_paragraphs(body, max_len=_MAX_CHUNK_LEN) if body else [""]
    sids: List[str] = []

    for idx, chunk in enumerate(chunks):
        data = {"to": to_formatted}
        if chunk:
            data["body"] = chunk
        # attach media only to last chunk
        if media_url and idx == len(chunks) - 1:
            data["media_url"] = [media_url]

        def _create(client: Client):
            if TWILIO_MESSAGING_SERVICE_SID:
                return client.messages.create(messaging_service_sid=TWILIO_MESSAGING_SERVICE_SID, **data)
            else:
                return client.messages.create(from_=from_formatted, **data)

        sid = _send_with_retry(_create)
        if sid:
            sids.append(sid)
            log.info("WhatsApp sent sid=%s to=%s (chunk %d/%d)", sid, to_formatted, idx + 1, len(chunks))
        else:
            # Do NOT raise here → background flow should continue without looping error-sends
            log.error("WhatsApp send failed (dropped) to=%s (chunk %d/%d)", to_formatted, idx + 1, len(chunks))

        if idx != len(chunks) - 1:
            time.sleep(_SLEEP_BETWEEN)

    return sids

def send_sms(to: str, body: str) -> str:
    """
    SMS send with the same retry/backoff semantics.
    """
    to = (to or "").strip()
    if not to:
        raise RuntimeError("Empty SMS number")
    data = {"to": to, "body": body or ""}

    def _create(client: Client):
        if TWILIO_MESSAGING_SERVICE_SID:
            return client.messages.create(messaging_service_sid=TWILIO_MESSAGING_SERVICE_SID, **data)
        else:
            if not TWILIO_SMS_FROM:
                raise RuntimeError("No SMS sender configured. Set TWILIO_MESSAGING_SERVICE_SID or TWILIO_SMS_FROM")
            return client.messages.create(from_=TWILIO_SMS_FROM, **data)

    sid = _send_with_retry(_create)
    return sid or ""

def validate_twilio_request(flask_request) -> bool:
    """
    Keep your existing signature validation if you already have one.
    If not implemented, consider adding it. For now we simply return True.
    """
    try:
        return True
    except Exception as e:
        log.warning("validate_twilio_request failed: %s", e)
        return False
