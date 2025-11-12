# app/api.py
import os
from flask import Blueprint, request, Response, send_from_directory, current_app
from flask_restx import Api, Resource, fields
from .services.vector_store import ListingStore
from .services.agent import handle_message, handle_selection, handle_contact, ensure_session
from xml.sax.saxutils import escape as xmlesc
from .services.line_io import validate_line_signature, reply_text, push_text
from .services.twilio_io import validate_twilio_request, send_whatsapp, send_sms
import threading
import logging
from typing import Optional, Dict
import time


log = logging.getLogger(__name__)
api_bp = Blueprint("api", __name__)
api = Api(api_bp, version="1.0", title="Apartment Agent Demo", doc="/", description="Example data + OpenAI + Twilio webhook")
media_bp = Blueprint("media", __name__, url_prefix="/static/wa-media")
def _resolve_media_dir() -> str:
    # Prefer ENV; fallback to <app.root_path>/static/wa-media
    env_dir = (os.getenv("MEDIA_LOCAL_DIR") or "").strip()
    if env_dir:
        return os.path.abspath(env_dir)
    return os.path.abspath(os.path.join(current_app.root_path, "static", "wa-media"))

@media_bp.route("/<path:filename>", methods=["GET", "HEAD"])
def serve_wa_media(filename):
    current_app.logger.info(f"[wa-media] hit: {filename}")
    base_dir = _resolve_media_dir()
    resp = send_from_directory(base_dir, filename, conditional=True, max_age=31536000)
    resp.headers["X-Served-By"] = "flask-media-bp"
    return resp

listing_model = api.model("Listing", {
    "id": fields.String, "url": fields.String, "title": fields.String,
    "description": fields.String, "price_usd": fields.Integer, "bedrooms": fields.Integer,
    "bathrooms": fields.Float, "pet_dog": fields.Boolean, "neighborhood": fields.String,
    "city": fields.String, "deal_type": fields.String, "images": fields.List(fields.String),
    "source": fields.String, "_score": fields.Float
})




@api.route("/dev/load_examples")
class LoadExamples(Resource):
    def post(self):
        # Try to locate `seed_examples` in common module locations
        mods_to_try = [
            "app.dev.seed",
            "app.seed",
            "dev.seed",
            "seed",  # root-level seed.py
        ]
        seed_fn = None
        tried = []
        for mod in mods_to_try:
            try:
                m = __import__(mod, fromlist=["seed_examples"])
                seed_fn = getattr(m, "seed_examples", None)
                if callable(seed_fn):
                    break
                tried.append(f"{mod} (no seed_examples)")
            except Exception as e:
                tried.append(f"{mod} ({e.__class__.__name__}: {e})")
                continue

        if not callable(seed_fn):
            # Return a clear JSON error instead of a 500 trace
            return {
                "status": "error",
                "message": "Could not find function `seed_examples` in any known module.",
                "tried": tried
            }, 500

        try:
            n = seed_fn()
            return {"status": "ok", "seeded": n}, 200
        except Exception as e:
            # If your seed uses flexible imports and fails, report cleanly.
            return {
                "status": "error",
                "message": f"seed_examples raised: {e.__class__.__name__}: {e}"
            }, 500
from .dev.seed_from_rds import main as seed_rds_main

@api.route("/dev/seed_rds", methods=["POST"])
class SeedRDS(Resource):
    def post(self):
        try:
            added = seed_rds_main()
            return {"status": "ok", "added": int(added)}, 200
        except Exception as e:
            import traceback
            return {
                "status": "error",
                "message": str(e),
                "trace_tail": traceback.format_exc().splitlines()[-12:]
            }, 500
@api.route("/listings/search")
class Listings(Resource):
    @api.marshal_list_with(listing_model, code=200)
    def get(self):
        q = request.args.get("q","")
        store = ListingStore()
        return store.search(q or "apartment", n=20), 200

chat_in = api.model("ChatIn", {"user_id": fields.String, "message": fields.String})
@api.route("/chat")
class Chat(Resource):
    @api.expect(chat_in)
    def post(self):
        body = request.get_json(force=True)
        user = body.get("user_id", "demo-user")
        msg = body.get("message", "")
        if any(w in msg.lower() for w in ["first","second","third","1","2","3","4","5"]):
            text = handle_selection(user, msg); return {"reply": text}
        if any(w in msg.lower() for w in ["email","reach me","@","gmail","phone","contact"]):
            text = handle_contact(user, msg); return {"reply": text}
        text, media = handle_message(user, msg); return {"reply": text}

import re
def _twiml_response(text: str) -> Response:
    # Make text ASCII-safe for WhatsApp + XML
    # (Twilio supports UTF-8, but some clients choke on certain punctuation)
    safe = (text or "").replace("—", "-").replace("–", "-")
    # Preserve line breaks in a way Twilio/WA renders consistently
    # (plain \n generally works, but encoding them avoids edge-cases)
    safe = safe.replace("\r\n", "\n").replace("\r", "\n")
    safe = xmlesc(safe)

    twiml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Response><Message><Body>' + safe + '</Body></Message></Response>'
    )
    # Build a Response with explicit headers
    resp = Response(twiml.encode("utf-8"))
    resp.headers["Content-Type"] = "text/xml; charset=utf-8"
    resp.headers["Content-Length"] = str(len(twiml.encode("utf-8")))
    return resp

_IDEMP_CACHE: Dict[str, float] = {}      # MessageSid -> epoch
_IDEMP_TTL_SECONDS = 60 * 5              # 5 minutes
_IDEMP_LOCK = threading.Lock()

def _idem_seen(message_sid: Optional[str]) -> bool:
    if not message_sid:
        return False
    now = time.time()
    with _IDEMP_LOCK:
        # cleanup old
        expired = [sid for sid, ts in _IDEMP_CACHE.items() if now - ts > _IDEMP_TTL_SECONDS]
        for sid in expired:
            _IDEMP_CACHE.pop(sid, None)
        if message_sid in _IDEMP_CACHE:
            return True
        _IDEMP_CACHE[message_sid] = now
        return False

# -------------------------
# Compose a single, batched WhatsApp message
# -------------------------
def _format_single_message(clean_results: list) -> str:
    """Return one compact message containing up to 5 results, single outbound send."""
    if not clean_results:
        return ("Sorry — I couldn’t find strong matches fast enough.\n"
                "Tip: share nearest station or raise budget slightly (e.g., ¥60,000).")
    lines = ["Here are some options:"]
    for i, r in enumerate(clean_results[:5], 1):
        title = r.get("title") or "Untitled"
        price = ""
        if "price_yen" in r and isinstance(r["price_yen"], (int, float)):
            price = f" — ¥{int(r['price_yen']):,}/mo"
        elif r.get("price"):
            price = f" — {r['price']}"
        hood = r.get("neighborhood") or r.get("ward") or r.get("city") or r.get("prefecture") or ""
        url = r.get("url") or ""
        line = f"{i}) {title}{price}"
        if hood:
            line += f" — {hood}"
        if url:
            line += f"\n{url}"
        lines.append(line)
    lines.append("\nReply with the numbers of any options you like (e.g., '1 and 3').")
    return "\n".join(lines)

# -------------------------
# Background processing (NO immediate receipt → cost stays flat)
# -------------------------
def _process_and_respond_async(sender: str, body: str):
    """
    Heavy flow (LLM + vector search + normalization) in a background thread.
    Sends exactly ONE final WhatsApp message (batched).
    """
    try:
        # Determine intent quickly
        is_selection = bool(re.search(r"\b([1-9])\b", body))
        if is_selection:
            sess = ensure_session(sender)
            if not sess.get("last_results"):
                reply = "I don't have any options stored yet. Please ask me to search apartments first."
                send_whatsapp(to=sender, body=reply)
                return
            reply = handle_selection(sender, body)
            # Selection handler usually returns final composed text already:
            send_whatsapp(to=sender, body=reply)
            return

        if any(w in body.lower() for w in ["email", "@", "phone", "call", "contact"]):
            reply = handle_contact(sender, body)
            send_whatsapp(to=sender, body=reply)
            return

        # Main path: search + normalize
        # handle_message is your heavy pipeline; it should return (reply_text, debug_obj)
        reply, debug_obj = handle_message(sender, body)

        # If your handle_message returns raw hits in debug_obj, you can reformat to a single batched message:
        # Expecting debug_obj.get("clean_results") or debug_obj.get("results")
        clean = None
        if isinstance(debug_obj, dict):
            clean = debug_obj.get("clean_results") or debug_obj.get("results")

        if isinstance(clean, list) and clean:
            reply = _format_single_message(clean)

        if not reply or not isinstance(reply, str):
            reply = "Sorry — I couldn't compose a response just now."

        # Send ONE final message
        send_whatsapp(to=sender, body=reply)

    except Exception as e:
        log.exception("Background processing failed: %s", e)
        try:
            send_whatsapp(to=sender, body="Hmm, something went wrong fetching listings. Please try again.")
        except Exception:
            log.exception("Failed to send error message to user")

# -------------------------
# Webhook route (fast ACK, no extra message)
# -------------------------
@api.route("/twilio/webhook", methods=["POST"])
class TwilioWebhook(Resource):
    def post(self):
        if not validate_twilio_request(request):
            # Return TwiML with a short message or empty — short is fine here.
            return _twiml_response("Invalid signature.")

        sender = request.form.get("From", "user")
        body = request.form.get("Body", "") or ""
        message_sid = request.form.get("MessageSid")

        # Debug
        try:
            log.info("Twilio IN: sid=%s from=%s body=%r", message_sid, sender, body)
            print(f"DEBUG Twilio IN: sid={message_sid} from={sender} body={body!r}")
        except Exception:
            pass

        # Idempotency: ignore Twilio retries
        if _idem_seen(message_sid):
            log.info("Duplicate webhook (retry) ignored: %s", message_sid)
            return _twiml_response("")  # empty TwiML

        # Launch background worker — NO immediate receipt (keeps cost flat)
        th = threading.Thread(target=_process_and_respond_async, args=(sender, body), daemon=True)
        th.start()

        # Return fast to avoid sandbox timeout
        return _twiml_response("")  # empty → no extra Twilio message here

# -------------------------
# Outbound send test route
# -------------------------
send_in = api.model("SendIn", {"to": fields.String, "message": fields.String, "channel": fields.String})

@api.route("/twilio/send", methods=["POST"])
class TwilioSend(Resource):
    @api.expect(send_in)
    def post(self):
        data = request.get_json(force=True) or {}
        to = data.get("to")
        body = data.get("message", "Hello from Apartment Agent")
        channel = (data.get("channel", "sms") or "sms").lower()

        if not to:
            return {"error": "Missing 'to' (E.164 +123..., or 'whatsapp:+123...')"}, 400

        try:
            if channel == "whatsapp":
                sid = send_whatsapp(to, body)
            else:
                sid = send_sms(to, body)
        except Exception as e:
            log.exception("Send failed: %s", e)
            return {"status": "error", "message": str(e)}, 500

        return {"status": "ok", "sid": sid}
# NOTE: kept the route but made it Pinecone-safe for easy debugging
@api.route("/debug/chroma")
class VectorDebug(Resource):
    def get(self):
        store = ListingStore()
        return {
            "backend": "pinecone",
            "index": store.index_name,
            "namespace": store.namespace,
            "count": store.count(),
            "sample": (store.all(limit=1) or [None])[0],
            "probe_results": store.search("渋谷 2LDK ペット可", n=3)
        }

@api.route("/debug/listings")
class DebugListings(Resource):
    def get(self):
        store = ListingStore(collection="listings_openai")
        listings = store.all(limit=10)
        return {
            "count": store.count(),
            "sample": listings[:3]
        }

@api.route("/debug/search")
class DebugSearch(Resource):
    def get(self):
        query = request.args.get('q', 'shibuya apartment')
        store = ListingStore(collection="listings_openai")

        try:
            results = store.search(query, n=12)
            try:
                count = store.count()
            except Exception as e:
                count = f"error: {str(e)}"

            return {
                "query": query,
                "index": store.index_name,
                "namespace": store.namespace,
                "count": count,
                "results_found": len(results),
                "results": results,
                "status": "success"
            }
        except Exception as e:
            return {
                "error": str(e),
                "query": query,
                "status": "error"
            }, 500

@api.route("/listings/all")
class AllListings(Resource):
    @api.marshal_list_with(listing_model, code=200)
    def get(self):
        try:
            store = ListingStore()
            all_listings = store.all(limit=1000)
            return all_listings, 200
        except Exception as e:
            api.logger.error(f"Error fetching all listings: {str(e)}")
            return {"error": "Failed to fetch listings"}, 500

@api.route("/line/webhook", methods=["POST"])
class LineWebhook(Resource):
    def post(self):
        # signature check
        if not validate_line_signature(request):
            return Response("invalid signature", status=403)

        body = request.get_json(force=True, silent=True) or {}
        events = body.get("events", [])

        for ev in events:
            etype = ev.get("type")
            reply_token = ev.get("replyToken")
            source = ev.get("source", {})
            user_id = source.get("userId", "line-user")

            if etype == "message" and ev.get("message", {}).get("type") == "text":
                text_in = ev["message"]["text"] or ""
                low = text_in.lower()

                if any(w in low for w in ["first","second","third","1","2","3","4","5"]):
                    reply = handle_selection(user_id, text_in)
                elif any(w in low for w in ["email","@","phone","call","contact"]):
                    reply = handle_contact(user_id, text_in)
                else:
                    reply, _ = handle_message(user_id, text_in)

                if reply_token:
                    reply_text(reply_token, reply)

            elif etype == "postback":
                if reply_token:
                    reply_text(reply_token, "Got your selection.")
        return "ok"

push_model = api.model("LinePushIn", {"to": fields.String, "message": fields.String})
@api.route("/line/push", methods=["POST"])
class LinePush(Resource):
    @api.expect(push_model)
    def post(self):
        data = request.get_json(force=True)
        to = data.get("to")
        msg = data.get("message","Hello from LINE Agent")
        if not to:
            return {"error":"Missing 'to' (userId)"}, 400
        res = push_text(to, msg)
        return {"status":"ok","line":res}