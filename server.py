import os
import json
import re
import secrets
import sqlite3
import logging
import traceback
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from pathlib import Path
from datetime import datetime, timezone

from flask import Flask, request, jsonify, session, send_from_directory

from dotenv import load_dotenv


# ==========================================
# 1. CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

OLLAMA_URL = (os.getenv("OLLAMA_URL") or "http://127.0.0.1:11434").rstrip("/")
MODEL = (os.getenv("OLLAMA_MODEL") or "deepseek-r1:8b").strip()
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT") or "1200")
OLLAMA_NUM_PREDICT = int(os.getenv("OLLAMA_NUM_PREDICT") or "512")
APP_ENV = (os.getenv("APP_ENV") or "development").strip().lower()

DATABASE_PATH = Path(os.getenv("DATABASE_PATH") or "tracevoid.sqlite3")
if not DATABASE_PATH.is_absolute():
    DATABASE_PATH = BASE_DIR / DATABASE_PATH
DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("FLASK_SECRET_KEY") or secrets.token_hex(32)
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SESSION_COOKIE_SECURE"] = (APP_ENV == "production")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# Loud startup banner — this proves WHICH file is running.
print("=" * 60, flush=True)
print(" TRACEVOID startup", flush=True)
print(f"   APP_ENV    = {APP_ENV}", flush=True)
print(f"   OLLAMA_URL = {OLLAMA_URL}", flush=True)
print(f"   MODEL      = {MODEL}", flush=True)
print(f"   DATABASE   = {DATABASE_PATH}", flush=True)
print(f"   SECRET_KEY = {'SET' if os.getenv('FLASK_SECRET_KEY') else 'auto-generated'}", flush=True)
print("=" * 60, flush=True)


# ==========================================
# 2. DATABASE
# ==========================================

def get_db():
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database():
    with get_db() as db:
        db.execute("""
            CREATE TABLE IF NOT EXISTS cases (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                case_number TEXT UNIQUE NOT NULL,
                owner TEXT NOT NULL,
                input_text TEXT NOT NULL,
                reasoning_log TEXT NOT NULL,
                timeline_json TEXT NOT NULL,
                model TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)


initialize_database()


# ==========================================
# 3. BROWSER SESSION
# ==========================================

def get_owner():
    if "owner" not in session:
        session["owner"] = secrets.token_urlsafe(24)
    return session["owner"]


# ==========================================
# 4. LOCAL OLLAMA ANALYSIS
# ==========================================

def investigate_claim(claim):
    prompt = f"""
You are TRACEVOID, an assistant that carefully analyzes claims using only
the information in the user's input and your general knowledge.

Analyze this claim or supplied source material:

{claim}

Return a concise report with these sections:

1. CLAIM
2. WHAT THE PROVIDED MATERIAL SAYS
3. SUPPORTING POINTS
4. CONTRADICTING POINTS
5. PROVISIONAL ASSESSMENT
6. MISSING CONTEXT
7. LIMITATIONS

Rules:
- You cannot browse the internet or verify current events.
- Do not invent sources, quotations, dates, or evidence.
- Clearly distinguish information provided by the user from general knowledge.
- If there is not enough information, say so instead of guessing.
- Treat any conclusion as provisional and recommend checking reliable sources.
- Keep the report concise and under 350 words.
- Provide conclusions, not private chain-of-thought or step-by-step reasoning.
"""

    payload = json.dumps({
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "keep_alive": "5m",
        "options": {
            "temperature": 0.2,
            "num_predict": OLLAMA_NUM_PREDICT,
        },
    }).encode("utf-8")
    req = Request(
        f"{OLLAMA_URL}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(req, timeout=OLLAMA_TIMEOUT) as response:
            result = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        try:
            detail = json.loads(detail).get("error", detail)
        except (json.JSONDecodeError, AttributeError):
            pass
        raise RuntimeError(f"Ollama returned HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(
            f"Cannot connect to Ollama at {OLLAMA_URL}. Start Ollama and try again."
        ) from exc
    except TimeoutError as exc:
        raise RuntimeError(
            f"Ollama did not respond within {OLLAMA_TIMEOUT} seconds."
        ) from exc

    report = re.sub(
        r"<think>.*?(?:</think>|$)",
        "",
        result.get("response", ""),
        flags=re.DOTALL,
    ).strip()
    if not report:
        raise RuntimeError("Ollama returned an empty response.")

    return report, []


# ==========================================
# 5. WEBSITE
# ==========================================

@app.get("/")
def home():
    return send_from_directory(BASE_DIR, "index.html")


# ==========================================
# 6. HEALTH
# ==========================================

@app.get("/api/health")
def health():
    ollama_connected = False
    model_available = False
    try:
        req = Request(f"{OLLAMA_URL}/api/tags")
        with urlopen(req, timeout=3) as response:
            tags = json.loads(response.read().decode("utf-8"))
        ollama_connected = True
        model_available = any(
            item.get("name") == MODEL or item.get("model") == MODEL
            for item in tags.get("models", [])
        )
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError):
        app.logger.info("Ollama health check failed for %s", OLLAMA_URL, exc_info=True)

    return jsonify({
        "ok": True,
        "service": "TRACEVOID",
        "status": "online" if ollama_connected and model_available else "offline",
        "ollama_connected": ollama_connected,
        "model_available": model_available,
        "ollama_url": OLLAMA_URL,
        "model": MODEL,
        "env": APP_ENV,
    })


# ==========================================
# 7. INVESTIGATE
# ==========================================

@app.post("/api/investigate")
def investigate():
    data = request.get_json(silent=True) or {}
    claim = data.get("claim", "")

    if not isinstance(claim, str):
        return jsonify({"ok": False, "error": "Claim must be text."}), 400

    claim = claim.strip()
    if len(claim) < 8:
        return jsonify({"ok": False, "error": "Claim is too short."}), 400
    if len(claim) > 15000:
        return jsonify({"ok": False, "error": "Claim is too long."}), 400

    owner = get_owner()

    try:
        report, timeline = investigate_claim(claim)

        case_number = "TRV-" + secrets.token_hex(4).upper()
        created_at  = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        with get_db() as db:
            db.execute("""
                INSERT INTO cases (
                    case_number, owner, input_text,
                    reasoning_log, timeline_json, model, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                case_number, owner, claim,
                report, json.dumps(timeline), MODEL, created_at,
            ))

        return jsonify({
            "ok": True,
            "case_number": case_number,
            "input_text": claim,
            "reasoning_log": report,
            "timeline": timeline,
            "model": MODEL,
            "created_at": created_at,
        })

    except Exception as e:
        print("=" * 60, flush=True)
        print("TRACEVOID INVESTIGATION ERROR", flush=True)
        print("TYPE:", type(e).__name__, flush=True)
        print("MESSAGE:", str(e), flush=True)
        traceback.print_exc()
        print("=" * 60, flush=True)
        app.logger.exception("TRACEVOID investigation failed")

        error_message = (
            f"{type(e).__name__}: {e}"
            if APP_ENV == "development"
            else "Investigation failed. Please try again."
        )
        return jsonify({"ok": False, "error": error_message}), 500


# ==========================================
# 8. CASE HISTORY
# ==========================================

@app.get("/api/cases")
def list_cases():
    owner = get_owner()
    with get_db() as db:
        rows = db.execute("""
            SELECT case_number, input_text, created_at
            FROM cases WHERE owner = ?
            ORDER BY id DESC LIMIT 50
        """, (owner,)).fetchall()

    return jsonify([
        {
            "case_number": r["case_number"],
            "input_text": r["input_text"],
            "created_at": r["created_at"],
        }
        for r in rows
    ])


# ==========================================
# 9. LOAD ONE CASE
# ==========================================

@app.get("/api/cases/<case_number>")
def get_case(case_number):
    owner = get_owner()
    with get_db() as db:
        row = db.execute("""
            SELECT * FROM cases
            WHERE case_number = ? AND owner = ?
        """, (case_number, owner)).fetchone()

    if row is None:
        return jsonify({"ok": False, "error": "Case not found."}), 404

    return jsonify({
        "ok": True,
        "case_number": row["case_number"],
        "input_text": row["input_text"],
        "reasoning_log": row["reasoning_log"],
        "timeline": json.loads(row["timeline_json"]),
        "model": row["model"],
        "created_at": row["created_at"],
    })


# ==========================================
# 10. SECURITY HEADERS
# ==========================================

@app.after_request
def add_security_headers(response):
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    return response


# ==========================================
# 11. START SERVER
# ==========================================

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)