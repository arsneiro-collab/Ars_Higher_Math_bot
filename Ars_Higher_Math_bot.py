# Ars_Higher_Math_bot.py
# Telegram tutor bot for higher math + OCR + trial, ready for Railway (webhooks).

import os
import io
import time
import base64
import asyncio
import logging
import sqlite3
import httpx
from contextlib import closing
from datetime import datetime, timezone

from telegram import (
    Update,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    InputFile,
)
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ContextTypes,
    filters,
)

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("math_tutor_bot")

# ---------- Helpers ----------
def _mask(s: str) -> str:
    return (s[:8] + "..." + s[-4:]) if s and len(s) > 12 else ("set" if s else "none")

def _clean_key(s: str) -> str:
    if not s:
        return s
    s = s.strip().strip('"').strip("'")
    for bad in ("\u200b", "\u200c", "\u200d", "\u2060", "\ufeff"):
        s = s.replace(bad, "")
    return s

# ---------- ENV ----------
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()

# Prioritized API key (FORCE overrides any other sources)
OPENAI_API_KEY = _clean_key(os.getenv("OPENAI_API_KEY_FORCE") or os.getenv("OPENAI_API_KEY", ""))
OPENAI_PROJECT  = os.getenv("OPENAI_PROJECT", "").strip()

MODEL         = os.getenv("MODEL", "gpt-4o-mini").strip()
VISION_MODEL  = os.getenv("VISION_MODEL", os.getenv("MODEL", "gpt-4o-mini")).strip()

USE_WEBHOOK = os.getenv("USE_WEBHOOK", "true").lower() == "true"
WEBHOOK_URL = os.getenv("WEBHOOK_URL", "").strip()
PORT        = int(os.getenv("PORT", "8080"))

HISTORY_DB  = os.getenv("HISTORY_DB", "history.db")
MAX_HISTORY = int(os.getenv("MAX_HISTORY", "20"))
WELCOME_IMAGE_PATH = os.getenv("WELCOME_IMAGE_PATH", "welcome.png")

TRIAL_HOURS = float(os.getenv("TRIAL_HOURS", "3"))
SILENT_AFTER_EXPIRE = os.getenv("SILENT_AFTER_EXPIRE", "true").lower() == "true"

ALLOWED_USERNAMES = set(name.strip().lstrip("@").lower() for name in os.getenv("ALLOWED_USERNAMES", "ars_telegramm,ArsMovses").split(",") if name.strip())
ALLOWED_USER_IDS  = set(int(x) for x in os.getenv("ALLOWED_USER_IDS", "").split(",") if x.strip().isdigit())

# OCR config
OCR_MODE      = os.getenv("OCR_MODE", "vision")  # vision | tesseract
OCR_LANG      = os.getenv("OCR_LANG", "rus+eng")

# Preprocess config
OCR_PREPROCESS = os.getenv("OCR_PREPROCESS", "true").lower() == "true"
PRE_GRAYSCALE  = os.getenv("PRE_GRAYSCALE", "true").lower() == "true"
PRE_BRIGHTNESS = float(os.getenv("PRE_BRIGHTNESS", "1.0"))
PRE_CONTRAST   = float(os.getenv("PRE_CONTRAST", "1.35"))
PRE_SHARPNESS  = float(os.getenv("PRE_SHARPNESS", "1.1"))
PRE_UNSHARP    = os.getenv("PRE_UNSHARP", "true").lower() == "true"
PRE_BINARIZE   = os.getenv("PRE_BINARIZE", "false").lower() == "true"
PRE_BIN_THRESHOLD = int(os.getenv("PRE_BIN_THRESHOLD", "180"))

logger.info("ENV check → MODEL=%s | VISION_MODEL=%s | OPENAI_API_KEY=%s | OPENAI_PROJECT=%s",
            MODEL, VISION_MODEL, _mask(OPENAI_API_KEY), OPENAI_PROJECT or "—")

# ---------- OpenAI ----------
try:
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY, project=OPENAI_PROJECT or None) if OPENAI_API_KEY else None
except Exception as e:
    client = None
    logger.warning("OpenAI SDK not available: %s", e)

# ---------- PIL / OCR ----------
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
try:
    import pytesseract
except Exception:
    pytesseract = None

# ---------- DB ----------
def init_db() -> None:
    with closing(sqlite3.connect(HISTORY_DB)) as conn:
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS messages (
              chat_id INTEGER NOT NULL,
              ts TEXT NOT NULL,
              role TEXT NOT NULL,
              content TEXT NOT NULL
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
              user_id   INTEGER PRIMARY KEY,
              owner_id  INTEGER NOT NULL,
              start_ts  INTEGER NOT NULL,
              expired   INTEGER NOT NULL DEFAULT 0,
              notified  INTEGER NOT NULL DEFAULT 0
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS settings (
              chat_id  INTEGER PRIMARY KEY,
              ocr_mode TEXT
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS ocr_logs (
              id           INTEGER PRIMARY KEY AUTOINCREMENT,
              chat_id      INTEGER NOT NULL,
              user_id      INTEGER NOT NULL,
              ts           TEXT    NOT NULL,
              mode         TEXT    NOT NULL,
              tg_file_id   TEXT,
              content_type TEXT,
              bytes        BLOB,
              text         TEXT
            )
        """)
        conn.commit()

def save_message(chat_id: int, role: str, content: str) -> None:
    with closing(sqlite3.connect(HISTORY_DB)) as conn, closing(conn.cursor()) as cur:
        cur.execute(
            "INSERT INTO messages (chat_id, ts, role, content) VALUES (?, ?, ?, ?)",
            (chat_id, datetime.now(timezone.utc).isoformat(), role, content)
        )
        conn.commit()

def get_history(chat_id: int, limit: int = MAX_HISTORY):
    with closing(sqlite3.connect(HISTORY_DB)) as conn, closing(conn.cursor()) as cur:
        cur.execute("SELECT role, content FROM messages WHERE chat_id=? ORDER BY ts DESC LIMIT ?", (chat_id, limit))
        rows = cur.fetchall()
    return list(reversed(rows))

def clear_history(chat_id: int) -> None:
    with closing(sqlite3.connect(HISTORY_DB)) as conn, closing(conn.cursor()) as cur:
        cur.execute("DELETE FROM messages WHERE chat_id=?", (chat_id,))
        conn.commit()

# Users / Trial
def ensure_user(user_id: int) -> None:
    with closing(sqlite3.connect(HISTORY_DB)) as conn, closing(conn.cursor()) as cur:
        cur.execute("SELECT user_id FROM users WHERE user_id=?", (user_id,))
        row = cur.fetchone()
        if row is None:
            cur.execute(
                "INSERT INTO users (user_id, owner_id, start_ts, expired, notified) VALUES (?, ?, ?, 0, 0)",
                (user_id, user_id, int(time.time()))
            )
            conn.commit()

def get_user(user_id: int):
    with closing(sqlite3.connect(HISTORY_DB)) as conn, closing(conn.cursor()) as cur:
        cur.execute("SELECT user_id, owner_id, start_ts, expired, notified FROM users WHERE user_id=?", (user_id,))
        return cur.fetchone()

def set_user_expired(user_id: int) -> None:
    with closing(sqlite3.connect(HISTORY_DB)) as conn, closing(conn.cursor()) as cur:
        cur.execute("UPDATE users SET expired=1 WHERE user_id=?", (user_id,))
        conn.commit()

def set_user_notified(user_id: int) -> None:
    with closing(sqlite3.connect(HISTORY_DB)) as conn, closing(conn.cursor()) as cur:
        cur.execute("UPDATE users SET notified=1 WHERE user_id=?", (user_id,))
        conn.commit()

def reset_user_trial(user_id: int) -> None:
    now = int(time.time())
    with closing(sqlite3.connect(HISTORY_DB)) as conn, closing(conn.cursor()) as cur:
        cur.execute("SELECT user_id FROM users WHERE user_id=?", (user_id,))
        if cur.fetchone() is None:
            cur.execute("INSERT INTO users (user_id, owner_id, start_ts, expired, notified) VALUES (?, ?, ?, 0, 0)", (user_id, user_id, now))
        else:
            cur.execute("UPDATE users SET start_ts=?, expired=0, notified=0 WHERE user_id=?", (now, user_id))
        conn.commit()

# Settings & OCR logs
def sanitize_ocr_mode(mode: str) -> str:
    m = (mode or "").strip().lower()
    if m.startswith("tess"):
        return "tesseract"
    if m.startswith("vis") or m == "":
        return "vision"
    return m

def get_effective_ocr_mode(chat_id: int) -> str:
    try:
        with closing(sqlite3.connect(HISTORY_DB)) as conn, closing(conn.cursor()) as cur:
            cur.execute("SELECT ocr_mode FROM settings WHERE chat_id=?", (chat_id,))
            row = cur.fetchone()
            if row and row[0]:
                return sanitize_ocr_mode(row[0])
    except Exception:
        logger.exception("get_effective_ocr_mode error")
    return sanitize_ocr_mode(OCR_MODE)

def set_chat_ocr_mode(chat_id: int, mode: str) -> None:
    try:
        m = sanitize_ocr_mode(mode)
        with closing(sqlite3.connect(HISTORY_DB)) as conn, closing(conn.cursor()) as cur:
            cur.execute("INSERT INTO settings (chat_id, ocr_mode) VALUES (?, ?) ON CONFLICT(chat_id) DO UPDATE SET ocr_mode=excluded.ocr_mode", (chat_id, m))
            conn.commit()
    except Exception:
        logger.exception("set_chat_ocr_mode error")

def log_ocr_event(chat_id: int, user_id: int, mode: str, tg_file_id: str, content_type: str, img_bytes: bytes, text: str) -> None:
    try:
        with closing(sqlite3.connect(HISTORY_DB)) as conn, closing(conn.cursor()) as cur:
            cur.execute(
                "INSERT INTO ocr_logs (chat_id, user_id, ts, mode, tg_file_id, content_type, bytes, text) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (chat_id, user_id, datetime.now(timezone.utc).isoformat(), mode, tg_file_id, content_type, img_bytes, text)
            )
            conn.commit()
    except Exception:
        logger.exception("log_ocr_event error")

# ---------- Role & OpenAI ----------
TUTOR_PROMPT = (
    "Ты — репетитор по высшей математике для студентов из России. Общайся дружелюбно и доступно.\n"
    "Цель: помочь повысить знание теории и практики.\n"
    "Правила:\n"
    "- Объясняй простыми словами, используй жизненные примеры.\n"
    "- Если студент просит решить задачу, не давай сразу ответ: сначала укажи тему, затем шаги без решения; предложи попробовать самому.\n"
    "- Если студент ошибается — мягко укажи на ошибку и помоги исправить.\n"
    "- В конце каждого объяснения предложи небольшую практическую задачу для закрепления.\n"
    "- Избегай политики/медицины/психологии и пр.; только вузовская математика по стандартным программам.\n"
)

def build_messages(chat_id: int, user_text: str):
    hist = get_history(chat_id, MAX_HISTORY)
    messages = [{"role": "system", "content": TUTOR_PROMPT}]
    for role, content in hist:
        messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": user_text})
    return messages

def call_openai(messages: list) -> str:
    if client is None:
        return "Сейчас модель недоступна (нет ключа OPENAI_API_KEY). Попробуй позже."
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.3,
            max_tokens=800
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception:
        logger.exception("OpenAI error")
        return "Произошла ошибка при обращении к модели. Попробуй ещё раз."

# ---------- TeX rendering ----------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def render_tex_to_png(tex: str) -> bytes:
    fig = plt.figure(figsize=(6, 1.2), dpi=200)
    fig.patch.set_alpha(0.0)
    ax = plt.axes([0,0,1,1])
    ax.axis("off")
    ax.text(0.5, 0.5, f"${tex}$", ha="center", va="center", fontsize=22)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", transparent=True, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    buf.seek(0)
    return buf.read()

# ---------- OCR & preprocessing ----------
def preprocess_image_for_ocr(image_bytes: bytes) -> bytes:
    if not OCR_PREPROCESS:
        return image_bytes
    try:
        im = Image.open(io.BytesIO(image_bytes))
        try:
            im = ImageOps.exif_transpose(im)  # auto-rotate
        except Exception:
            pass
        if PRE_GRAYSCALE:
            im = im.convert("L")
        else:
            im = im.convert("RGB")
        if abs(PRE_BRIGHTNESS - 1.0) > 1e-3:
            im = ImageEnhance.Brightness(im).enhance(PRE_BRIGHTNESS)
        if abs(PRE_CONTRAST - 1.0) > 1e-3:
            im = ImageEnhance.Contrast(im).enhance(PRE_CONTRAST)
        if PRE_UNSHARP:
            im = im.filter(ImageFilter.UnsharpMask(radius=1.2, percent=140, threshold=3))
        if abs(PRE_SHARPNESS - 1.0) > 1e-3:
            im = ImageEnhance.Sharpness(im).enhance(PRE_SHARPNESS)
        if PRE_BINARIZE:
            if im.mode != "L":
                im = im.convert("L")
            im = im.point(lambda p: 255 if p >= PRE_BIN_THRESHOLD else 0, mode="1").convert("L")
        out = io.BytesIO()
        im.save(out, format="PNG", optimize=True)
        out.seek(0)
        return out.read()
    except Exception:
        logger.exception("preprocess_image_for_ocr error")
        return image_bytes

def _encode_image_b64(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode("utf-8")

def ocr_with_vision(image_bytes: bytes) -> str:
    if client is None:
        return ""
    try:
        b64 = _encode_image_b64(image_bytes)
        resp = client.chat.completions.create(
            model=VISION_MODEL,
            messages=[
                {"role": "system", "content": "Извлеки текст математической задачи с фото. Верни только текст без комментариев."},
                {"role": "user",
                 "content": [
                    {"type": "text", "text": "Только текст задачи. Если текста нет — верни пустую строку."},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                 ]},
            ],
            temperature=0.0,
            max_tokens=800,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception:
        logger.exception("Vision OCR error")
        return ""

def ocr_with_tesseract(image_bytes: bytes) -> str:
    if pytesseract is None:
        return ""
    try:
        im = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return (pytesseract.image_to_string(im, lang=OCR_LANG) or "").strip()
    except Exception:
        logger.exception("Tesseract OCR error")
        return ""

def ocr_extract_bytes(image_bytes: bytes, mode: str | None = None) -> str:
    processed = preprocess_image_for_ocr(image_bytes)
    text = ""
    m = sanitize_ocr_mode(mode or OCR_MODE)
    if m == "vision":
        text = ocr_with_vision(processed) or ocr_with_tesseract(processed)
    else:
        text = ocr_with_tesseract(processed) or ocr_with_vision(processed)
    return (text or "").strip()

# ---------- Trial gate ----------
def _is_admin(user) -> bool:
    if not user:
        return False
    uname = (user.username or "").lower()
    return (uname in ALLOWED_USERNAMES) or (user.id in ALLOWED_USER_IDS)

async def check_trial_and_maybe_stop(update: Update) -> bool:
    user = update.effective_user
    if user is None:
        return False
    uid = user.id
    uname = (user.username or "").lower()
    if uname in ALLOWED_USERNAMES or uid in ALLOWED_USER_IDS:
        return False

    ensure_user(uid)
    row = get_user(uid)
    if not row:
        return False
    _, _, start_ts, expired, notified = row
    now = int(time.time())
    limit = int(TRIAL_HOURS * 3600)

    if expired or (now - start_ts >= limit):
        if not expired:
            set_user_expired(uid)
        if not notified:
            try:
                await update.effective_message.reply_text("Тестовый период закончился")
                kb_author = InlineKeyboardMarkup([[InlineKeyboardButton("Связаться с автором", url="https://t.me/ars_telegramm")]])
                await update.effective_message.reply_text("По вопросам приобретения обращайтесь к автору.", reply_markup=kb_author)
                kb_id = InlineKeyboardMarkup([
                    [InlineKeyboardButton("Ваш ID", callback_data="myid")],
                    [InlineKeyboardButton("Сообщить Ваш ID админу", url="https://t.me/ars_telegramm")]
                ])
                await update.effective_message.reply_text("Для продления времени тестирования сообщите админу Ваш ID", reply_markup=kb_id)
            except Exception:
                pass
            set_user_notified(uid)
        return SILENT_AFTER_EXPIRE
    return False

# ---------- Handlers ----------
async def diag(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    masked = _mask(OPENAI_API_KEY)

    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    if OPENAI_PROJECT:
        headers["OpenAI-Project"] = OPENAI_PROJECT

    # GET /v1/models
    try:
        async with httpx.AsyncClient(timeout=15.0) as ac:
            r = await ac.get("https://api.openai.com/v1/models", headers=headers)
        m_status = r.status_code
        m_body = r.text[:250]
    except Exception as e:
        m_status = f"error:{type(e).__name__}"
        m_body = str(e)

    # POST /v1/chat/completions
    try:
        async with httpx.AsyncClient(timeout=20.0) as ac:
            r2 = await ac.post(
                "https://api.openai.com/v1/chat/completions",
                headers={**headers, "Content-Type": "application/json"},
                json={"model": MODEL, "messages":[{"role":"user","content":"ping"}], "max_tokens":5, "temperature":0}
            )
        c_status = r2.status_code
        c_body = r2.text[:250]
    except Exception as e:
        c_status = f"error:{type(e).__name__}"
        c_body = str(e)

    info = (
        f"Диагностика OpenAI\n"
        f"- KEY: {masked}\n"
        f"- MODEL: {MODEL}\n"
        f"- PROJECT: {OPENAI_PROJECT or '—'}\n"
        f"- /v1/models → {m_status}\n{m_body}\n"
        f"- /v1/chat/completions → {c_status}\n{c_body}\n"
    )
    await update.effective_message.reply_text(info)

async def env_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_admin(update.effective_user):
        return
    txt = (
        "ENV runtime:\n"
        f"- MODEL={MODEL}\n"
        f"- VISION_MODEL={VISION_MODEL}\n"
        f"- OPENAI_API_KEY={_mask(OPENAI_API_KEY)}\n"
        f"- OPENAI_PROJECT={OPENAI_PROJECT or '—'}\n"
    )
    await update.effective_message.reply_text(txt)

from openai import OpenAI as _OpenAI  # for reinit in set_project

async def set_project(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_admin(update.effective_user):
        await update.effective_message.reply_text("Недостаточно прав.")
        return
    if not context.args:
        await update.effective_message.reply_text("Использование: /set_project proj_xxxxx")
        return
    new_proj = context.args[0].strip()
    if not new_proj.startswith("proj_"):
        await update.effective_message.reply_text("Неверный формат. Нужен ID вида proj_...")
        return

    global OPENAI_PROJECT, client
    OPENAI_PROJECT = new_proj
    os.environ["OPENAI_PROJECT"] = new_proj
    try:
        client = _OpenAI(api_key=OPENAI_API_KEY, project=OPENAI_PROJECT or None) if OPENAI_API_KEY else None
        await update.effective_message.reply_text(f"OPENAI_PROJECT установлен: {OPENAI_PROJECT}")
    except Exception as e:
        await update.effective_message.reply_text(f"Ошибка инициализации клиента: {type(e).__name__}")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    init_db()
    if await check_trial_and_maybe_stop(update):
        return
    msg = update.effective_message
    chat_id = update.effective_chat.id

    if os.path.exists(WELCOME_IMAGE_PATH):
        try:
            with open(WELCOME_IMAGE_PATH, "rb") as f:
                await msg.reply_photo(InputFile(f, filename="welcome.png"))
        except Exception:
            logger.exception("Failed to send welcome.png")

    save_message(chat_id, "assistant",
        "Привет! Я твой репетитор по высшей математике. Называй меня просто — Профессор. С чего начнём?"
    )
    await msg.reply_text("Привет! Я твой репетитор по высшей математике. Называй меня просто — Профессор. С чего начнём?")

async def welcome_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    init_db()
    if await check_trial_and_maybe_stop(update):
        return
    if os.path.exists(WELCOME_IMAGE_PATH):
        try:
            with open(WELCOME_IMAGE_PATH, "rb") as f:
                await update.effective_message.reply_photo(InputFile(f, filename="welcome.png"))
        except Exception:
            logger.exception("Failed to send welcome.png")

async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    init_db()
    if await check_trial_and_maybe_stop(update):
        return
    text = (
        "/start — начать\n"
        "/help — помощь\n"
        "/reset — очистить историю диалога\n"
        "/welcome — показать приветственную картинку\n"
        "/myid — показать ваш Telegram ID\n"
        "/ocr_mode — показать/сменить OCR-режим (vision|tesseract)\n"
        "/tex <формула> — рендер LaTeX, пример: /tex \\int_0^1 x^2 dx\n"
        "/diag — диагностика OpenAI\n"
        "/env — окружение (для админа)\n"
        "/set_project proj_xxx — сменить проект (для админа)\n"
    )
    await update.effective_message.reply_text(text)

async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    init_db()
    if await check_trial_and_maybe_stop(update):
        return
    chat_id = update.effective_chat.id
    clear_history(chat_id)
    await update.effective_message.reply_text("История очищена. Продолжим?")

async def myid(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    init_db()
    if await check_trial_and_maybe_stop(update):
        return
    user = update.effective_user
    if not user:
        return
    uid = user.id
    uname = user.username or ""
    txt = f"Ваш ID: {uid}"
    if uname:
        txt += f"\nUsername: @{uname}"
    await update.effective_message.reply_text(txt)

async def on_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    q = update.callback_query
    if not q:
        return
    try:
        await q.answer()
    except Exception:
        pass
    if q.data == "myid":
        user = q.from_user
        uid = user.id if user else None
        uname = (user.username if user and user.username else "")
        txt = f"Ваш ID: {uid}" if uid is not None else "Не удалось определить ваш ID."
        if uname:
            txt += f"\nUsername: @{uname}"
        try:
            await q.message.reply_text(txt)
        except Exception:
            pass

async def tex_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    init_db()
    if await check_trial_and_maybe_stop(update):
        return
    args = context.args or []
    if not args:
        await update.effective_message.reply_text("Использование: /tex <формула>\nПример: /tex \\int_0^1 x^2 dx")
        return
    tex = " ".join(args)
    try:
        png = render_tex_to_png(tex)
        await update.effective_message.reply_photo(InputFile(io.BytesIO(png), filename="tex.png"))
    except Exception:
        logger.exception("TeX render error")
        await update.effective_message.reply_text("Не удалось отрисовать формулу. Проверь синтаксис.")

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    init_db()
    if await check_trial_and_maybe_stop(update):
        return
    chat_id = update.effective_chat.id
    user_text = update.effective_message.text or ""
    save_message(chat_id, "user", user_text)
    messages = build_messages(chat_id, user_text)
    loop = asyncio.get_event_loop()
    reply = await loop.run_in_executor(None, call_openai, messages)
    save_message(chat_id, "assistant", reply)
    await update.effective_message.reply_text(reply, disable_web_page_preview=True)

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    init_db()
    if await check_trial_and_maybe_stop(update):
        return
    msg = update.message
    if not msg:
        return
    photo = (msg.photo[-1] if msg.photo else None)
    if not photo:
        return
    try:
        tg_file = await photo.get_file()
        img_bytes = await tg_file.download_as_bytearray()
        tg_file_id = photo.file_id
        content_type = "photo"
    except Exception:
        logger.exception("Failed to download photo")
        await msg.reply_text("Не удалось скачать фото. Пришли ещё раз, пожалуйста.")
        return
    try:
        await msg.reply_chat_action("typing")
    except Exception:
        pass
    mode = get_effective_ocr_mode(update.effective_chat.id)
    text = ocr_extract_bytes(bytes(img_bytes), mode)
    if not text:
        tips = (
            "Не удалось распознать текст. Попробуй так:\n"
            "• Снимай строго перпендикулярно, без бликов.\n"
            "• Заполни кадр задачей.\n"
            "• Пришли картинку как *файл* (без сжатия).\n\n"
            f"Сейчас OCR-режим: {mode}. /ocr_mode vision | tesseract\n"
            "Формулы можно ввести через /tex, например: /tex \\int_0^1 x^2 dx"
        )
        await msg.reply_text(tips, disable_web_page_preview=True)
        try:
            log_ocr_event(update.effective_chat.id, update.effective_user.id, mode, tg_file_id, content_type, preprocess_image_for_ocr(bytes(img_bytes)), "")
        except Exception:
            pass
        return
    chat_id = update.effective_chat.id
    user_text = f"[OCR] {text}"
    save_message(chat_id, "user", user_text)
    messages = build_messages(chat_id, user_text)
    loop = asyncio.get_event_loop()
    reply = await loop.run_in_executor(None, call_openai, messages)
    save_message(chat_id, "assistant", reply)
    try:
        log_ocr_event(chat_id, update.effective_user.id, mode, tg_file_id, content_type, preprocess_image_for_ocr(bytes(img_bytes)), text)
    except Exception:
        pass
    await msg.reply_text(reply, disable_web_page_preview=True)

async def handle_image_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    init_db()
    if await check_trial_and_maybe_stop(update):
        return
    msg = update.message
    if not msg or not msg.document:
        return
    doc = msg.document
    if not (doc.mime_type and doc.mime_type.startswith("image/")):
        return
    try:
        tg_file = await doc.get_file()
        img_bytes = await tg_file.download_as_bytearray()
        tg_file_id = doc.file_id
        content_type = doc.mime_type or "image"
    except Exception:
        logger.exception("Failed to download image document")
        await msg.reply_text("Не удалось скачать изображение. Пришли ещё раз, пожалуйста.")
        return
    try:
        await msg.reply_chat_action("typing")
    except Exception:
        pass
    mode = get_effective_ocr_mode(update.effective_chat.id)
    text = ocr_extract_bytes(bytes(img_bytes), mode)
    if not text:
        tips = (
            "Не удалось распознать текст. Попробуй так:\n"
            "• Сканируй без наклона и бликов.\n"
            "• Заполни кадр задачей.\n"
            "• Присылай как *файл* (без сжатия).\n\n"
            f"Сейчас OCR-режим: {mode}. /ocr_mode vision | tesseract\n"
            "Формулы: /tex \\int_0^1 x^2 dx"
        )
        await msg.reply_text(tips, disable_web_page_preview=True)
        try:
            log_ocr_event(update.effective_chat.id, update.effective_user.id, mode, tg_file_id, content_type, preprocess_image_for_ocr(bytes(img_bytes)), "")
        except Exception:
            pass
        return
    chat_id = update.effective_chat.id
    user_text = f"[OCR] {text}"
    save_message(chat_id, "user", user_text)
    messages = build_messages(chat_id, user_text)
    loop = asyncio.get_event_loop()
    reply = await loop.run_in_executor(None, call_openai, messages)
    save_message(chat_id, "assistant", reply)
    try:
        log_ocr_event(chat_id, update.effective_user.id, mode, tg_file_id, content_type, preprocess_image_for_ocr(bytes(img_bytes)), text)
    except Exception:
        pass
    await msg.reply_text(reply, disable_web_page_preview=True)

# Admin: /reset_trial
async def reset_trial(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    init_db()
    user = update.effective_user
    if user is None:
        return
    uid = user.id
    uname = (user.username or '').lower()
    if not (uname in ALLOWED_USERNAMES or uid in ALLOWED_USER_IDS):
        await update.effective_message.reply_text("Недостаточно прав для этой команды.")
        return
    if not context.args:
        await update.effective_message.reply_text("Использование: /reset_trial <user_id или @username>")
        return
    target = context.args[0]
    handle = target.lstrip("@")
    target_id = None
    try:
        target_id = int(handle)
    except ValueError:
        target_id = None
    if target_id is None:
        try:
            chat = await context.bot.get_chat(f"@{handle}")
            if chat and chat.id:
                target_id = chat.id
        except Exception:
            target_id = None
    if target_id is None:
        await update.effective_message.reply_text("Не удалось определить пользователя. Укажи числовой ID или корректный @username.")
        return
    reset_user_trial(target_id)
    await update.effective_message.reply_text(f"Тестовый период для пользователя {target} сброшен.")

# OCR mode per chat
async def ocr_mode_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    user = update.effective_user
    if not user:
        return
    current = get_effective_ocr_mode(chat_id)
    args = context.args or []
    if not args:
        await update.effective_message.reply_text(f"OCR-режим для этого чата: *{current}*\nДоступные: vision, tesseract", parse_mode=ParseMode.MARKDOWN_V2)
        return
    uid = user.id
    uname = (user.username or '').lower()
    if not (uname in ALLOWED_USERNAMES or uid in ALLOWED_USER_IDS):
        await update.effective_message.reply_text("Недостаточно прав. Текущий режим: " + current)
        return
    target = args[0].strip().lower()
    if target not in {"vision", "tesseract"}:
        await update.effective_message.reply_text("Неверное значение. Используй: /ocr_mode vision или /ocr_mode tesseract")
        return
    set_chat_ocr_mode(chat_id, target)
    await update.effective_message.reply_text(f"OCR-режим для этого чата установлен: *{target}*", parse_mode=ParseMode.MARKDOWN_V2)

# ---------- Build app & main ----------
def build_app() -> Application:
    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not set")
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    # Commands
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("reset", reset))
    app.add_handler(CommandHandler("welcome", welcome_cmd))
    app.add_handler(CommandHandler("myid", myid))
    app.add_handler(CommandHandler("tex", tex_cmd))
    app.add_handler(CommandHandler("ocr_mode", ocr_mode_cmd))
    app.add_handler(CommandHandler("reset_trial", reset_trial))

    # Diagnostics / Admin
    app.add_handler(CommandHandler("diag", diag))
    app.add_handler(CommandHandler("env", env_cmd))
    app.add_handler(CommandHandler("set_project", set_project))

    # Callbacks & messages
    app.add_handler(CallbackQueryHandler(on_callback))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.Document.IMAGE, handle_image_document))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    return app

def main() -> None:
    init_db()
    app = build_app()
    if USE_WEBHOOK:
        if not WEBHOOK_URL:
            raise RuntimeError("WEBHOOK_URL is required when USE_WEBHOOK=true")
        logger.info("Запуск в режиме webhook на порту %s ...", PORT)
        app.run_webhook(
            listen="0.0.0.0",
            port=PORT,
            url_path="",
            webhook_url=WEBHOOK_URL,
            allowed_updates=Update.ALL_TYPES,
        )
    else:
        logger.info("Запуск в режиме polling ...")
        app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
