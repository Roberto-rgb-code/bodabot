import os
import io
import re
import json
import wave
import base64
import unicodedata
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
from collections import defaultdict, Counter
from datetime import datetime
from fuzzywuzzy import fuzz
import requests
from fastapi import FastAPI, HTTPException, UploadFile, File, Query, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# =========================
# Cargar .env
# =========================
ROOT = Path(__file__).resolve().parent
load_dotenv(dotenv_path=ROOT / ".env")

# =========================
# Config desde .env
# =========================
DATA_PATH = Path(os.getenv("DATA_PATH", str(ROOT / "data.json")))
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")
if GOOGLE_APPLICATION_CREDENTIALS:
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GOOGLE_APPLICATION_CREDENTIALS

GOOGLE_API_KEY = (os.getenv("GOOGLE_API_KEY") or "").strip()
LANG_CODE = os.getenv("LANG_CODE", "es-MX")
TTS_VOICE = os.getenv("TTS_VOICE", "es-MX-Neural2-D")
TTS_VOICE_FALLBACK = os.getenv("TTS_VOICE_FALLBACK", "es-MX-Standard-C")
CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",")]

# =========================
# Google SDKs (STT/TTS)
# =========================
from google.cloud import speech_v1 as speech
from google.cloud import texttospeech

# =========================
# FastAPI
# =========================
app = FastAPI(title="BodaBot API (Invitados & Anfitrión)", version="4.5-invitados-anfitrion")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS if CORS_ORIGINS != ["*"] else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# Memoria en runtime
# =========================
TABLES: Dict[str, List[Dict[str, Any]]] = {}
ALIASES: Dict[str, str] = {}
INVERTED: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
SCHEMA: Dict[str, List[Dict[str, str]]] = {}
SESSIONS: Dict[str, Dict[str, Any]] = defaultdict(dict)
DEFAULT_SESSION = "default"

WORD_RE = re.compile(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ0-9@._+-]+")
DATE_RE = re.compile(
    r"(\d{1,2})\s*(?:de)?\s*(enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|octubre|noviembre|diciembre)\s*(?:de)?\s*(\d{4})?",
    flags=re.IGNORECASE
)
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
DIGITS_RE = re.compile(r"\d+")

# =========================
# Schema: SOLO tblInvitados
# =========================
SCHEMA_DEF = {
    "tblInvitados": [
        {"column_name": "idInvitado", "data_type": "int"},
        {"column_name": "apodo", "data_type": "varchar"},
        {"column_name": "nombre", "data_type": "varchar"},
        {"column_name": "aPaterno", "data_type": "varchar"},
        {"column_name": "aMaterno", "data_type": "varchar"},
        {"column_name": "boletos", "data_type": "int"},
        {"column_name": "idGenero", "data_type": "int"},
        {"column_name": "boletosConfirmados", "data_type": "int"},
        {"column_name": "correo", "data_type": "varchar"},
        {"column_name": "idTipoInvitado", "data_type": "int"},
        {"column_name": "vienePor", "data_type": "int"},
        {"column_name": "telefono", "data_type": "varchar"},
        {"column_name": "idTipoMenu", "data_type": "int"},
        {"column_name": "accedio", "data_type": "datetime"},
        {"column_name": "llegoWeb", "data_type": "datetime"},
        {"column_name": "contrasena", "data_type": "varchar"},
        {"column_name": "invitacionEnviada", "data_type": "bit"},
        {"column_name": "soloMisa", "data_type": "bit"},
        {"column_name": "asistira", "data_type": "bit"},
        {"column_name": "qrEnviado", "data_type": "bit"},
        {"column_name": "qrConfirmado", "data_type": "bit"},
        {"column_name": "asistioBoda", "data_type": "bit"},
        {"column_name": "mesa", "data_type": "int"},
        {"column_name": "idNovios", "data_type": "int"},
        {"column_name": "mensaje", "data_type": "text"},
        {"column_name": "regalo", "data_type": "bit"},
        {"column_name": "mensajeRegalo", "data_type": "text"},
        {"column_name": "idIdioma", "data_type": "int"},
    ],
}

# =========================
# Utilidades
# =========================
CONNECTORS = {"y","o","con","por","para","de","del","la","el","los","las","al","a","en"}

def normalize(s: Optional[str]) -> str:
    if s is None:
        return ""
    s = str(s).lower().strip()
    s = ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')
    s = s.replace("salon", "salón").replace("invitacion", "invitación")
    return s

def tokenize(val: Union[str, int, float, None]) -> List[str]:
    if val is None:
        return []
    return [normalize(t) for t in WORD_RE.findall(str(val))]

def parse_date(text: str) -> Optional[datetime]:
    months = {
        "enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5, "junio": 6,
        "julio": 7, "agosto": 8, "septiembre": 9, "octubre": 10, "noviembre": 11, "diciembre": 12
    }
    match = DATE_RE.search(text)
    if match:
        day = int(match.group(1))
        month = months[normalize(match.group(2))]
        year = int(match.group(3) or datetime.now().year)
        return datetime(year, month, day)
    return None

def _bit(v: Any) -> int:
    if v is None:
        return 0
    if isinstance(v, bool):
        return 1 if v else 0
    try:
        return 1 if int(v) == 1 else 0
    except Exception:
        return 0

def _int(v: Any) -> int:
    try:
        return int(v)
    except Exception:
        return 0

def _format_phone(val: str) -> str:
    if not val:
        return "—"
    nums = ''.join(DIGITS_RE.findall(val))
    if len(nums) == 10:
        return f"{nums[0:2]} {nums[2:6]} {nums[6:10]}"
    if len(nums) == 12:
        return f"{nums[0:2]} {nums[2:4]} {nums[4:8]} {nums[8:12]}"
    return val

def validate_row(row: Dict[str, Any], table_name: str) -> bool:
    if not row or all(v is None for v in row.values()):
        return False
    schema = SCHEMA.get(table_name, [])
    for col in schema:
        col_name = col["column_name"]
        if col_name in row and row[col_name] is not None:
            if col["data_type"] == "int" and not isinstance(row[col_name], (int, float)):
                return False
            if col["data_type"] == "decimal" and not isinstance(row[col_name], (int, float)):
                return False
            if col["data_type"] in ("varchar", "text", "nvarchar") and not isinstance(row[col_name], str):
                return False
            if col["data_type"] == "bit" and not isinstance(row[col_name], (bool, int)):
                return False
            if col["data_type"] in ("datetime", "date") and not isinstance(row[col_name], str):
                return False
    return True

def _register_table(canon: str, rows: List[Any]) -> None:
    cleaned_rows = []
    for row in rows:
        if isinstance(row, dict) and validate_row(row, canon):
            cleaned = {k: v for k, v in row.items() if v is not None}
            cleaned_rows.append(cleaned)
    if cleaned_rows:
        TABLES[canon] = cleaned_rows
        ALIASES[normalize(canon)] = canon
        short = canon.split(".")[-1]
        ALIASES[normalize(short)] = canon
        no_tbl = re.sub(r"^tbl", "", short, flags=re.IGNORECASE)
        if no_tbl:
            ALIASES[normalize(no_tbl)] = canon
        singular = re.sub(r"(es|s)$", "", no_tbl, flags=re.IGNORECASE)
        if singular and normalize(singular) not in ALIASES:
            ALIASES[normalize(singular)] = canon

def _build_inverted() -> None:
    INVERTED.clear()
    for tname, rows in TABLES.items():
        for i, row in enumerate(rows):
            tokens = set()
            for k, v in row.items():
                if k in ("idInvitado", "apodo", "nombre", "aPaterno", "aMaterno", "correo", "telefono", "mensaje", "mensajeRegalo"):
                    tokens.update(tokenize(v))
                else:
                    tokens.update(tokenize(v))
            for tok in tokens:
                INVERTED[tok].append((tname, i))

def _extract_tbl_invitados(obj: Any) -> List[Dict[str, Any]]:
    if isinstance(obj, dict):
        if "tblInvitados" in obj and isinstance(obj["tblInvitados"], list):
            return obj["tblInvitados"]
        for v in obj.values():
            res = _extract_tbl_invitados(v)
            if res:
                return res
    elif isinstance(obj, list):
        if all(isinstance(it, dict) and "idInvitado" in it for it in obj):
            return obj
        for it in obj:
            res = _extract_tbl_invitados(it)
            if res:
                return res
    return []

def load_data() -> None:
    if not DATA_PATH.exists():
        raise RuntimeError(f"No se encontró data.json en: {DATA_PATH}")
    with DATA_PATH.open("r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Error decodificando JSON: {e}")
    TABLES.clear()
    ALIASES.clear()
    SCHEMA.clear()
    SCHEMA["tblInvitados"] = SCHEMA_DEF["tblInvitados"]

    if isinstance(data, dict) and isinstance(data.get("tblInvitados"), list):
        invitados_data = data["tblInvitados"]
    else:
        invitados_data = _extract_tbl_invitados(data)

    _register_table("tblInvitados", invitados_data or [])
    _build_inverted()

# Carga inicial
load_data()

def resolve_table(name: str) -> str:
    key = normalize(name)
    if key in ALIASES:
        return ALIASES[key]
    key2 = normalize(name.strip(' "\'')) if name else ""
    if key2 in ALIASES:
        return ALIASES[key2]
    for alias, canon in ALIASES.items():
        if fuzz.ratio(key, alias) > 80:
            return canon
    raise HTTPException(status_code=404, detail=f"Tabla '{name}' no encontrada. Revisa /tables")

# =========================
# RAG enfocado a invitados
# =========================
def rank_preferred_tables() -> List[str]:
    return ["tblInvitados"] if "tblInvitados" in TABLES else []

PREFERRED = rank_preferred_tables()

def retrieve_context(question: str, max_items: int = 40) -> Dict[str, Any]:
    q_tokens = tokenize(question)
    ctx: Dict[str, Any] = {}
    rows = TABLES.get("tblInvitados", [])
    if not rows:
        return ctx
    if not q_tokens:
        ctx["tblInvitados"] = rows[:min(len(rows), max_items)]
        return ctx
    hits: List[Tuple[str, int]] = []
    for t in q_tokens:
        hits.extend(INVERTED.get(t, []))
    counts = Counter(hits)
    idxs = [idx for (_, idx), _ in counts.most_common()]
    if not idxs:
        ctx["tblInvitados"] = rows[:min(len(rows), max_items)]
        return ctx
    ctx["tblInvitados"] = [rows[i] for i in idxs[:max_items] if 0 <= i < len(rows)]
    return ctx

# =========================
# Google NLP opcional
# =========================
LANGUAGE_ENDPOINT = "https://language.googleapis.com/v2/documents:analyzeEntities"
CLASSIFY_ENDPOINT = "https://language.googleapis.com/v2/documents:classifyText"

def gnlp_analyze_entities(text: str, language: str = "es") -> Dict[str, Any]:
    if not GOOGLE_API_KEY:
        return {"entities": []}
    payload = {
        "document": {"type": "PLAIN_TEXT", "language": language, "content": text},
        "encodingType": "UTF8"
    }
    url = f"{LANGUAGE_ENDPOINT}?key={GOOGLE_API_KEY}"
    r = requests.post(url, json=payload, timeout=15)
    if r.status_code != 200:
        return {"entities": []}
    return r.json()

def gnlp_classify_text(text: str, language: str = "es") -> Dict[str, Any]:
    if not GOOGLE_API_KEY or len(text) < 20:
        return {"categories": []}
    payload = {"document": {"type": "PLAIN_TEXT", "language": language, "content": text}}
    url = f"{CLASSIFY_ENDPOINT}?key={GOOGLE_API_KEY}"
    r = requests.post(url, json=payload, timeout=15)
    if r.status_code != 200:
        return {"categories": []}
    return r.json()

# =========================
# NLU mejorado
# =========================
GREETING_WORDS = {"hola", "buenos dias", "buenas tardes", "buenas noches", "hey", "qué tal", "que tal"}
WHO_ARE_YOU = {"quien eres", "quién eres", "como te llamas", "cómo te llamas"}
HELP_WORDS = {"ayuda", "que puedes hacer", "qué puedes hacer", "ayudame", "ayúdame", "como funcionas", "cómo funcionas"}
FOLLOW_MORE = {"mas", "más", "siguiente", "otra", "otro", "muestrame mas", "muéstrame más"}
FOLLOW_DETAILS = {"detalles", "detalle", "por tipo", "por mesa"}

FIELD_SYNONYMS = {
    "telefono": ["tel", "teléfono", "telefono", "cel", "celular", "whatsapp", "whats"],
    "correo": ["correo", "mail", "email", "e-mail"],
    "mesa": ["(mesa de)", "(que mesa)", "(qué mesa)"],
    "boletos": ["boletos", "pases", "tickets"],
    "boletosConfirmados": ["boletos confirmados", "pases confirmados", "tickets confirmados"],
    "apodo": ["apodo"],
    "nombre": ["nombre"],
    "qrEnviado": ["qr enviado", "qr enviados", "codigo enviado", "código enviado", "codigos enviados", "códigos enviados"],
    "qrConfirmado": ["qr confirmado", "qr confirmados", "codigo confirmado", "código confirmado", "codigos confirmados", "códigos confirmados"],
    "asistira": ["asistira", "asistirá", "asisten", "asistiran", "asistirán", "confirmado", "confirmados"],
}

# stopwords para limpiar target_text (más robustas para evitar caer en fact_query)
STOPWORDS_FOR_TARGET = (
    r"telefono|tel[eé]fono|cel|celular|whatsapp|correo|email|mail|"
    r"mesa|boletos|boletos confirmados|"
    r"qr|qr enviado|qr enviados|qr confirmado|qr confirmados|"
    r"confirmad(?:o|a|os|as)|enviad(?:o|a|os|as)|asistir(?:a|á|an|án)?|asisten|asistencia|"
    r"de|del|de la|de los|de las|para|por|con|"
    r"busca|dame|muestra|quiero|quien es|quién es|quienes|quiénes|"
    r"datos|info|informaci[oó]n|ficha|contacto"
)

def _has_mesa_filter(t: str) -> bool:
    return bool(re.search(r"(?:^| )(?:invitados\s+)?(?:de\s+(?:la\s+)?)?mesa\s*#?\s*\d+\b", t)) or ("sin mesa" in t)

def _has_mesas_grouping(t: str) -> bool:
    return ("mesas" in t) or ("por mesa" in t)

def _has_confirmados(t: str) -> bool:
    return bool(re.search(r"\bconfirmad[oa]s?\b|\basistir(?:a|á|an|án)\b|\basiste[nrs]?\b", t))

def _has_qr_enviado(t: str) -> bool:
    return ("qr" in t and "enviad" in t)

def _has_qr_confirmado(t: str) -> bool:
    return ("qr" in t and "confirmad" in t)

def _meaningful_tokens(s: str) -> List[str]:
    toks = [w for w in tokenize(s) if len(w) >= 3 and w not in CONNECTORS]
    return toks

def _match_field(text: str) -> Optional[str]:
    t = normalize(text)
    # 'mesa' como campo solo si viene en frases de persona ("mesa de", "qué mesa")
    if re.search(r"\b(que|qué)\s+mesa\b|\bmesa\s+de\b", t):
        return "mesa"
    # otros campos
    for field, words in FIELD_SYNONYMS.items():
        if field == "mesa":
            continue
        for w in words:
            if w in t:
                return field
    return None

def _extract_person_query(text: str) -> Tuple[Optional[int], str]:
    t = normalize(text)
    m = re.search(r"\b(?:id|#)\s*(\d+)\b", t)
    if m:
        try:
            return int(m.group(1)), ""
        except Exception:
            pass
    q = re.sub(rf"\b({STOPWORDS_FOR_TARGET})\b", "", t)
    q = re.sub(r"[^\w@._+-]+", " ", q).strip()
    return None, q

def _looks_like_person_query(text: str) -> bool:
    tid, q = _extract_person_query(text)
    return bool(tid or EMAIL_RE.search(q) or len(_meaningful_tokens(q)) > 0)

def detect_intent_and_slots(text: str) -> Dict[str, Any]:
    t = normalize(text)
    slots = {"date": parse_date(t), "entidades": [], "categorias": [], "field": None, "target_id": None, "target_text": ""}

    if any(w in t for w in GREETING_WORDS):
        return {"intent": "greeting", "slots": slots}
    if any(w in t for w in WHO_ARE_YOU):
        return {"intent": "who_are_you", "slots": slots}
    if any(w in t for w in HELP_WORDS):
        return {"intent": "help", "slots": slots}
    if any(w in t for w in FOLLOW_MORE):
        return {"intent": "follow_more", "slots": slots}
    if any(w in t for w in FOLLOW_DETAILS):
        return {"intent": "detalles_invitados", "slots": slots}

    # Filtros claros antes de fact_query
    if _has_mesa_filter(t) or _has_mesas_grouping(t):
        return {"intent": "invitados", "slots": slots}

    # Listados genéricos: confirmados / QR / boletos confirmados (sin persona) -> invitados
    if (_has_confirmados(t) or _has_qr_enviado(t) or _has_qr_confirmado(t) or re.search(r"boletos\s+confirmad", t)):
        if not _looks_like_person_query(t):
            return {"intent": "invitados", "slots": slots}

    # "ficha de {persona}"
    if re.search(r"\bficha\s+de\b", t) or t.startswith("ficha "):
        tid, q = _extract_person_query(t)
        slots["target_id"], slots["target_text"] = tid, q
        return {"intent": "detail_guest", "slots": slots}

    # Conteo específico "¿cuántos boletos faltan?"
    if "bolet" in t and ("faltan" in t or "restan" in t):
        return {"intent": "count_boletos_faltan", "slots": slots}

    # Conteos rápidos genéricos
    if re.search(r"\bcu[aá]nt[oa]s?\b|\btotal\b", t):
        return {"intent": "count_query", "slots": slots}

    # Quién es X
    if "quien es" in t or "quién es" in t:
        tid, q = _extract_person_query(t)
        slots["target_id"], slots["target_text"] = tid, q
        return {"intent": "detail_guest", "slots": slots}

    # Campo de persona (tel/correo/mesa/boletos/qr) solo si parece consulta de persona
    fld = _match_field(t)
    if fld is not None and _looks_like_person_query(t):
        tid, q = _extract_person_query(t)
        slots["field"] = fld
        slots["target_id"], slots["target_text"] = tid, q
        return {"intent": "fact_query", "slots": slots}

    # Palabras clave de invitados/invitación
    if any(k in t for k in [
        "invitado", "invitados", "lista de invitados", "lista", "buscar", "busca",
        "mesa", "mesas", "sin mesa",
        "confirmado", "confirmados", "asistira", "asistirá", "asisten", "asistiran", "asistirán",
        "qr", "qr enviado", "qr confirmados", "boletos", "boletos confirmados",
        "correo", "email", "tel", "telefono", "teléfono", "gmail.com", "solo misa"
    ]):
        return {"intent": "invitados", "slots": slots}

    ents = gnlp_analyze_entities(text).get("entities", [])
    cats = gnlp_classify_text(text).get("categories", [])
    slots["entidades"] = ents
    slots["categorias"] = cats
    return {"intent": "invitados", "slots": slots}

# =========================
# Helpers invitados
# =========================
def _nombre_completo(g: Dict[str, Any]) -> str:
    return " ".join(
        [str(g.get("nombre") or "").strip(),
         str(g.get("aPaterno") or "").strip(),
         str(g.get("aMaterno") or "").strip()]
    ).strip() or "(sin nombre)"

def _agg_boletos(rows: List[Dict[str, Any]]) -> Tuple[int, int]:
    total = sum(_int(g.get("boletos")) for g in rows)
    confirm = sum(_int(g.get("boletosConfirmados")) for g in rows)
    return total, confirm

def _mesas_breakdown(rows: List[Dict[str, Any]]) -> Dict[int, int]:
    mesas: Dict[int, int] = defaultdict(int)
    for g in rows:
        mesas[_int(g.get("mesa"))] += 1
    return dict(sorted(mesas.items(), key=lambda x: x[0]))

def _suggest_next() -> str:
    return (
        "\n\n¿Seguimos? Puedes decirme por ejemplo:\n"
        "• “invitados de la mesa 12” · “sin mesa”\n"
        "• “teléfono de Carmen Chávez” · “correo de Farah”\n"
        "• “¿cuántos confirmados?” · “QR confirmados”\n"
        "• “busca a Enrique 333” o “gmail.com”"
    )

def _guest_tokens_match(g: Dict[str, Any], tokens: List[str]) -> bool:
    if not tokens:
        return False
    name = normalize(_nombre_completo(g))
    mail = normalize(g.get("correo") or "")
    phone_digits = "".join(DIGITS_RE.findall(g.get("telefono") or ""))
    for tok in tokens:
        if tok not in name and tok not in mail and tok not in phone_digits:
            return False
    return True

def _find_guest_by_id_or_text(rows: List[Dict[str, Any]], target_id: Optional[int], target_text: str) -> List[Dict[str, Any]]:
    if target_id is not None:
        return [g for g in rows if _int(g.get("idInvitado")) == target_id]
    t = normalize(target_text)
    if not t:
        return []
    tokens = _meaningful_tokens(t)
    # Primero: exigir que todos los tokens aparezcan
    strong = [g for g in rows if _guest_tokens_match(g, tokens)] if tokens else []
    if strong:
        strong.sort(key=lambda g: fuzz.token_set_ratio(t, normalize(_nombre_completo(g))), reverse=True)
        return strong
    # fallback: fuzzy general
    def score(g):
        s = 0
        s = max(s, fuzz.token_set_ratio(t, normalize(_nombre_completo(g))))
        s = max(s, fuzz.partial_ratio(t, normalize(g.get("correo") or "")))
        s = max(s, fuzz.partial_ratio(t, normalize(g.get("telefono") or "")))
        return s
    cand = [g for g in rows if score(g) >= 70]
    cand.sort(key=lambda g: score(g), reverse=True)
    return cand

# =========================
# Respuestas
# =========================
def compose_answer(intent: str, text: str, ctx: Dict[str, Any], session: Dict[str, Any]) -> str:
    if intent == "greeting":
        return "¡Hola! Soy BodaBot 💍. Te ayudo con tus invitados: lista, confirmados, mesas, QR, boletos… y búsquedas por nombre/correo/teléfono." + _suggest_next()
    if intent == "who_are_you":
        return "Soy BodaBot, tu asistente para la boda. Trabajo con **tblInvitados**: listar, buscar, contar y darte detalles con un tono humano 😉."
    if intent == "help":
        return ("Puedo ayudarte con invitados:\n"
                "• Lista general: “lista de invitados”\n"
                "• Confirmaciones: “quiénes asistirán”, “confirmados”\n"
                "• Boletos: “boletos”, “boletos confirmados”\n"
                "• Mesas: “invitados de la mesa 5”, “sin mesa”, “mesas (por mesa)”\n"
                "• QR: “qr enviados”, “qr confirmados”\n"
                "• Búsqueda: “ficha de Farah”, “correo gmail”, “tel 333”" + _suggest_next())

    rows = TABLES.get("tblInvitados", [])
    if not rows:
        return "No hay datos en **tblInvitados**. Asegúrate de que `data.json` incluya esa tabla o un array de invitados."

    t = normalize(text)

    # Paginación
    if intent == "follow_more":
        last = session.get("last", {})
        if last.get("type") == "invitados" and last.get("candidates"):
            shown = last.get("shown", 0)
            step = 10
            nxt = last["candidates"][shown:shown+step]
            if not nxt:
                return "No hay más resultados."
            session["last"]["shown"] = shown + len(nxt)
            session["last"]["display"] = nxt
            return render_invitados_list(nxt, prefix="Más invitados:\n")
        return "No hay una lista previa para continuar."

    # Detalle: “quién es …” o “ficha de …”
    if intent == "detail_guest":
        from_slots = session.get("last_slots") or {}
        target_id = from_slots.get("target_id")
        target_text = from_slots.get("target_text", "")
        if not target_id and not target_text:
            tid2, q2 = _extract_person_query(t)
            target_id, target_text = tid2, q2
        cand = _find_guest_by_id_or_text(rows, target_id, target_text)
        if not cand:
            return "No encontré a esa persona. ¿Me das nombre, correo, teléfono o el ID? 😊"
        if len(cand) > 1:
            top = cand[:5]
            return render_invitados_list(top, prefix="Encontré varios, ¿cuál de estos es?\n")
        g = cand[0]
        return render_guest_card(g)

    # Campo de persona: teléfono/correo/mesa/boletos/qr…
    if intent == "fact_query":
        field = session.get("last_slots", {}).get("field")
        target_id = session.get("last_slots", {}).get("target_id")
        target_text = session.get("last_slots", {}).get("target_text", "")
        if not field:
            field = _match_field(t)
        if not (target_id or target_text):
            tid2, q2 = _extract_person_query(t)
            target_id, target_text = tid2, q2

        cand = _find_guest_by_id_or_text(rows, target_id, target_text)
        if not cand:
            return "No pude ubicar a esa persona. ¿Me repites nombre, correo o ID? 🙏"
        if len(cand) > 1:
            sample = render_invitados_list(cand[:5], prefix="Encontré varios, ¿cuál de estos es?\n")
            return sample

        g = cand[0]
        nombre = _nombre_completo(g)
        if field == "telefono":
            return f"📞 Teléfono de {nombre}: { _format_phone(g.get('telefono') or '—') }"
        if field == "correo":
            return f"📧 Correo de {nombre}: { g.get('correo') or '—' }"
        if field == "mesa":
            mesa = _int(g.get("mesa"))
            mesa_txt = mesa if mesa > 0 else "—"
            return f"🍽️ Mesa de {nombre}: {mesa_txt}"
        if field == "boletos":
            return f"🎟️ Boletos asignados a {nombre}: { _int(g.get('boletos')) }"
        if field == "boletosConfirmados":
            return f"✅ Boletos confirmados de {nombre}: { _int(g.get('boletosConfirmados')) }"
        if field == "qrEnviado":
            val = "sí" if _bit(g.get("qrEnviado")) == 1 else "no"
            return f"✉️ ¿QR enviado para {nombre}?: {val}"
        if field == "qrConfirmado":
            val = "sí" if _bit(g.get("qrConfirmado")) == 1 else "no"
            return f"✔️ ¿QR confirmado para {nombre}?: {val}"
        if field == "apodo":
            return f"🧾 Apodo de {nombre}: { g.get('apodo') or '—' }"
        if field == "nombre":
            return f"👤 Nombre completo: { nombre }"
        return "Puedo darte: teléfono, correo, mesa, boletos, boletos confirmados, QR enviado/confirmado, apodo o nombre." + _suggest_next()

    # Conteo general
    if intent == "count_query":
        total_boletos, total_boletos_conf = _agg_boletos(rows)
        counts = {
            "total": len(rows),
            "asistira": sum(1 for g in rows if _bit(g.get("asistira")) == 1),
            "no_confirmados": sum(1 for g in rows if _bit(g.get("asistira")) == 0),
            "sin_mesa": sum(1 for g in rows if _int(g.get("mesa")) == 0),
            "qr_enviado": sum(1 for g in rows if _bit(g.get("qrEnviado")) == 1),
            "qr_confirmado": sum(1 for g in rows if _bit(g.get("qrConfirmado")) == 1),
            "boletos": total_boletos,
            "boletos_confirmados": total_boletos_conf,
        }
        return (f"Resumen rápido:\n"
                f"• Invitados: {counts['total']}\n"
                f"• Confirmados (asistirá): {counts['asistira']}\n"
                f"• Sin confirmar: {counts['no_confirmados']}\n"
                f"• Sin mesa: {counts['sin_mesa']}\n"
                f"• QR enviados: {counts['qr_enviado']} · QR confirmados: {counts['qr_confirmado']}\n"
                f"• Boletos: {counts['boletos']} / Confirmados: {counts['boletos_confirmados']}"
                + _suggest_next())

    # Conteo específico
    if intent == "count_boletos_faltan":
        total_boletos, total_boletos_conf = _agg_boletos(rows)
        faltan = max(0, total_boletos - total_boletos_conf)
        return f"🎟️ Faltan **{faltan}** boletos por confirmar (Totales: {total_boletos} · Confirmados: {total_boletos_conf})."

    # Vista / filtros (incluye combinados)
    if intent in ("detalles_invitados", "invitados"):
        filtered = rows

        # Mesa específica (acepta “mesa 12”, “de mesa 12”, “de la mesa 12”)
        m = re.search(r"(?:de\s+(?:la\s+)?)?mesa\s*#?\s*(\d+)", t)
        mesa_val = int(m.group(1)) if m else None
        if mesa_val is not None:
            filtered = [g for g in filtered if _int(g.get("mesa")) == mesa_val]

        # Sin mesa
        if "sin mesa" in t:
            filtered = [g for g in filtered if _int(g.get("mesa")) == 0]

        # Confirmados (asistirá)
        if _has_confirmados(t):
            filtered = [g for g in filtered if _bit(g.get("asistira")) == 1]

        # Solo misa
        if re.search(r"\bsolo\s+misa\b", t):
            filtered = [g for g in filtered if _bit(g.get("soloMisa")) == 1]

        # QR
        if _has_qr_enviado(t):
            filtered = [g for g in filtered if _bit(g.get("qrEnviado")) == 1]
        if _has_qr_confirmado(t):
            filtered = [g for g in filtered if _bit(g.get("qrConfirmado")) == 1]

        # Boletos confirmados (lista de quienes tienen > 0 confirmados)
        if re.search(r"\bboletos\s+confirmad", t):
            filtered = [g for g in filtered if _int(g.get("boletosConfirmados")) > 0]

        # Prefijo de teléfono
        m_tel = re.search(r"\b(?:tel(?:efono)?|cel|celular|whats(?:app)?)\s*([0-9]{2,})", t)
        if m_tel:
            phone_prefix = m_tel.group(1)
            def phone_has_prefix(g):
                digits = "".join(DIGITS_RE.findall(g.get("telefono") or ""))
                return phone_prefix in digits
            filtered = [g for g in filtered if phone_has_prefix(g)]

        # Búsqueda libre por nombre/correo/teléfono
        wants_boletos = "boletos" in t
        name_like_raw = re.sub(
            r"\b(invitados?|lista|buscar|busca|de|la|el|los|las|por|mesa|mesas|confirmad[oa]s?|asistir[aá](?:n)?|asisten|qr|boletos|sin mesa|email|correo|tel[eé]fono|tel|cel|celular|whats(?:app)?)\b",
            "", t
        )
        name_like_raw = re.sub(r"[^\w@._+-]+", " ", name_like_raw).strip()
        tokens = _meaningful_tokens(name_like_raw)
        if (EMAIL_RE.search(name_like_raw) or len(tokens) > 0) and not m_tel:
            def score(g):
                s = 0
                s = max(s, fuzz.token_set_ratio(name_like_raw, normalize(_nombre_completo(g))))
                s = max(s, fuzz.partial_ratio(name_like_raw, normalize(g.get("correo") or "")))
                s = max(s, fuzz.partial_ratio(name_like_raw, normalize(g.get("telefono") or "")))
                return s
            filtered = [g for g in filtered if score(g) >= 60]
            filtered.sort(key=lambda g: fuzz.token_set_ratio(name_like_raw, normalize(_nombre_completo(g))), reverse=True)

        # Resumen
        total_boletos, total_boletos_conf = _agg_boletos(rows)
        header = (f"Invitados (total: {len(rows)}) · "
                  f"Asistirá: {sum(1 for g in rows if _bit(g.get('asistira')) == 1)} · "
                  f"Solo misa: {sum(1 for g in rows if _bit(g.get('soloMisa')) == 1)} · "
                  f"QR enviado: {sum(1 for g in rows if _bit(g.get('qrEnviado')) == 1)} · "
                  f"QR confirmado: {sum(1 for g in rows if _bit(g.get('qrConfirmado')) == 1)} · "
                  f"Boletos: {total_boletos} / Confirmados: {total_boletos_conf}")

        extra = ""
        if wants_boletos:
            faltan = max(0, total_boletos - total_boletos_conf)
            extra += f"\nBoletos: Totales={total_boletos} · Confirmados={total_boletos_conf} · Faltan={faltan}"

        # Desglose por mesa
        if _has_mesas_grouping(t):
            mesas = _mesas_breakdown(filtered)
            top = [f"Mesa {k or 0}: {v} invitado(s)" for k, v in list(mesas.items())[:50]]
            extra += ("\n\nPor mesa:\n" + ("\n".join(top) if top else "—"))

        page = filtered[:10]
        session["last"] = {"type": "invitados", "candidates": filtered, "shown": len(page), "display": page}
        if not filtered:
            return header + extra + "\n\nNo encontré invitados con ese criterio." + _suggest_next()
        return header + extra + "\n\n" + render_invitados_list(page)

    ctx_preview = len(ctx.get("tblInvitados", []))
    return (f"Listo para ayudarte con **tblInvitados** (contexto: {ctx_preview} registros).\n"
            f"Ejemplos: “lista de invitados”, “confirmados”, “boletos”, “invitados de la mesa 4”, “sin mesa”, “qr confirmados”, “ficha de Farah”."
            + _suggest_next())

def render_guest_card(g: Dict[str, Any]) -> str:
    nombre = _nombre_completo(g)
    mesa = _int(g.get("mesa"))
    mesa_txt = mesa if mesa > 0 else "—"
    asistira = "✅" if _bit(g.get("asistira")) == 1 else "—"
    solo_misa = "⛪" if _bit(g.get("soloMisa")) == 1 else "—"
    qr = "✉" if _bit(g.get("qrEnviado")) == 1 else "—"
    qr_ok = "✔" if _bit(g.get("qrConfirmado")) == 1 else "—"
    correo = g.get("correo") or "—"
    tel = _format_phone(g.get("telefono") or "")
    return (f"👤 {nombre} (ID { _int(g.get('idInvitado')) })\n"
            f"• Mesa: {mesa_txt}\n"
            f"• Asistirá: {asistira} · Solo misa: {solo_misa}\n"
            f"• QR: {qr}/{qr_ok}\n"
            f"• 📧 {correo} · 📞 {tel}" + _suggest_next())

def invitado_summary(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    total_boletos, total_boletos_conf = _agg_boletos(rows)
    return {
        "total": len(rows),
        "asistira": sum(1 for g in rows if _bit(g.get("asistira")) == 1),
        "solo_misa": sum(1 for g in rows if _bit(g.get("soloMisa")) == 1),
        "qr_enviado": sum(1 for g in rows if _bit(g.get("qrEnviado")) == 1),
        "qr_confirmado": sum(1 for g in rows if _bit(g.get("qrConfirmado")) == 1),
        "boletos": total_boletos,
        "boletos_confirmados": total_boletos_conf,
    }

def render_invitados_list(items: List[Dict[str, Any]], prefix: str = "Invitados:\n") -> str:
    lines = []
    for i, g in enumerate(items, start=1):
        nombre = _nombre_completo(g)
        mesa = _int(g.get("mesa"))
        mesa_txt = mesa if mesa > 0 else "—"
        correo = g.get("correo") or "—"
        tel = _format_phone(g.get("telefono") or "")
        asistira = "✅" if _bit(g.get("asistira")) == 1 else "—"
        solo_misa = "⛪" if _bit(g.get("soloMisa")) == 1 else "—"
        qr = "✉" if _bit(g.get("qrEnviado")) == 1 else "—"
        qr_ok = "✔" if _bit(g.get("qrConfirmado")) == 1 else "—"
        lines.append(
            f"{i}. {nombre} (ID { _int(g.get('idInvitado')) }) · Mesa: {mesa_txt} · "
            f"Asistirá: {asistira} · Solo misa: {solo_misa} · QR: {qr}/{qr_ok} · 📧 {correo} · 📞 {tel}"
        )
    if len(items) >= 10:
        lines.append("\nDi “más” para ver más resultados.")
    return prefix + "\n".join(lines)

# =========================
# STT/TTS
# =========================
def speech_client() -> speech.SpeechClient:
    return speech.SpeechClient()

def tts_client() -> texttospeech.TextToSpeechClient:
    return texttospeech.TextToSpeechClient()

def stt_webm_opus(content: bytes, language_code: str = LANG_CODE) -> str:
    cli = speech_client()
    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.WEBM_OPUS,
        language_code=language_code,
        enable_automatic_punctuation=True,
    )
    resp = cli.recognize(config=config, audio=audio)
    if not resp.results:
        return ""
    return resp.results[0].alternatives[0].transcript

def stt_wav_linear16(content: bytes, language_code: str = LANG_CODE) -> str:
    cli = speech_client()
    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code=language_code,
        enable_automatic_punctuation=True,
    )
    resp = cli.recognize(config=config, audio=audio)
    if not resp.results:
        return ""
    return resp.results[0].alternatives[0].transcript

def pcm16le_to_wav_bytes(pcm: bytes, sample_rate: int = 16000, channels: int = 1) -> bytes:
    buff = io.BytesIO()
    with wave.open(buff, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    return buff.getvalue()

def tts_mp3(text: str, language_code: str = LANG_CODE, voice_name: Optional[str] = None) -> bytes:
    cli = tts_client()
    voice_name = voice_name or TTS_VOICE_FALLBACK
    synthesis_in = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(
        language_code=language_code,
        name=voice_name,
        ssml_gender=texttospeech.SsmlVoiceGender.FEMALE
    )
    cfg = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)
    out = cli.synthesize_speech(input=synthesis_in, voice=voice, audio_config=cfg)
    return out.audio_content

def tts_wav_linear16(text: str, language_code: str = LANG_CODE, voice_name: Optional[str] = None) -> bytes:
    cli = tts_client()
    voice_name = voice_name or TTS_VOICE
    synthesis_in = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(
        language_code=language_code,
        name=voice_name,
        ssml_gender=texttospeech.SsmlVoiceGender.FEMALE
    )
    cfg = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000
    )
    out = cli.synthesize_speech(input=synthesis_in, voice=voice, audio_config=cfg)
    return pcm16le_to_wav_bytes(out.audio_content, sample_rate=16000, channels=1)

# =========================
# Schemas
# =========================
class AskRequest(BaseModel):
    question: str
    max_ctx_items: Optional[int] = 40
    session_id: Optional[str] = None
    temperature: Optional[float] = 0.0  # compatible con el front

class AskResponse(BaseModel):
    model: str
    answer: str
    question: str
    used_sections: List[str]
    intent: str
    session_id: str

# =========================
# Endpoints
# =========================
@app.get("/")
def root():
    return {
        "name": "BodaBot API (Invitados & Anfitrión)",
        "version": "4.5-invitados-anfitrion",
        "endpoints": [
            "/health", "/schema", "/tables", "/table/{name}", "/search",
            "/invitados/summary", "/invitados/find", "/invitados/{idInvitado}",
            "/ask", "/ask_audio", "/ask_audio_wav", "/refresh"
        ],
    }

@app.get("/health")
def health():
    return {
        "ok": True,
        "data_path": str(DATA_PATH),
        "has_data": DATA_PATH.exists(),
        "tables_count": len(TABLES),
        "sample_tables": list(TABLES.keys())[:10],
        "google_nlp_key_loaded": bool(GOOGLE_API_KEY),
        "gcp_key_in_use": bool(os.getenv("GOOGLE_APPLICATION_CREDENTIALS")),
        "lang_code": LANG_CODE,
        "tts_voice": TTS_VOICE,
        "cors_origins": CORS_ORIGINS,
    }

@app.get("/schema")
def schema():
    return {"schema": SCHEMA_DEF["tblInvitados"]}

@app.post("/refresh")
def refresh():
    load_data()
    global PREFERRED
    PREFERRED = rank_preferred_tables()
    SESSIONS.clear()
    return {"reloaded": True, "tables": list(TABLES.keys())}

@app.get("/tables")
def tables():
    return {"tables": list(TABLES.keys())}

@app.get("/table/{name}")
def get_table(
    name: str,
    limit: int = Query(100, ge=1, le=2000),
    offset: int = Query(0, ge=0),
):
    canon = resolve_table(name)
    rows = TABLES.get(canon, [])
    return {"table": canon, "total": len(rows), "items": rows[offset: offset + limit]}

@app.get("/search")
def search(
    q: str = Query(..., min_length=1),
    table: Optional[str] = Query(None),
    max_hits: int = Query(50, ge=1, le=1000),
):
    toks = tokenize(q)
    seen = set()
    results = []
    allowed = resolve_table(table) if table else None

    for t in toks:
        for tname, idx in INVERTED.get(t, []):
            if allowed and tname != allowed:
                continue
            if (tname, idx) in seen:
                continue
            seen.add((tname, idx))
            rows = TABLES.get(tname, [])
            if 0 <= idx < len(rows):
                results.append({"table": tname, "index": idx, "item": rows[idx]})
            if len(results) >= max_hits:
                break
    return {"query": q, "count": len(results), "results": results}

@app.get("/invitados/summary")
def invitados_summary():
    rows = TABLES.get("tblInvitados", [])
    return invitado_summary(rows)

@app.get("/invitados/{idInvitado}")
def invitados_get(idInvitado: int):
    rows = TABLES.get("tblInvitados", [])
    for g in rows:
        if _int(g.get("idInvitado")) == idInvitado:
            return g
    raise HTTPException(status_code=404, detail="Invitado no encontrado")

@app.get("/invitados/find")
def invitados_find(
    q: str = Query(..., min_length=1),
    mesa: Optional[int] = Query(None),
    confirmados: Optional[bool] = Query(None),
    qr: Optional[str] = Query(None, description="enviado|confirmado"),
    limit: int = Query(20, ge=1, le=200)
):
    rows = TABLES.get("tblInvitados", [])
    t = normalize(q)
    def score(g):
        s = 0
        s = max(s, fuzz.partial_ratio(t, normalize(_nombre_completo(g))))
        s = max(s, fuzz.partial_ratio(t, normalize(g.get("correo") or "")))
        s = max(s, fuzz.partial_ratio(t, normalize(g.get("telefono") or "")))
        return s
    filt = [g for g in rows if score(g) >= 60]
    if mesa is not None:
        filt = [g for g in filt if _int(g.get("mesa")) == mesa]
    if confirmados is not None:
        filt = [g for g in filt if _bit(g.get("asistira")) == (1 if confirmados else 0)]
    if qr:
        if qr == "enviado":
            filt = [g for g in filt if _bit(g.get("qrEnviado")) == 1]
        if qr == "confirmado":
            filt = [g for g in filt if _bit(g.get("qrConfirmado")) == 1]
    return {"count": len(filt[:limit]), "items": filt[:limit]}

def _get_session(session_id: Optional[str], session_header: Optional[str]) -> Tuple[str, Dict[str, Any]]:
    sid = session_id or session_header or DEFAULT_SESSION
    return sid, SESSIONS[sid]

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest, x_session_id: Optional[str] = Header(None)):
    sid, session = _get_session(req.session_id, x_session_id)
    nlu = detect_intent_and_slots(req.question)
    session["last_slots"] = nlu.get("slots", {})
    ctx = retrieve_context(req.question, req.max_ctx_items or 40)
    answer = compose_answer(nlu["intent"], req.question, ctx, session)
    session["last_intent"] = nlu["intent"]
    return AskResponse(
        model="google-nlp",
        answer=answer,
        question=req.question,
        used_sections=list(ctx.keys()),
        intent=nlu["intent"],
        session_id=sid,
    )

@app.post("/ask_audio")
async def ask_audio(audio: UploadFile = File(...), language: str = LANG_CODE, x_session_id: Optional[str] = Header(None)):
    sid, session = _get_session(None, x_session_id)
    raw = await audio.read()
    user_text = stt_webm_opus(raw, language_code=language)
    if not user_text.strip():
        raise HTTPException(status_code=400, detail="No se pudo transcribir el audio.")
    nlu = detect_intent_and_slots(user_text)
    session["last_slots"] = nlu.get("slots", {})
    ctx = retrieve_context(user_text, 40)
    answer = compose_answer(nlu["intent"], user_text, ctx, session)
    session["last_intent"] = nlu["intent"]
    mp3 = tts_mp3(answer, language_code=language)
    return {
        "texto_usuario": user_text,
        "respuesta_texto": answer,
        "audio_base64": base64.b64encode(mp3).decode("utf-8"),
        "mime": "audio/mpeg",
        "used_sections": list(ctx.keys())["tblInvitados"] if ctx else [],
        "intent": nlu["intent"],
        "session_id": sid,
    }

@app.post("/ask_audio_wav")
async def ask_audio_wav(audio: UploadFile = File(...), language: str = LANG_CODE, x_session_id: Optional[str] = Header(None)):
    sid, session = _get_session(None, x_session_id)
    try:
        raw = await audio.read()
        try:
            with wave.open(io.BytesIO(raw), 'rb') as wf:
                nchannels, sampwidth, framerate = wf.getnchannels(), wf.getsampwidth(), wf.getframerate()
                if nchannels != 1 or framerate != 16000 or sampwidth != 2:
                    raise HTTPException(status_code=400, detail="El audio WAV debe ser mono, 16 kHz, 16-bit.")
        except wave.Error as e:
            raise HTTPException(status_code=400, detail=f"Error al procesar el archivo WAV: {str(e)}")

        user_text = stt_wav_linear16(raw, language_code=language)
        if not user_text.strip():
            raise HTTPException(status_code=400, detail="No se pudo transcribir el audio (WAV).")

        nlu = detect_intent_and_slots(user_text)
        session["last_slots"] = nlu.get("slots", {})
        ctx = retrieve_context(user_text, max_items=40)
        answer = compose_answer(nlu["intent"], user_text, ctx, session)
        session["last_intent"] = nlu["intent"]

        wav_bytes = tts_wav_linear16(answer, language_code=language)

        return {
            "texto_usuario": user_text,
            "respuesta_texto": answer,
            "audio_wav_base64": base64.b64encode(wav_bytes).decode("utf-8"),
            "mime": "audio/wav",
            "used_sections": list(ctx.keys())["tblInvitados"] if ctx else [],
            "intent": nlu["intent"],
            "session_id": sid,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error procesando la solicitud de audio: {str(e)}")
