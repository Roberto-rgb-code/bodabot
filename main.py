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
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse



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

# CONFIGURACIÓN DIRECTA - VOCES FEMENINAS MODERNAS
GOOGLE_API_KEY = "AIzaSyAKWV2JvFvIwfKZRKYxLkuahY2aD2UJcUQ"
LANG_CODE = "es-ES"
TTS_VOICE = "es-ES-Neural2-H"
TTS_VOICE_FALLBACK = "es-ES-Neural2-H"
CORS_ORIGINS = ["https://bodabot-9st7.onrender.com", "http://127.0.0.1:8000"]
MESA_TIPS_PATH = None

# =========================
# Google SDKs (STT/TTS)
# =========================
from google.cloud import speech_v1 as speech
from google.cloud import texttospeech

# =========================
# FastAPI
# =========================
app = FastAPI(title="BodaBot API (Invitados & Anfitrión)", version="5.0-sin-emojis")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS if CORS_ORIGINS != ["*"] else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Servir archivos estáticos y el index.html
app.mount("/static", StaticFiles(directory="."), name="static")

@app.get("/", include_in_schema=False)
def read_index():
    return FileResponse('index.html')

# =========================
# Memoria en runtime
# =========================
TABLES: Dict[str, List[Dict[str, Any]]] = {}
ALIASES: Dict[str, str] = {}
INVERTED: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
SCHEMA: Dict[str, List[Dict[str, str]]] = {}
SESSIONS: Dict[str, Dict[str, Any]] = defaultdict(dict)
DEFAULT_SESSION = "default"
MESA_TIPS: Dict[int, str] = {}

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
        return "N/D"
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

def _load_mesa_tips() -> None:
    MESA_TIPS.clear()
    if MESA_TIPS_PATH and MESA_TIPS_PATH.exists():
        try:
            with MESA_TIPS_PATH.open("r", encoding="utf-8") as f:
                data = json.load(f)
            for k, v in data.items():
                try:
                    MESA_TIPS[int(k)] = str(v)
                except Exception:
                    continue
        except Exception:
            pass

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
    _load_mesa_tips()

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
# NLU + scopes + control de audio
# =========================
GREETING_WORDS = {"hola", "buenos dias", "buenas tardes", "buenas noches", "hey", "qué tal", "que tal"}
WHO_ARE_YOU = {"quien eres", "quién eres", "como te llamas", "cómo te llamas"}
HELP_WORDS = {"ayuda", "que puedes hacer", "qué puedes hacer", "ayudame", "ayúdame", "como funcionas", "cómo funcionas"}
FOLLOW_MORE = {"mas", "más", "siguiente", "otra", "otro", "muestrame mas", "muéstrame más"}
FOLLOW_DETAILS = {"detalles", "detalle", "por tipo", "por mesa"}

# Control de audio
CANCEL_WORDS = {"cancelar", "cancela", "cancélalo", "ya no", "alto", "detente", "para", "stop"}
MUTE_ON_WORDS = {"silencio", "mute", "cállate", "callate", "sin voz", "sin sonido", "no hables", "apaga voz"}
MUTE_OFF_WORDS = {"habla", "con voz", "activa voz", "quitar silencio", "quita silencio", "desmute", "enciende voz"}

FIELD_SYNONYMS = {
    "telefono": ["tel", "teléfono", "telefono", "cel", "celular", "whatsapp", "whats"],
    "correo": ["correo", "mail", "email", "e-mail"],
    "mesa": ["(mesa de)", "(que mesa)", "(qué mesa)", "mi mesa"],
    "boletos": ["boletos", "pases", "tickets", "mis boletos", "mis pases"],
    "boletosConfirmados": ["boletos confirmados", "pases confirmados", "tickets confirmados"],
    "apodo": ["apodo"],
    "nombre": ["nombre"],
    "qrEnviado": ["qr enviado", "qr enviados", "codigo enviado", "código enviado"],
    "qrConfirmado": ["qr confirmado", "qr confirmados", "codigo confirmado", "código confirmado", "mi qr"],
    "asistira": ["asistira", "asistirá", "asisten", "asistiran", "asistirán", "confirmado", "confirmados"],
}

STOPWORDS_FOR_TARGET = (
    r"telefono|tel[eé]fono|cel|celular|whatsapp|correo|email|mail|"
    r"mesa|boletos|boletos confirmados|"
    r"qr|qr enviado|qr enviados|qr confirmado|qr confirmados|mi qr|"
    r"confirmad(?:o|a|os|as)|enviad(?:o|a|os|as)|asistir(?:a|á|an|án)?|asisten|asistencia|"
    r"de|del|de la|de los|de las|para|por|con|"
    r"busca|dame|muestra|quiero|quien es|quién es|quienes|quiénes|"
    r"datos|info|informaci[oó]n|ficha|contacto|mi|soy|somos|me llamo|mi nombre es|mesa de"
)

SCOPES = ("mesas","invitados","boletos","confirmados","qr","solo_misa","contacto","ficha","general")
ADMIN_KEYWORDS = {"lista","invitados","confirmados","boletos","qr","mesa","mesas","sin mesa","correo","telefono","teléfono","email"}

def determine_scope(t_text: str) -> str:
    t = normalize(t_text)
    if re.search(r"\b(ficha\s+de|^ficha\b|quien es|quién es)\b", t):
        return "ficha"
    if any(w in t for w in ["tel", "teléfono", "telefono", "correo", "email", "mail"]):
        return "contacto"
    if "solo misa" in t:
        return "solo_misa"
    if "bolet" in t:
        return "boletos"
    if "qr" in t or "código" in t or "codigo" in t:
        return "qr"
    if any(w in t for w in ["confirmado", "confirmados", "asistira", "asistirá", "asisten", "asistiran", "asistirán"]):
        return "confirmados"
    if "mesa" in t or "mesas" in t or "sin mesa" in t:
        return "mesas"
    if "invitado" in t or "lista" in t:
        return "invitados"
    return "general"

def _meaningful_tokens(s: str) -> List[str]:
    toks = [w for w in tokenize(s) if len(w) >= 3 and w not in CONNECTORS]
    return toks

def _match_field(text: str) -> Optional[str]:
    t = normalize(text)
    if re.search(r"\bmi\s+mesa\b", t):
        return "mesa"
    if re.search(r"\bmi\s+qr\b", t):
        return "qrConfirmado"
    if re.search(r"\bmis\s+(boletos|pases)\b", t):
        return "boletosConfirmados"
    if re.search(r"\b(que|qué)\s+mesa\b|\bmesa\s+de\b", t):
        return "mesa"
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

def _looks_like_guest_intro(t: str) -> Optional[str]:
    m = re.search(r"\b(?:soy|me\s+llamo|mi\s+nombre\s+es|somos)\s+(.+)$", t)
    if m:
        return m.group(1).strip()
    m2 = re.search(r"\bmesa\s+de\s+(.+)$", t)
    if m2:
        return m2.group(1).strip()
    tokens = _meaningful_tokens(t)
    if 1 < len(tokens) <= 4 and not any(k in t for k in ADMIN_KEYWORDS):
        return t.strip()
    return None

def _is_cancel(text: str) -> bool:
    t = normalize(text)
    return any(w in t for w in CANCEL_WORDS)

def _parse_pick_index(text: str) -> Optional[int]:
    t = normalize(text)
    m = re.search(r"(?:^|\b)(?:soy\s+(?:el|la)\s+|opci[oó]n\s+|#|num(?:ero)?\s*)(\d+)\b", t)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    return None

def detect_intent_and_slots(text: str) -> Dict[str, Any]:
    t = normalize(text)
    slots = {
        "date": parse_date(t),
        "entidades": [],
        "categorias": [],
        "field": None,
        "target_id": None,
        "target_text": "",
        "scope": determine_scope(t)
    }

    if any(w in t for w in GREETING_WORDS):
        return {"intent": "greeting", "slots": slots}
    if any(w in t for w in WHO_ARE_YOU):
        return {"intent": "who_are_you", "slots": slots}
    if any(w in t for w in HELP_WORDS):
        return {"intent": "help", "slots": slots}

    if _is_cancel(t):
        return {"intent": "cancel", "slots": slots}
    if any(w in t for w in MUTE_ON_WORDS):
        return {"intent": "mute_on", "slots": slots}
    if any(w in t for w in MUTE_OFF_WORDS):
        return {"intent": "mute_off", "slots": slots}

    pick = _parse_pick_index(t)
    if pick is not None:
        slots["pick_index"] = pick
        return {"intent": "pick_candidate", "slots": slots}

    if any(w in t for w in FOLLOW_MORE):
        return {"intent": "follow_more", "slots": slots}
    if any(w in t for w in FOLLOW_DETAILS):
        return {"intent": "detalles_invitados", "slots": slots}

    intro = _looks_like_guest_intro(t)
    if intro:
        slots["target_text"] = intro
        return {"intent": "guest_quick", "slots": slots}

    if re.search(r"(?:^| )(?:invitados\s+)?(?:de\s+(?:la\s+)?)?mesa\s*#?\s*\d+\b", t) or ("sin mesa" in t):
        return {"intent": "invitados", "slots": slots}

    if (re.search(r"\bconfirmad[oa]s?\b|\basistir(?:a|á|an|án)\b|\basiste[nrs]?\b", t)
        or ("qr" in t and "enviad" in t)
        or ("qr" in t and "confirmad" in t)
        or re.search(r"boletos\s+confirmad", t)):
        if not _looks_like_person_query(t):
            return {"intent": "invitados", "slots": slots}

    if re.search(r"\bficha\s+de\b", t) or t.startswith("ficha "):
        tid, q = _extract_person_query(t)
        slots["target_id"], slots["target_text"] = tid, q
        return {"intent": "detail_guest", "slots": slots}

    if "bolet" in t and ("faltan" in t or "restan" in t):
        return {"intent": "count_boletos_faltan", "slots": slots}

    if re.search(r"\bcu[aá]nt[oa]s?\b|\btotal\b", t):
        return {"intent": "count_query", "slots": slots}

    if "quien es" in t or "quién es" in t:
        tid, q = _extract_person_query(t)
        slots["target_id"], slots["target_text"] = tid, q
        return {"intent": "detail_guest", "slots": slots}

    fld = _match_field(t)
    if fld is not None and _looks_like_person_query(t):
        tid, q = _extract_person_query(t)
        slots["field"] = fld
        slots["target_id"], slots["target_text"] = tid, q
        return {"intent": "fact_query", "slots": slots}

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

def _first_name(g: Dict[str, Any]) -> str:
    n = (g.get("nombre") or "").strip()
    return n.split()[0] if n else _nombre_completo(g).split()[0]

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
        "\n\n¿Te ayudo con algo más?\n"
        "- invitados de la mesa 12 · sin mesa\n"
        "- teléfono de Carmen Chávez · correo de Farah\n"
        "- ¿cuántos confirmados? · QR confirmados\n"
        "- busca a Enrique 333 o gmail.com"
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
    strong = [g for g in rows if _guest_tokens_match(g, tokens)] if tokens else []
    if strong:
        strong.sort(key=lambda g: fuzz.token_set_ratio(t, normalize(_nombre_completo(g))), reverse=True)
        return strong
    def score(g):
        s = 0
        s = max(s, fuzz.token_set_ratio(t, normalize(_nombre_completo(g))))
        s = max(s, fuzz.partial_ratio(t, normalize(g.get("correo") or "")))
        s = max(s, fuzz.partial_ratio(t, normalize(g.get("telefono") or "")))
        return s
    cand = [g for g in rows if score(g) >= 70]
    cand.sort(key=lambda g: score(g), reverse=True)
    return cand

def _auto_pick_best(candidates: List[Dict[str, Any]], query_text: str) -> Tuple[Optional[Dict[str, Any]], bool]:
    if not candidates:
        return None, True
    if len(candidates) == 1:
        return candidates[0], False
    q = normalize(query_text or "")
    def score(g):
        return max(
            fuzz.token_set_ratio(q, normalize(_nombre_completo(g))),
            fuzz.partial_ratio(q, normalize(g.get("correo") or "")),
            fuzz.partial_ratio(q, normalize(g.get("telefono") or "")),
        )
    scored = [(g, score(g)) for g in candidates]
    scored.sort(key=lambda x: x[1], reverse=True)
    top, s1 = scored[0]
    s2 = scored[1][1] if len(scored) > 1 else 0
    exact = q and (q == normalize(_nombre_completo(top)) or q in normalize(_nombre_completo(top)))
    clear = exact or s1 >= 92 or (s1 - s2 >= 12 and s1 >= 80)
    return top, (not clear)

# =========================
# Render helpers (sin emojis)
# =========================
def render_guest_card(g: Dict[str, Any]) -> str:
    nombre = _nombre_completo(g)
    mesa = _int(g.get("mesa"))
    mesa_txt = str(mesa) if mesa > 0 else "N/D"
    asistira = "Si" if _bit(g.get("asistira")) == 1 else "No"
    solo_misa = "Si" if _bit(g.get("soloMisa")) == 1 else "No"
    qr_env = "Si" if _bit(g.get("qrEnviado")) == 1 else "No"
    qr_ok = "Si" if _bit(g.get("qrConfirmado")) == 1 else "No"
    correo = g.get("correo") or "N/D"
    tel = _format_phone(g.get("telefono") or "")
    return (f"{nombre} (ID {_int(g.get('idInvitado'))})\n"
            f"- Mesa: {mesa_txt}\n"
            f"- Asistirá: {asistira} · Solo misa: {solo_misa}\n"
            f"- QR enviado: {qr_env} · QR confirmado: {qr_ok}\n"
            f"- Correo: {correo} · Teléfono: {tel}" + _suggest_next())

def render_invitados_list(items: List[Dict[str, Any]], prefix: str = "Invitados:\n", cols: Tuple[str,...]=("id","nombre")) -> str:
    lines = []
    for i, g in enumerate(items, start=1):
        nombre = _nombre_completo(g)
        parts = []
        if "index" in cols:
            parts.append(f"{i}.")
        if "nombre" in cols:
            parts.append(f"{nombre}")
        if "id" in cols:
            parts.append(f"(ID {_int(g.get('idInvitado'))})")
        if "mesa" in cols:
            mesa = _int(g.get("mesa")); mesa_txt = str(mesa) if mesa > 0 else "N/D"
            parts.append(f"- Mesa: {mesa_txt}")
        if "asistira" in cols:
            parts.append(f"- Asistirá: {'Si' if _bit(g.get('asistira')) == 1 else 'No'}")
        if "solo_misa" in cols:
            parts.append(f"- Solo misa: {'Si' if _bit(g.get('soloMisa')) == 1 else 'No'}")
        if "qr" in cols:
            parts.append(f"- QR: {'enviado' if _bit(g.get('qrEnviado')) == 1 else 'no enviado'}/"
                         f"{'confirmado' if _bit(g.get('qrConfirmado')) == 1 else 'no confirmado'}")
        if "correo" in cols:
            parts.append(f"- Correo: {g.get('correo') or 'N/D'}")
        if "tel" in cols:
            parts.append(f"- Teléfono: {_format_phone(g.get('telefono') or '')}")
        lines.append(" ".join(parts))
    if len(items) >= 10:
        lines.append(f"\nMostrando {min(10,len(items))}/{len(items)} · di “más” para continuar.")
    return prefix + "\n".join(lines)

def render_guest_welcome(g: Dict[str, Any], rows: List[Dict[str, Any]]) -> str:
    mesa = _int(g.get("mesa"))
    mesa_txt = str(mesa) if mesa > 0 else "N/D"
    qr_env = "Si" if _bit(g.get("qrEnviado")) == 1 else "No"
    qr_ok  = "Si" if _bit(g.get("qrConfirmado")) == 1 else "No"
    bol_c  = _int(g.get("boletosConfirmados"))
    nombre = _first_name(g)

    same = []
    if mesa > 0:
        same = [x for x in rows if _int(x.get("mesa")) == mesa and _int(x.get("idInvitado")) != _int(g.get("idInvitado"))]
        same = same[:6]
    names = ", ".join(_nombre_completo(x) for x in same) if same else "N/D"

    tip = MESA_TIPS.get(mesa, "")
    tip_line = f"\nTip: {tip}" if tip else ""

    return (f"Que gusto verte, {nombre}!\n"
            f"- Tu mesa: {mesa_txt}\n"
            f"- Boletos confirmados: {bol_c}\n"
            f"- QR enviado: {qr_env} · QR confirmado: {qr_ok}\n"
            f"- En tu mesa también: {names}"
            f"{tip_line}")

# =========================
# Respuestas (sin emojis)
# =========================
def compose_answer(intent: str, text: str, ctx: Dict[str, Any], session: Dict[str, Any]) -> str:
    if intent == "greeting":
        return "Hola. Soy BodaBot. Pídeme cosas de invitados, mesas, confirmaciones, QR o boletos. Si eres invitado, di: soy [tu nombre]." + _suggest_next()
    if intent == "who_are_you":
        return "Soy BodaBot, tu asistente de invitados. Puedo listar, buscar, contar y darte detalles (tel/correo/mesa/QR/boletos)."
    if intent == "help":
        return ("Puedo ayudarte con:\n"
                "- lista de invitados\n"
                "- quienes asistirán (confirmados)\n"
                "- boletos y boletos confirmados\n"
                "- invitados de la mesa 5, sin mesa, mesas (por mesa)\n"
                "- QR enviados, QR confirmados\n"
                "- invitado: soy Karla Calleros, mesa de Juan Romo, mi mesa (después de identificarte)"
                + _suggest_next())

    # Audio control intents
    if intent == "cancel":
        session["cancel"] = True
        return "Cancelado."
    if intent == "mute_on":
        session["mute"] = True
        return "Silencio activado. Seguiré respondiendo sin voz hasta que digas: habla."
    if intent == "mute_off":
        session["mute"] = False
        return "Voz activada."

    rows = TABLES.get("tblInvitados", [])
    if not rows:
        return "No hay datos en tblInvitados. Asegúrate de que data.json incluya esa tabla o un array de invitados."

    t = normalize(text)
    scope = (session.get("last_slots") or {}).get("scope") or determine_scope(t)

    def header_scoped(sc: str, n: int) -> str:
        if sc == "mesas":         return f"Mesas - {n} resultado(s)"
        if sc == "invitados":     return f"Invitados - {n}"
        if sc == "boletos":       return f"Boletos"
        if sc == "confirmados":   return f"Confirmados - {n}"
        if sc == "qr":            return f"Codigos QR - {n}"
        if sc == "solo_misa":     return f"Solo misa - {n}"
        if sc == "contacto":      return f"Contacto"
        if sc == "ficha":         return f"Ficha"
        return f"Resultados - {n}"

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
            return render_invitados_list(nxt, prefix="Más:\n", cols=("index","nombre","id","mesa"))
        return "No hay una lista previa para continuar."

    # Selección por índice
    if intent == "pick_candidate":
        pick_idx = (session.get("last_slots") or {}).get("pick_index")
        last = session.get("last", {})
        cand = last.get("candidates") or []
        if not cand:
            return "No tengo una lista previa para elegir."
        if not pick_idx or pick_idx < 1 or pick_idx > len(cand):
            return "Número fuera de rango. Di algo como: soy el numero 2."
        g = cand[pick_idx - 1]
        session["guest_id"] = _int(g.get("idInvitado"))
        return render_guest_welcome(g, rows)

    # Modo invitado: “soy X…”
    if intent == "guest_quick":
        from_slots = session.get("last_slots") or {}
        target_text = from_slots.get("target_text", "")
        cand = _find_guest_by_id_or_text(rows, None, target_text)
        if not cand:
            return "No te encontré. Dime tu nombre tal como aparece en la invitación."
        best, ambiguous = _auto_pick_best(cand, target_text)
        if not best:
            return "No te encontré. Dime tu nombre tal como aparece en la invitación."
        session["guest_id"] = _int(best.get("idInvitado"))
        msg = render_guest_welcome(best, rows)
        if ambiguous:
            msg += "\n\nSi no eres esa persona, dime su apellido o su ID."
        return msg

    # Detalle
    if intent == "detail_guest":
        from_slots = session.get("last_slots") or {}
        target_id = from_slots.get("target_id")
        target_text = from_slots.get("target_text", "")
        if not target_id and not target_text:
            tid2, q2 = _extract_person_query(t)
            target_id, target_text = tid2, q2
        cand = _find_guest_by_id_or_text(rows, target_id, target_text)
        if not cand:
            return "No encontré a esa persona. Dame su nombre, correo, teléfono o ID."
        best, ambiguous = _auto_pick_best(cand, target_text)
        g = best
        msg = render_guest_card(g)
        if ambiguous:
            msg += "\n\nSi no te referías a esta persona, dime su apellido o su ID."
        return msg

    # Campo persona
    if intent == "fact_query":
        field = session.get("last_slots", {}).get("field")
        target_id = session.get("last_slots", {}).get("target_id")
        target_text = session.get("last_slots", {}).get("target_text", "")

        if not (target_id or target_text):
            if session.get("guest_id"):
                target_id = session["guest_id"]

        if not field:
            field = _match_field(t)

        cand = _find_guest_by_id_or_text(rows, target_id, target_text)
        if not cand:
            return "No pude ubicar a esa persona. Repite nombre, correo o ID."

        best, ambiguous = _auto_pick_best(cand, target_text or "")
        g = best
        session["guest_id"] = _int(g.get("idInvitado"))
        nombre = _nombre_completo(g)

        if field == "telefono":
            msg = f"Telefono de {nombre}: {_format_phone(g.get('telefono') or 'N/D')}"
        elif field == "correo":
            msg = f"Correo de {nombre}: {g.get('correo') or 'N/D'}"
        elif field == "mesa":
            mesa = _int(g.get("mesa"))
            mesa_txt = str(mesa) if mesa > 0 else "N/D"
            msg = f"Mesa de {nombre}: {mesa_txt}"
        elif field == "boletos":
            msg = f"Boletos asignados a {nombre}: {_int(g.get('boletos'))}"
        elif field == "boletosConfirmados":
            msg = f"Boletos confirmados de {nombre}: {_int(g.get('boletosConfirmados'))}"
        elif field in ("qrEnviado","qrConfirmado"):
            env = "Si" if _bit(g.get("qrEnviado")) == 1 else "No"
            ok  = "Si" if _bit(g.get("qrConfirmado")) == 1 else "No"
            msg = f"QR enviado: {env} · QR confirmado: {ok} (de {nombre})"
        elif field == "apodo":
            msg = f"Apodo de {nombre}: {g.get('apodo') or 'N/D'}"
        elif field == "nombre":
            msg = f"Nombre completo: {nombre}"
        else:
            msg = "Puedo darte: telefono, correo, mesa, boletos, boletos confirmados, QR enviado/confirmado, apodo o nombre." + _suggest_next()

        if ambiguous:
            msg += "\n\nSi no es la persona correcta, dime su apellido o su ID."
        return msg

    # Conteos
    if intent == "count_query":
        if scope == "confirmados":
            n = sum(1 for g in rows if _bit(g.get("asistira")) == 1)
            return f"Confirmados: {n}"
        if scope == "boletos":
            total_boletos, total_boletos_conf = _agg_boletos(rows)
            faltan = max(0, total_boletos - total_boletos_conf)
            return f"Boletos - Totales: {total_boletos} - Confirmados: {total_boletos_conf} - Faltan: {faltan}"
        if scope == "qr":
            env = sum(1 for g in rows if _bit(g.get("qrEnviado")) == 1)
            ok = sum(1 for g in rows if _bit(g.get("qrConfirmado")) == 1)
            return f"QR - Enviados: {env} - Confirmados: {ok}"
        if scope == "mesas":
            bd = _mesas_breakdown(rows)
            return f"Mesas distintas: {len(bd)} - Sin mesa: {bd.get(0,0)}"
        if scope == "solo_misa":
            n = sum(1 for g in rows if _bit(g.get("soloMisa")) == 1)
            return f"Solo misa: {n}"
        n = len(rows)
        return f"Invitados: {n}"

    if intent == "count_boletos_faltan":
        total_boletos, total_boletos_conf = _agg_boletos(rows)
        faltan = max(0, total_boletos - total_boletos_conf)
        return f"Faltan {faltan} boletos por confirmar (Totales: {total_boletos} - Confirmados: {total_boletos_conf})."

    # Vista / filtros
    if intent in ("detalles_invitados", "invitados"):
        filtered = rows

        m = re.search(r"(?:de\s+(?:la\s+)?)?mesa\s*#?\s*(\d+)", t)
        mesa_val = int(m.group(1)) if m else None
        if mesa_val is not None:
            filtered = [g for g in filtered if _int(g.get("mesa")) == mesa_val]

        if "sin mesa" in t:
            filtered = [g for g in filtered if _int(g.get("mesa")) == 0]

        if re.search(r"\bconfirmad[oa]s?\b|\basistir(?:a|á|an|án)\b|\basiste[nrs]?\b", t):
            filtered = [g for g in filtered if _bit(g.get("asistira")) == 1]

        if re.search(r"\bsolo\s+misa\b", t):
            filtered = [g for g in filtered if _bit(g.get("soloMisa")) == 1]

        if ("qr" in t and "enviad" in t):
            filtered = [g for g in filtered if _bit(g.get("qrEnviado")) == 1]
        if ("qr" in t and "confirmad" in t):
            filtered = [g for g in filtered if _bit(g.get("qrConfirmado")) == 1]

        if re.search(r"\bboletos\s+confirmad", t):
            filtered = [g for g in filtered if _int(g.get("boletosConfirmados")) > 0]

        m_tel = re.search(r"\b(?:tel(?:efono)?|cel|celular|whats(?:app)?)\s*([0-9]{2,})", t)
        if m_tel:
            phone_prefix = m_tel.group(1)
            def phone_has_prefix(g):
                digits = "".join(DIGITS_RE.findall(g.get("telefono") or ""))
                return phone_prefix in digits
            filtered = [g for g in filtered if phone_has_prefix(g)]

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

        page = filtered[:10]
        session["last"] = {"type": "invitados", "candidates": filtered, "shown": len(page), "display": page}

        if scope == "mesas":
            if mesa_val is None and "sin mesa" not in t:
                bd = _mesas_breakdown(filtered)
                top = [f"Mesa {k or 0}: {v}" for k, v in list(bd.items())[:80]]
                return header_scoped(scope, len(bd)) + "\n\n" + ("Por mesa:\n" + "\n".join(top) if top else "N/D")
            return header_scoped(scope, len(filtered)) + "\n\n" + render_invitados_list(page, cols=("index","nombre","id","mesa"))

        elif scope == "invitados":
            return header_scoped(scope, len(filtered)) + "\n\n" + render_invitados_list(page, cols=("index","nombre","id","mesa"))

        elif scope == "confirmados":
            only_conf = [g for g in filtered if _bit(g.get("asistira")) == 1]
            page2 = only_conf[:10]
            return header_scoped(scope, len(only_conf)) + "\n\n" + render_invitados_list(page2, cols=("index","nombre","id","mesa"))

        elif scope == "solo_misa":
            only_mass = [g for g in filtered if _bit(g.get("soloMisa")) == 1]
            page2 = only_mass[:10]
            if not page2:
                return header_scoped(scope, 0) + "\n\nNo hay invitados marcados como solo misa."
            return header_scoped(scope, len(only_mass)) + "\n\n" + render_invitados_list(page2, cols=("index","nombre","id","mesa"))

        elif scope == "qr":
            want_enviados = ("qr" in t and "enviad" in t)
            want_confirmados = ("qr" in t and "confirmad" in t)
            if want_enviados and not want_confirmados:
                only = [g for g in filtered if _bit(g.get("qrEnviado")) == 1]
                return header_scoped(scope, len(only)) + " - enviados\n\n" + render_invitados_list(only[:10], cols=("index","nombre","id"))
            if want_confirmados and not want_enviados:
                only = [g for g in filtered if _bit(g.get("qrConfirmado")) == 1]
                return header_scoped(scope, len(only)) + " - confirmados\n\n" + render_invitados_list(only[:10], cols=("index","nombre","id"))
            enviados = sum(1 for g in filtered if _bit(g.get("qrEnviado")) == 1)
            confirm = sum(1 for g in filtered if _bit(g.get("qrConfirmado")) == 1)
            return header_scoped(scope, enviados + confirm) + f"\n\nQR enviados: {enviados} - QR confirmados: {confirm}\n(Pide: QR enviados o QR confirmados para ver la lista.)"

        elif scope == "boletos":
            total_boletos, total_boletos_conf = _agg_boletos(filtered if filtered else rows)
            faltan = max(0, total_boletos - total_boletos_conf)
            if re.search(r"\bboletos\s+confirmad", t):
                only = [g for g in filtered if _int(g.get("boletosConfirmados")) > 0]
                return (header_scoped(scope, len(only)) +
                        f"\n\nConfirmados: {total_boletos_conf} / {total_boletos} - Faltan: {faltan}\n\n" +
                        render_invitados_list(only[:10], cols=("index","nombre","id")))
            return header_scoped(scope, 1) + f"\n\nTotales: {total_boletos} - Confirmados: {total_boletos_conf} - Faltan: {faltan}"

        return f"{header_scoped('general', len(filtered))}\n\n" + render_invitados_list(page, cols=("index","nombre","id","mesa"))

    ctx_preview = len(ctx.get("tblInvitados", []))
    return (f"Listo para ayudarte con tblInvitados (contexto: {ctx_preview} registros).\n"
            f"Ejemplos: soy Karla Calleros, mi mesa, lista de invitados, confirmados, boletos, invitados de la mesa 4, qr confirmados, ficha de Farah."
            + _suggest_next())

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

# ===== Sanitizado de emojis/símbolos =====
EMOJI_RE = re.compile(
    "[" 
    "\U0001F600-\U0001F64F"  # Emoticons
    "\U0001F300-\U0001F5FF"  # Pictogramas
    "\U0001F680-\U0001F6FF"  # Transporte/mapas
    "\U0001F1E0-\U0001F1FF"  # Banderas
    "\U00002700-\U000027BF"  # Dingbats (incluye ✔ ✉)
    "\U0001F900-\U0001F9FF"  # Suplementarios
    "\U00002600-\U000026FF"  # Símbolos varios
    "\U00002B00-\U00002BFF"  # Flechas
    "\U00002300-\U000023FF"  # Técnicos
    "\U0001FA70-\U0001FAFF"  # Extensiones
    "\U0001F700-\U0001F77F"  # Alquímicos
    "]",
    flags=re.UNICODE
)

def strip_emojis_for_tts(text: str) -> str:
    if not text:
        return text
    text = re.sub(r"[\u200D\uFE0E\uFE0F]", "", text)  # ZWJ y variation selectors
    text = EMOJI_RE.sub("", text)                     # iconos/emoji/símbolos
    text = re.sub(r"\s{2,}", " ", text).strip()
    return text

def _choose_voice(language_code: str, voice_name: Optional[str]) -> texttospeech.VoiceSelectionParams:
    # Prioridad: voice_name del request → TTS_VOICE → TTS_VOICE_FALLBACK
    name = (voice_name or TTS_VOICE or TTS_VOICE_FALLBACK).strip()
    # Si el nombre ya incluye el prefijo de idioma (p. ej. es-ES-...), úsalo. Si no, toma language_code o LANG_CODE.
    lang_from_name = "-".join(name.split("-")[:2]) if "-" in name else None
    lang = lang_from_name or language_code or LANG_CODE
    return texttospeech.VoiceSelectionParams(language_code=lang, name=name)


def tts_mp3(text: str, language_code: str = LANG_CODE, voice_name: Optional[str] = None) -> bytes:
    cli = tts_client()
    clean_text = strip_emojis_for_tts(text)
    synthesis_in = texttospeech.SynthesisInput(text=clean_text)
    voice = _choose_voice(language_code, voice_name)
    cfg = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)
    out = cli.synthesize_speech(input=synthesis_in, voice=voice, audio_config=cfg)
    return out.audio_content

def tts_wav_linear16(text: str, language_code: str = LANG_CODE, voice_name: Optional[str] = None) -> bytes:
    cli = tts_client()
    clean_text = strip_emojis_for_tts(text)
    synthesis_in = texttospeech.SynthesisInput(text=clean_text)
    voice = _choose_voice(language_code, voice_name)
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
    temperature: Optional[float] = 0.0

class AskResponse(BaseModel):
    model: str
    answer: str
    question: str
    used_sections: List[str]
    intent: str
    session_id: str

class GuestCheckinRequest(BaseModel):
    idInvitado: Optional[int] = None
    nombre: Optional[str] = None
    arrived: Optional[bool] = True

class MuteRequest(BaseModel):
    enabled: bool

# =========================
# Endpoints
# =========================
@app.get("/meta")
def meta():
    return {
        "name": "BodaBot API (Invitados & Anfitrión)",
        "version": "5.0-sin-emojis",
        "endpoints": [
            "/health", "/schema", "/tables", "/table/{name}", "/search",
            "/invitados/summary", "/invitados/find", "/invitados/{idInvitado}",
            "/ask", "/ask_audio", "/ask_audio_wav",
            "/audio/mute", "/audio/cancel",
            "/guest/checkin", "/refresh", "/tts"  # 👈 agrega aquí también
        ],
    }


class TtsRequest(BaseModel):
    text: str
    language: Optional[str] = None
    voice_name: Optional[str] = None
    format: Optional[str] = "mp3"  # "mp3" | "wav"

@app.post("/tts")
def tts(req: TtsRequest):
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="Texto vacío.")
    lang = req.language or LANG_CODE
    if req.format == "wav":
        audio = tts_wav_linear16(req.text, language_code=lang, voice_name=req.voice_name)
        return {
            "audio_base64": base64.b64encode(audio).decode("utf-8"),
            "mime": "audio/wav"
        }
    else:
        audio = tts_mp3(req.text, language_code=lang, voice_name=req.voice_name)
        return {
            "audio_base64": base64.b64encode(audio).decode("utf-8"),
            "mime": "audio/mpeg"
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
        "mesa_tips_loaded": bool(MESA_TIPS)
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
    raw_answer = compose_answer(nlu["intent"], req.question, ctx, session)
    session["last_intent"] = nlu["intent"]

    # SIEMPRE devolvemos texto sin emojis/símbolos
    answer = strip_emojis_for_tts(raw_answer)

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
    answer_full = compose_answer(nlu["intent"], user_text, ctx, session)
    session["last_intent"] = nlu["intent"]

    muted = bool(session.get("mute"))
    cancelled = bool(session.pop("cancel", False) or nlu["intent"] == "cancel")

    audio_b64 = ""
    if not muted and not cancelled:
        mp3 = tts_mp3(answer_full, language_code=language)
        audio_b64 = base64.b64encode(mp3).decode("utf-8")

    # Texto SIEMPRE sin emojis/símbolos
    answer_text = strip_emojis_for_tts(answer_full)

    return {
        "texto_usuario": user_text,
        "respuesta_texto": answer_text,
        "audio_base64": audio_b64,
        "mime": "audio/mpeg" if audio_b64 else None,
        "used_sections": list(ctx.keys()),
        "intent": nlu["intent"],
        "session_id": sid,
        "muted": muted,
        "cancelled": cancelled,
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
        answer_full = compose_answer(nlu["intent"], user_text, ctx, session)
        session["last_intent"] = nlu["intent"]

        muted = bool(session.get("mute"))
        cancelled = bool(session.pop("cancel", False) or nlu["intent"] == "cancel")

        audio_b64 = ""
        mime = None
        if not muted and not cancelled:
            wav_bytes = tts_wav_linear16(answer_full, language_code=language)
            audio_b64 = base64.b64encode(wav_bytes).decode("utf-8")
            mime = "audio/wav"

        # Texto SIEMPRE sin emojis/símbolos
        answer_text = strip_emojis_for_tts(answer_full)

        return {
            "texto_usuario": user_text,
            "respuesta_texto": answer_text,
            "audio_wav_base64": audio_b64,
            "mime": mime,
            "used_sections": list(ctx.keys()),
            "intent": nlu["intent"],
            "session_id": sid,
            "muted": muted,
            "cancelled": cancelled,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error procesando la solicitud de audio: {str(e)}")

# =========================
# Audio controls (REST)
# =========================
@app.post("/audio/mute")
def audio_mute(req: MuteRequest, x_session_id: Optional[str] = Header(None)):
    sid, session = _get_session(None, x_session_id)
    session["mute"] = bool(req.enabled)
    return {"ok": True, "muted": session["mute"], "session_id": sid}

@app.post("/audio/cancel")
def audio_cancel(x_session_id: Optional[str] = Header(None)):
    sid, session = _get_session(None, x_session_id)
    session["cancel"] = True
    return {"ok": True, "cancelled": True, "session_id": sid}

# =========================
# Guest Check-in (opcional)
# =========================
class GuestCard(BaseModel):
    idInvitado: int
    nombre: str
    mesa: int
    asistioBoda: int

@app.post("/guest/checkin")
def guest_checkin(req: GuestCheckinRequest, x_session_id: Optional[str] = Header(None)) -> Dict[str, Any]:
    rows = TABLES.get("tblInvitados", [])
    if not rows:
        raise HTTPException(status_code=400, detail="No hay invitados cargados.")
    guest = None
    if req.idInvitado:
        cand = _find_guest_by_id_or_text(rows, req.idInvitado, "")
        guest = cand[0] if cand else None
    elif req.nombre:
        cand = _find_guest_by_id_or_text(rows, None, req.nombre)
        if cand:
            guest = _auto_pick_best(cand, req.nombre)[0]
    if not guest:
        raise HTTPException(status_code=404, detail="Invitado no encontrado.")

    guest["asistioBoda"] = 1 if (req.arrived is None or req.arrived) else 0
    guest["accedio"] = datetime.now().isoformat(timespec="seconds")

    return {
        "ok": True,
        "idInvitado": _int(guest.get("idInvitado")),
        "nombre": _nombre_completo(guest),
        "mesa": _int(guest.get("mesa")),
        "asistioBoda": _bit(guest.get("asistioBoda")),
        "mensaje": "Check-in registrado (memoria en runtime)."
    }
