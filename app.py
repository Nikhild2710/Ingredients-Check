import io, os, re, json, traceback
from typing import List, Dict, Any, Tuple, Optional

from fastapi import FastAPI, UploadFile, Form, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

import torch
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    AutoTokenizer,
    AutoModelForCausalLM,
)

# =========================
# FastAPI setup + CORS
# =========================
app = FastAPI(title="Ingredient Harm Checker v1.1", version="1.1")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# Env toggles / cache dirs
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_TROCR = os.getenv("USE_TROCR", "1") == "1"
USE_LLM   = os.getenv("USE_LLM", "1") == "1"

# make HF cache dirs writable on Spaces if needed
os.makedirs(os.getenv("TRANSFORMERS_CACHE", "/data/hf_cache"), exist_ok=True)
os.makedirs(os.getenv("HF_HOME", "/data/hf_home"), exist_ok=True)

# =========================
# Lazy singletons (light models)
# =========================
# OCR: smaller base model to avoid OOM
OCR_MODEL_ID = os.getenv("OCR_MODEL_ID", "microsoft/trocr-base-printed")
_OCR: Optional[Tuple[TrOCRProcessor, VisionEncoderDecoderModel]] = None

def get_ocr() -> Tuple[TrOCRProcessor, VisionEncoderDecoderModel]:
    global _OCR
    if _OCR is None:
        proc = TrOCRProcessor.from_pretrained(OCR_MODEL_ID)
        mdl = VisionEncoderDecoderModel.from_pretrained(OCR_MODEL_ID).to(DEVICE).eval()
        _OCR = (proc, mdl)
    return _OCR

# LLM: tiny instruct model to keep memory low
LLM_MODEL_ID = os.getenv("LLM_MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct")
_LLM: Optional[Tuple[AutoTokenizer, AutoModelForCausalLM]] = None

def get_llm() -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    global _LLM
    if _LLM is None:
        tok = AutoTokenizer.from_pretrained(LLM_MODEL_ID, use_fast=True)
        llm = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL_ID,
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
            device_map="auto" if DEVICE == "cuda" else None,
        ).eval()
        _LLM = (tok, llm)
    return _LLM

# =========================
# Rules layer (food-oriented)
# =========================
ALLERGENS = [
    "milk","egg","fish","shellfish","crustacean","mollusk","tree nut","almond","walnut",
    "hazelnut","cashew","pistachio","pecan","peanut","wheat","soy","sesame"
]
INTOLERANCES = ["gluten", "lactose"]
STIMULANTS = ["caffeine"]
ALCOHOL_NICOTINE = ["alcohol","ethanol","beer","wine","rum","vodka","whiskey","liqueur","nicotine"]
TRANS_FAT_MARKERS = ["partially hydrogenated","hydrogenated vegetable oil","hydrogenated palm oil"]
PROBLEM_ADDITIVES = [
    "tartrazine","e102","sunset yellow","e110","allura red","e129","ponceau 4r","e124","carmoisine","e122",
    "sodium benzoate","e211","bha","e320","bht","e321",
    "aspartame","e951","acesulfame k","e950","saccharin","e954","sucralose","e955",
    "monosodium glutamate","msg","e621"
]

def normalize_text(t: str) -> str:
    return re.sub(r"\s+", " ", t).strip()

def rules_scan(text: str) -> Dict[str, List[str]]:
    low = (" " + (text or "").lower() + " ")
    hits = {
        "allergens": sorted({a for a in ALLERGENS if a in low}),
        "intolerances": sorted({i for i in INTOLERANCES if i in low}),
        "stimulants": sorted({s for s in STIMULANTS if s in low}),
        "alcohol_nicotine": sorted({a for a in ALCOHOL_NICOTINE if a in low}),
        "problematic_additives": sorted({p for p in PROBLEM_ADDITIVES if p in low}),
        "trans_fats": sorted({t for t in TRANS_FAT_MARKERS if t in low}),
    }
    hints = []
    if re.search(r"\b(high|added)\s+sugar\b", low) or re.search(r"\b[3-9]\d%?\s*sugar\b", low):
        hints.append("high sugar (heuristic)")
    if re.search(r"\b(high|added)\s+salt\b", low) or " sodium " in low:
        hints.append("possible high sodium (heuristic)")
    hits["high_sugar_or_sodium"] = hints
    return hits

# =========================
# OCR
# =========================
@torch.inference_mode()
def ocr_image(pil_img: Image.Image) -> str:
    if not USE_TROCR:
        return ""
    proc, mdl = get_ocr()
    pixel_values = proc(images=pil_img, return_tensors="pt").pixel_values.to(DEVICE)
    ids = mdl.generate(pixel_values, max_new_tokens=256, num_beams=1, do_sample=False)
    text = proc.batch_decode(ids, skip_special_tokens=True)[0]
    return normalize_text(text)

# =========================
# LLM helpers
# =========================
SYSTEM_PROMPT = (
    "You analyze food ingredient text for potential harm.\n"
    "Return ONLY JSON with keys: safe_overall, reasons, findings, normalized_ingredients, advise.\n"
    "Keep normalized_ingredients: []. Keep reasons <= 2 short items; advise <= 2 short sentences.\n"
)

@torch.inference_mode()
def ask_llm(ingredients_text: str, product_name: str, rules_flags: Dict[str, List[str]]) -> Dict[str, Any]:
    if not USE_LLM:
        safe = not (rules_flags.get("allergens") or rules_flags.get("trans_fats"))
        return {
            "safe_overall": bool(safe),
            "reasons": ["Rule-only summary (LLM disabled)"],
            "findings": rules_flags,
            "normalized_ingredients": [],
            "advise": "Re-run later with LLM enabled for more context."
        }

    tok, llm = get_llm()
    payload = {
        "product_name": product_name or "",
        "ingredients_text": ingredients_text or "",
        "rules_flags": rules_flags
    }
    prompt = f"SYSTEM: {SYSTEM_PROMPT}\n\nUSER: {json.dumps(payload, ensure_ascii=False)}\n\nASSISTANT:"
    input_ids = tok(prompt, return_tensors="pt").to(llm.device)
    out = llm.generate(**input_ids, max_new_tokens=512, do_sample=False)
    text = tok.decode(out[0], skip_special_tokens=True)

    m = re.search(r"\{.*\}\s*$", text, flags=re.DOTALL)
    if not m:
        safe = not (rules_flags.get("allergens") or rules_flags.get("trans_fats"))
        return {
            "safe_overall": bool(safe),
            "reasons": ["LLM returned no JSON; using rules."],
            "findings": rules_flags,
            "normalized_ingredients": [],
            "advise": "If this seems off, retake a clearer photo."
        }
    try:
        parsed = json.loads(m.group(0))
    except Exception:
        safe = not (rules_flags.get("allergens") or rules_flags.get("trans_fats"))
        parsed = {
            "safe_overall": bool(safe),
            "reasons": ["Invalid JSON from LLM; using rules."],
            "findings": rules_flags,
            "normalized_ingredients": [],
            "advise": "If this seems off, retake a clearer photo."
        }
    parsed["findings"] = rules_flags
    parsed["normalized_ingredients"] = []
    parsed.setdefault("safe_overall", True)
    parsed.setdefault("reasons", [])
    parsed.setdefault("advise", "")
    return parsed

# =========================
# Health + Echo
# =========================
@app.get("/health")
def health():
    return {"ok": True}

@app.post("/echo")
async def echo(image: UploadFile = File(...)):
    b = await image.read()
    return {
        "filename": image.filename,
        "content_type": image.content_type,
        "size_bytes": len(b)
    }

# =========================
# Analyze  (FIELD NAME = "image")
# =========================
@app.post("/analyze")
async def analyze(
    image: UploadFile = File(...),              # <-- must be File(...)
    product_name: str = Form(""),
    rotate_deg: int = Form(0),
    scale_pct: int = Form(240),
):
    try:
        raw = await image.read()
        if not raw or len(raw) < 10:
            return JSONResponse(status_code=400, content={"error": "Empty or invalid image upload"})

        pil = Image.open(io.BytesIO(raw)).convert("RGB")

        # 1) OCR
        try:
            ocr_text = ocr_image(pil)
        except Exception:
            ocr_text = ""

        # 2) Rules
        flags = rules_scan(ocr_text)

        # 3) LLM (or fallback)
        analysis = ask_llm(ocr_text, product_name, flags)

        return {
            "product_name": product_name or "",
            "ocr_text": ocr_text,
            "rules_flags": flags,
            "analysis": analysis
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "trace": traceback.format_exc()[:4000]}
        )

# =========================
# Optional warm-up (best-effort)
# =========================
@app.on_event("startup")
async def warmup():
    try:
        if USE_TROCR:
            get_ocr()
        if USE_LLM:
            get_llm()
    except Exception:
        pass
