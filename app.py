import io, os, re, json
from typing import List, Dict, Any
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from PIL import Image

import torch
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    AutoTokenizer,
    AutoModelForCausalLM
)

# ---------- OCR (HF) ----------
_OCR_MODEL_ID = os.environ.get("OCR_MODEL_ID", "microsoft/trocr-base-printed")
_ocr_processor = TrOCRProcessor.from_pretrained(_OCR_MODEL_ID)
_ocr_model = VisionEncoderDecoderModel.from_pretrained(_OCR_MODEL_ID).eval()
_device = "cuda" if torch.cuda.is_available() else "cpu"
_ocr_model.to(_device)

# ---------- LLM (HF local or Inference API stub) ----------
_LLM_MODEL_ID = os.environ.get("LLM_MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")
_llm_device_map = "auto" if torch.cuda.is_available() else None
_tokenizer = AutoTokenizer.from_pretrained(_LLM_MODEL_ID, use_fast=True)
_llm = AutoModelForCausalLM.from_pretrained(
    _LLM_MODEL_ID,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map=_llm_device_map
).eval()

from prompts import SYSTEM_PROMPT

app = FastAPI(title="Ingredient Harm Checker", version="1.0")

# ---------- Rules Layer ----------
ALLERGENS = [
    "milk","egg","fish","shellfish","crustacean","mollusk","tree nut","almond","walnut",
    "hazelnut","cashew","pistachio","pecan","peanut","wheat","soy","sesame"
]
INTOLERANCES = ["gluten","lactose"]
STIMULANTS = ["caffeine"]
ALCOHOL_NICOTINE = ["alcohol","ethanol","beer","wine","rum","vodka","whiskey","liqueur","nicotine"]
TRANS_FAT_MARKERS = ["partially hydrogenated","hydrogenated vegetable oil","hydrogenated palm oil"]
PROBLEM_ADDITIVES = [
    # colors
    "tartrazine","e102","sunset yellow","e110","allura red","e129","ponceau 4r","e124","carmoisine","e122",
    # preservatives / antioxidants
    "sodium benzoate","e211","bha","e320","bht","e321",
    # sweeteners
    "aspartame","e951","acesulfame k","e950","saccharin","e954","sucralose","e955",
    # flavor enhancers
    "monosodium glutamate","msg","e621"
]

def normalize_text(t: str) -> str:
    return re.sub(r"\s+", " ", t).strip()

def rules_scan(text: str) -> Dict[str, List[str]]:
    low = text.lower()
    hits = {
        "allergens": sorted({a for a in ALLERGENS if a in low}),
        "intolerances": sorted({i for i in INTOLERANCES if i in low}),
        "stimulants": sorted({s for s in STIMULANTS if s in low}),
        "alcohol_nicotine": sorted({a for a in ALCOHOL_NICOTINE if a in low}),
        "problematic_additives": sorted({p for p in PROBLEM_ADDITIVES if p in low}),
        "trans_fats": sorted({t for t in TRANS_FAT_MARKERS if t in low})
    }
    # crude hints for sugar/salt from % or wording
    hints = []
    if re.search(r"\b(high|added)\s+sugar\b", low) or re.search(r"\b[3-9]\d%?\s*sugar\b", low):
        hints.append("high sugar (heuristic)")
    if re.search(r"\b(high|added)\s+salt\b", low) or re.search(r"\b(sodium)\b", low):
        hints.append("possible high sodium (heuristic)")
    hits["high_sugar_or_sodium"] = hints
    return hits

# ---------- OCR ----------
@torch.inference_mode()
def ocr_image(pil_img: Image.Image) -> str:
    pixel_values = _ocr_processor(images=pil_img, return_tensors="pt").pixel_values.to(_device)
    generated_ids = _ocr_model.generate(pixel_values, max_new_tokens=256)
    text = _ocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return normalize_text(text)

# ---------- LLM call ----------
@torch.inference_mode()
def ask_llm(ingredients_text: str, product_name: str, rules_flags: Dict[str, List[str]]) -> Dict[str, Any]:
    user_payload = {
        "product_name": product_name or "",
        "ingredients_text": ingredients_text,
        "rules_flags": rules_flags
    }
    user_msg = json.dumps(user_payload, ensure_ascii=False)

    # Chat-style prompt (works with most instruct models)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg}
    ]
    # Turn to a single prompt string; most instruct models accept <|system|>/<|user|> or simple concatenation.
    def fmt(msgs):
        out = []
        for m in msgs:
            out.append(f"{m['role'].upper()}: {m['content']}")
        out.append("ASSISTANT:")
        return "\n\n".join(out)

    prompt = fmt(messages)
    input_ids = _tokenizer(prompt, return_tensors="pt").to(_llm.device)
    outputs = _llm.generate(
        **input_ids,
        max_new_tokens=600,
        do_sample=False
    )
    text = _tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract trailing JSON (model prints the whole chat prefix)
    json_match = re.search(r"\{.*\}\s*$", text, flags=re.DOTALL)
    if not json_match:
        # Fallback minimal object
        return {
            "safe_overall": False,
            "reasons": ["LLM failed to return JSON"],
            "findings": {k: [] for k in ["allergens","intolerances","stimulants","alcohol_nicotine","problematic_additives","trans_fats","high_sugar_or_sodium"]},
            "normalized_ingredients": [],
            "advise": "Could not analyze. Please retake a clearer photo."
        }
    try:
        return json.loads(json_match.group(0))
    except Exception:
        return {
            "safe_overall": False,
            "reasons": ["Invalid JSON from LLM"],
            "findings": {k: [] for k in ["allergens","intolerances","stimulants","alcohol_nicotine","problematic_additives","trans_fats","high_sugar_or_sodium"]},
            "normalized_ingredients": [],
            "advise": "Could not analyze. Please retake a clearer photo."
        }

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/analyze")
async def analyze(
    file: UploadFile = File(..., description="Image of the ingredients panel"),
    product_name: str = Form(default="")
):
    content = await file.read()
    pil = Image.open(io.BytesIO(content)).convert("RGB")

    # 1) OCR
    text = ocr_image(pil)

    # 2) Rules layer
    flags = rules_scan(text)

    # 3) LLM analysis
    llm_json = ask_llm(text, product_name, flags)

    # 4) Attach raw OCR for debugging
    out = {
        "product_name": product_name,
        "ocr_text": text,
        "rules_flags": flags,
        "analysis": llm_json
    }
    return JSONResponse(out)
