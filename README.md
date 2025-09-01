Cosmetics Safety Scanner — Detect Harmful Ingredients from Images

This project helps you quickly check whether a cosmetic product may contain harmful or sensitive ingredients. Just upload a photo of the ingredient list and the system will scan it for potential concerns.

✨ What It Does

Upload a product image → the backend reads the text using OCR (Optical Character Recognition).

The detected text is checked against rules for known allergens, preservatives, UV filters, alcohols, and controversial additives.

(Optional) An AI model (GPT) gives a short, user-friendly summary of the safety profile.

Results are returned in a clear, structured JSON format, and displayed nicely in the Lovable frontend.

🏗 How It Works (in simple terms)

Frontend (Lovable)

Clean interface where you upload an image.

Shows analysis results in an easy-to-read way.

Backend (FastAPI + Python)

OCR: Reads text from the photo. Uses Tesseract (default) and optionally Hugging Face TrOCR.

Rules engine: Flags keywords like PEG/PPG, Homosalate, Phenoxyethanol, or Fragrance.

Optional GPT: Summarizes findings into short, plain-language safety notes.

Fallbacks: If AI fails or isn’t enabled, rules-only results are always provided.

🚀 Getting Started
1. Install Requirements

Python 3.10+

Tesseract OCR installed on your system (e.g. sudo apt-get install tesseract-ocr on Ubuntu)

Install Python packages:

pip install -r requirements.txt

2. Run the API
uvicorn app:app --reload


Check health:

http://localhost:8000/health
→ {"ok": true}

3. Use the Frontend

The Lovable app connects to this backend. Upload an image of an ingredient list and click “Analyze for Harmful Content”.

📊 Example Result

If you upload a sunscreen ingredient list, the response may look like this:

{
  "ocr_text_corrected": "Ingredients: Water, Homosalate, Phenoxyethanol, Fragrance",
  "rules_flags": {
    "allergens": ["fragrance/parfum"],
    "problematic_additives": ["PEG/PPG", "homosalate", "phenoxyethanol"]
  },
  "analysis": {
    "safe_overall": true,
    "reasons": [
      "Includes homosalate (UV filter); restricted in some regions and may irritate sensitive skin.",
      "Contains PEG/PPG surfactants which some users avoid.",
      "Uses phenoxyethanol as a preservative; generally safe at low % but can irritate sensitive skin."
    ],
    "advise": "Review label; re-upload a clearer photo if OCR looks wrong."
  }
}

🧠 Why This Matters

Many people are sensitive to fragrance, preservatives, or UV filters.

Ingredient lists are often long and hard to parse.

This tool highlights the main potential concerns automatically, giving you quick peace of mind.

⚖️ Disclaimer

This scanner is for informational purposes only. It does not replace professional medical or dermatological advice. Always consult a qualified expert if you have concerns.

📌 Future Ideas

Highlight flagged ingredients directly in the UI

Add more region-specific ingredient restrictions

Improve OCR on low-quality / WhatsApp-compressed images
