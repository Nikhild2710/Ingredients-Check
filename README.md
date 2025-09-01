# Cosmetics Safety Scanner — **Detect Harmful Ingredients from Images**

This project helps you quickly check whether a cosmetic product may contain **harmful or sensitive ingredients**.  
Upload a **photo of the ingredient list** and the system will scan it for potential concerns.

---

## What It Does

- **Upload a product image** → the backend reads the text using **OCR** (Optical Character Recognition).  
- The detected text is checked against **rules** for known allergens, preservatives, UV filters, alcohols, and controversial additives.  
- (Optional) An **AI model** (GPT) gives a short, user-friendly summary of the safety profile.  
- Results are returned in a **clear, structured JSON format**, and displayed nicely in the Lovable frontend.  

---

## How It Works (in simple terms)

### Frontend (Lovable)
- Clean interface where you upload an image.  
- Shows analysis results in an easy-to-read way.  

### Backend (FastAPI + Python)
- **OCR**: Reads text from the photo. Uses Tesseract (default) and optionally Hugging Face TrOCR.  
- **Rules engine**: Flags keywords like **PEG/PPG**, **Homosalate**, **Phenoxyethanol**, or **Fragrance**.  
- **Optional GPT**: Summarizes findings into short, plain-language safety notes.  
- **Fallbacks**: If AI fails or isn’t enabled, rules-only results are always provided.  

---

## Getting Started

### 1. Install Requirements
- Python **3.10+**
- Tesseract OCR installed on your system  
  *(example: `sudo apt-get install tesseract-ocr` on Ubuntu)*

Install Python packages:

```bash
pip install -r requirements.txt
