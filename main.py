import os
import io
import base64
import re
import json
import html
import logging
import tempfile
import time
import threading
from datetime import datetime, timezone
from typing import List, Optional

import fitz  # PyMuPDF
import openpyxl
from PIL import Image, ImageOps
from fastapi import FastAPI, Form, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from openai import OpenAI
import firebase_admin
from firebase_admin import credentials, firestore

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, KeepTogether
)

# ================================================================
# SAFAL ACADEMY LIVE - VISION AI COPY EVALUATOR (FIREBASE & TEXT OPTIMIZED)
# ================================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("safal-copy-check")

app = FastAPI(
    title="Safal Academy Live - Vision AI Copy Evaluator",
    version="6.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

GROQ_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b").strip()
TEACHER_PIN = os.getenv("TEACHER_PIN", "525077").strip()

GROQ_KEYS = [
    os.getenv("GROQ_API_KEY_1", "").strip(),
    os.getenv("GROQ_API_KEY_2", "").strip(),
]
GROQ_KEYS = [key for key in GROQ_KEYS if key]
AI_CLIENTS = [
    OpenAI(
        api_key=key,
        base_url="https://api.groq.com/openai/v1",
        timeout=240.0,
        max_retries=0,
    )
    for key in GROQ_KEYS
]
_CLIENT_LOCK = threading.Lock()
_NEXT_CLIENT_INDEX = 0

# ------------------------------ FIREBASE INIT ------------------------------
db = None
firebase_creds_json = os.getenv("FIREBASE_CREDENTIALS", "").strip()
if firebase_creds_json:
    try:
        cred_dict = json.loads(firebase_creds_json)
        cred = credentials.Certificate(cred_dict)
        firebase_admin.initialize_app(cred)
        db = firestore.client()
        logger.info("Firebase initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize Firebase: {e}")
else:
    logger.warning("FIREBASE_CREDENTIALS environment variable not found. Falling back to in-memory database.")

# Fallback DBs if Firebase is not configured
TESTS_DB = {}
RESULTS_DB = {}

MAX_STUDENT_PAGES = 20
MAX_PDF_PAGES = 40
VISION_IMAGES_PER_REQUEST = 3


def image_to_data_url(raw: bytes) -> str:
    try:
        with Image.open(io.BytesIO(raw)) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            # Resolution reduced to save tokens and processing time
            image.thumbnail((1200, 1200))
            output = io.BytesIO()
            image.save(output, format="JPEG", quality=82, optimize=True)
        encoded = base64.b64encode(output.getvalue()).decode("ascii")
        return "data:image/jpeg;base64," + encoded
    except Exception as exc:
        raise ValueError("Unsupported or damaged image. Upload JPG, PNG or WEBP.") from exc


def extract_pdf_data(pdf_bytes: bytes, label: str) -> list:
    """Extracts both selectable text and image data URLs for each page."""
    try:
        document = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"{label} is not a valid PDF.") from exc

    if document.page_count < 1:
        raise HTTPException(status_code=400, detail=f"{label} PDF has no pages.")
    if document.page_count > MAX_PDF_PAGES:
        raise HTTPException(status_code=400, detail=f"{label} PDF may contain at most {MAX_PDF_PAGES} pages.")

    rendered = []
    try:
        for page_number in range(document.page_count):
            page = document.load_page(page_number)
            text = page.get_text("text").strip()
            
            pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)
            image_bytes = pix.tobytes("png")
            image_url = image_to_data_url(image_bytes)
            
            rendered.append({
                "page_number": page_number + 1,
                "text": text,
                "image": image_url
            })
    finally:
        document.close()
    return rendered

# ------------------------------ COMMON HELPERS ------------------------------

def get_ai_clients():
    if not AI_CLIENTS:
        raise HTTPException(
            status_code=503,
            detail="AI service is not configured. Add GROQ_API_KEY_1 and GROQ_API_KEY_2 in Render."
        )
    return AI_CLIENTS

def _next_client_order():
    global _NEXT_CLIENT_INDEX
    clients = get_ai_clients()
    with _CLIENT_LOCK:
        start_index = _NEXT_CLIENT_INDEX % len(clients)
        _NEXT_CLIENT_INDEX = (start_index + 1) % len(clients)
    return clients[start_index:] + clients[:start_index]

def _retry_delay_from_error(exc, default_seconds=30):
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", {}) or {}
    retry_after = headers.get("retry-after") or headers.get("x-ratelimit-reset-requests")
    if retry_after:
        try:
            match = re.match(r"^(\d+(?:\.\d+)?)(ms|s|m)?$", str(retry_after).strip(), re.I)
            if match:
                value = float(match.group(1))
                unit = (match.group(2) or "s").lower()
                if unit == "ms": value /= 1000
                elif unit == "m": value *= 60
                return max(2, min(int(value) + 2, 300))
        except Exception:
            pass
    return default_seconds

def call_vision(prompt: str, image_urls: list, max_tokens: int = 2200, json_mode: bool = False) -> str:
    clients = get_ai_clients()
    content = [{"type": "text", "text": prompt}]
    for url in image_urls:
        content.append({"type": "image_url", "image_url": {"url": url}})

    kwargs = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": "You are a careful visual reader and school examination assistant. Read page images directly."},
            {"role": "user", "content": content}
        ],
        "temperature": 0.1,
        "max_completion_tokens": max_tokens,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    for cycle in range(20):
        rate_errors = []
        for account_number, client in enumerate(_next_client_order(), start=1):
            try:
                response = client.chat.completions.create(**kwargs)
                answer = response.choices[0].message.content
                if not answer or not answer.strip():
                    raise RuntimeError("Vision AI returned an empty response.")
                return answer.strip()
            except Exception as exc:
                message = str(exc).lower()
                if any(x in message for x in ["429", "rate_limit", "rate limit", "tokens per minute"]):
                    rate_errors.append(exc)
                    continue
                raise

        if rate_errors:
            delays = [_retry_delay_from_error(err) for err in rate_errors]
            wait_seconds = max(5, min(max(delays), 300))
            time.sleep(wait_seconds)
            continue

    raise RuntimeError("Both Groq accounts remained rate-limited for 10 minutes. Please retry.")

def parse_json_response(raw: str) -> dict:
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    first = cleaned.find("{")
    last = cleaned.rfind("}")
    if first < 0 or last <= first:
        raise RuntimeError("AI response was not valid JSON.")
    return json.loads(cleaned[first:last + 1])

# ----------------------- TEACHER PDF VISION READING -------------------------

def process_teacher_reference(pages: list, pdf_kind: str) -> str:
    """Uses extracted text if sufficient, otherwise falls back to Vision AI."""
    all_text = ""
    needs_vision = False
    
    for page in pages:
        if len(page["text"]) < 50: # Arbitrary threshold, if page has very little text, it might be an image/scan
            needs_vision = True
            break
        all_text += f"\n--- Page {page['page_number']} ---\n{page['text']}"

    if not needs_vision:
        logger.info(f"Using direct text extraction for {pdf_kind} PDF.")
        return all_text
        
    logger.info(f"Falling back to Vision AI for {pdf_kind} PDF due to insufficient text.")
    return read_teacher_pdf_pages(pages, pdf_kind)

def read_teacher_pdf_pages(pages: list, pdf_kind: str) -> str:
    all_notes = []
    for start in range(0, len(pages), VISION_IMAGES_PER_REQUEST):
        batch = pages[start:start + VISION_IMAGES_PER_REQUEST]
        image_urls = [p["image"] for p in batch]
        page_details = "\n".join(
            f"Page {p['page_number']} text:\n{p['text'][:2000] if p['text'] else '[No text]'}"
            for p in batch
        )

        instruction = (
            "Transcribe every question/sub-question in original order. Preserve marks, units, formulae."
            if pdf_kind == "question" else
            "Transcribe answer numbers. Preserve steps, formulae, key terms, step-wise marks."
        )

        prompt = f"""
This is the TEACHER'S {pdf_kind.upper()}.
{instruction}
Images are pages {[p['page_number'] for p in batch]}.
Selectable text (may be incomplete):
{page_details}
Return a faithful transcription. Do not evaluate any student.
"""
        notes = call_vision(prompt, image_urls, max_tokens=4000)
        all_notes.append(f"--- {pdf_kind.upper()} PAGES {batch[0]['page_number']}-{batch[-1]['page_number']} ---\n{notes}")
    return "\n\n".join(all_notes)

# ------------------------ STUDENT COPY VISION READING -----------------------

def read_student_copy_pages(page_urls: list) -> str:
    all_page_notes = []
    total_pages = len(page_urls)

    for start in range(0, total_pages, VISION_IMAGES_PER_REQUEST):
        batch_urls = page_urls[start:start + VISION_IMAGES_PER_REQUEST]
        first_page = start + 1
        last_page = start + len(batch_urls)

        prompt = f"""
Read student answer pages {first_page}-{last_page} of {total_pages}.
1. Transcribe handwriting faithfully (equations, units, crossed-out text).
2. Record page number for every answer fragment.
3. If unnumbered, transcribe and say "number unlabelled".
4. If unreadable, mark [unclear handwriting].
5. Do not award marks.
Return transcription page by page: page number, written label, exact writing, unclear parts, continuation.
"""
        notes = call_vision(prompt, batch_urls, max_tokens=2200)
        all_page_notes.append(f"===== PAGES {first_page}-{last_page} =====\n{notes}")

    return "\n\n".join(all_page_notes)

def audit_evaluation_against_original_images(questions: list, student_transcription: str, student_page_urls: list) -> list:
    audited = [dict(item) for item in questions]
    if not student_page_urls: return audited

    for start in range(0, len(student_page_urls), VISION_IMAGES_PER_REQUEST):
        batch_urls = student_page_urls[start:start + VISION_IMAGES_PER_REQUEST]
        first_page = start + 1
        last_page = start + len(batch_urls)

        relevant = []
        for item in audited:
            answer_text = str(item.get("student_answer", "")).lower()
            page_numbers = {int(x) for x in re.findall(r"\d+", str(item.get("pages_found", "")))}
            if ("no answer found" in answer_text or item.get("review_required") or any(first_page <= n <= last_page for n in page_numbers)):
                relevant.append(item)

        if not relevant: continue

        prompt = f"""
AUDIT: Check evaluation against ORIGINAL pages {first_page}-{last_page}.
EVALUATIONS: {json.dumps(relevant, ensure_ascii=False)[:10000]}
1. If "No answer found", search images. Transcribe if found.
2. Readable wrong word is WRONG, not illegible.
3. Change marks only on visual evidence.
Return JSON: {{"corrections": [{{"question_number": "2", "student_answer_correction": "", "awarded_marks": 1.5, "deduction_reason": "", "review_required": false, "review_reason": ""}}]}}
"""
        raw = call_vision(prompt, batch_urls, max_tokens=1000, json_mode=True)
        corrections = parse_json_response(raw).get("corrections", [])
        if not isinstance(corrections, list): continue

        for correction in corrections:
            qnum = str(correction.get("question_number", "")).strip()
            target = next((item for item in audited if str(item.get("question_number", "")).strip() == qnum), None)
            if not target: continue
            
            if correction.get("student_answer_correction"): target["student_answer"] = correction["student_answer_correction"].strip()
            if correction.get("awarded_marks") is not None:
                try:
                    rev = float(correction["awarded_marks"])
                    if 0 <= rev <= float(target["max_marks"]): target["awarded_marks"] = round(rev, 2)
                except: pass
            if correction.get("deduction_reason"): target["deduction_reason"] = correction["deduction_reason"].strip()
            target["review_required"] = bool(correction.get("review_required", target["review_required"]))
            target["review_reason"] = str(correction.get("review_reason", target.get("review_reason", ""))).strip()
    return audited

def evaluate_all_questions(test: dict, student_transcription: str, student_page_urls: list = None) -> dict:
    prompt = f"""
Evaluate handwritten answer sheet.
Test: Class {test['target_class']}, {test['subject']}, {test['test_code']}, {test['total_marks']} marks.
Q-PAPER:\n{test['question_reference'][:15000]}
IDEAL-ANS:\n{test['ideal_reference'][:15000]}
STUDENT:\n{student_transcription[:20000]}

RULES:
1. Match by content/sequence, not just written number. Search all pages.
2. Use ideal-answer PDF as primary reference.
3. Don't give 0 just for wrong number or unreadable handwriting (set review_required).
4. Identify WHAT changes in science (e.g. solution vs deposit color).
Return JSON: {{"questions": [{{"question_number": "1", "question": "", "student_answer": "", "pages_found": "", "max_marks": 2, "awarded_marks": 1.5, "deduction_reason": "", "ideal_answer": "", "review_required": false, "review_reason": ""}}], "overall_feedback": "", "overall_review_required": false, "overall_review_reason": ""}}
"""
    raw = call_vision(prompt, [], max_tokens=4000, json_mode=True)
    data = parse_json_response(raw)

    cleaned = []
    max_sum, awarded_sum, pending_count = 0.0, 0.0, 0

    for index, q in enumerate(data.get("questions", []), start=1):
        try: max_marks = float(q.get("max_marks", 0))
        except: max_marks = 0.0
        
        try: awarded = float(q.get("awarded_marks", 0)) if q.get("awarded_marks") is not None else None
        except: awarded = None

        if awarded is not None:
            awarded = max(0.0, min(awarded, max_marks))
            awarded_sum += awarded
            
        review = bool(q.get("review_required", False)) or awarded is None
        if review: pending_count += 1
        max_sum += max_marks
        
        cleaned.append({
            "question_number": str(q.get("question_number", index)),
            "question": str(q.get("question", "")),
            "student_answer": str(q.get("student_answer", "")),
            "pages_found": str(q.get("pages_found", "")),
            "max_marks": round(max_marks, 2),
            "awarded_marks": round(awarded, 2) if awarded is not None else None,
            "deduction_reason": str(q.get("deduction_reason", "")),
            "ideal_answer": str(q.get("ideal_answer", "")),
            "review_required": review,
            "review_reason": str(q.get("review_reason", "")),
        })

    cleaned = audit_evaluation_against_original_images(cleaned, student_transcription, student_page_urls or [])
    awarded_sum = sum(float(item["awarded_marks"]) for item in cleaned if item["awarded_marks"] is not None)
    pending_count = sum(1 for item in cleaned if item["review_required"])

    return {
        "questions": cleaned,
        "overall_feedback": str(data.get("overall_feedback", "")),
        "overall_review_required": bool(data.get("overall_review_required", False)),
        "overall_review_reason": str(data.get("overall_review_reason", "")),
        "awarded_marks": round(awarded_sum, 2),
        "max_marks_found": round(max_sum, 2),
        "pending_count": pending_count,
        "provisional": pending_count > 0 or bool(data.get("overall_review_required", False)),
    }

# ------------------------------- PDF REPORT ---------------------------------

def para_text(value) -> str:
    value = html.escape(str(value or ""))
    value = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", value, flags=re.DOTALL)
    return value.replace("\n", "<br/>")

def build_result_pdf(student_name: str, student_class: str, school_name: str, test: dict, evaluation: dict) -> str:
    fd, pdf_path = tempfile.mkstemp(prefix="safal_result_", suffix=".pdf")
    os.close(fd)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("T", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=18, textColor=colors.white)
    head_style = ParagraphStyle("H", parent=styles["Heading3"], fontSize=11, textColor=colors.HexColor("#173B75"))
    body_style = ParagraphStyle("B", parent=styles["BodyText"], fontSize=9)

    story = []
    header = Table([[Paragraph("SAFAL ACADEMY LIVE", title_style)]], colWidths=[510])
    header.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#173B75")), ("ALIGN", (0, 0), (-1, -1), "CENTER")]))
    story.append(header)
    story.append(Spacer(1, 10))

    story.append(Table([
        ["Student", student_name, "Class", student_class],
        ["School", school_name, "Evaluated By", "RAVISH SIR"]
    ], colWidths=[75, 175, 75, 185], style=TableStyle([("GRID", (0, 0), (-1, -1), 0.5, colors.grey)])))
    story.append(Spacer(1, 10))

    total = float(test["total_marks"])
    prov = float(evaluation["awarded_marks"])
    lbl = "PROVISIONAL" if evaluation["provisional"] else "TOTAL"
    story.append(Table([[Paragraph(f"<b>{lbl} MARKS: {prov:g} / {total:g}</b>", ParagraphStyle("S", alignment=TA_CENTER, textColor=colors.green))]], colWidths=[510], style=TableStyle([("BOX", (0, 0), (-1, -1), 1, colors.green)])))
    story.append(Spacer(1, 10))

    for item in evaluation["questions"]:
        awarded = "Review" if item["awarded_marks"] is None else f"{item['awarded_marks']:g}"
        blocks = [
            Paragraph(f"Q{item['question_number']} ({awarded}/{item['max_marks']:g})", head_style),
            Paragraph(f"<b>Student:</b> {para_text(item['student_answer'])}", body_style),
            Paragraph(f"<b>Ideal:</b> {para_text(item['ideal_answer'])}", body_style),
        ]
        if item["deduction_reason"]: blocks.append(Paragraph(f"<b>Deduction:</b> {para_text(item['deduction_reason'])}", body_style))
        if item["review_required"]: blocks.append(Paragraph(f"<b><font color='red'>Review:</font></b> {para_text(item['review_reason'])}", body_style))
        story.append(Table([[blocks]], colWidths=[510], style=TableStyle([("BOX", (0, 0), (-1, -1), 0.5, colors.grey)])))
        story.append(Spacer(1, 5))

    SimpleDocTemplate(pdf_path, pagesize=A4).build(story)
    return pdf_path

# -------------------------------- API ROUTES --------------------------------

@app.get("/")
def home(): return {"service": "Safal Academy AI", "status": "online"}

@app.post("/teacher/login")
def teacher_login(password: str = Form(...)):
    if password == TEACHER_PIN: return {"status": "success"}
    raise HTTPException(status_code=401, detail="Invalid PIN")

@app.post("/teacher/upload")
async def upload_test(
    test_code: str = Form(...), subject: str = Form(...), target_class: str = Form(...),
    total_marks: float = Form(...), sqp: UploadFile = File(...), ideal_answer: UploadFile = File(...)
):
    q_bytes = await sqp.read()
    i_bytes = await ideal_answer.read()
    
    # Extract text & images
    q_pages = extract_pdf_data(q_bytes, "Question Paper")
    i_pages = extract_pdf_data(i_bytes, "Ideal Answer")

    # Generate references immediately using optimized logic
    q_ref = process_teacher_reference(q_pages, "question")
    i_ref = process_teacher_reference(i_pages, "ideal answer")

    test_id = f"{target_class}_{test_code}".strip()
    test_data = {
        "id": test_id, "test_code": test_code, "subject": subject,
        "target_class": target_class, "total_marks": float(total_marks),
        "question_reference": q_ref, "ideal_reference": i_ref,
        "created_at": datetime.now(timezone.utc).isoformat()
    }

    if db:
        # Save to Firebase without image payloads to save space
        db.collection('tests').document(test_id).set(test_data)
    else:
        TESTS_DB[test_id] = test_data

    return {"status": "published", "test_id": test_id}

@app.get("/teacher/tests/{target_class}")
def get_teacher_tests(target_class: str):
    if db:
        docs = db.collection('tests').where('target_class', '==', target_class).stream()
        tests = [{"id": d.id, "test_code": d.to_dict().get('test_code'), "subject": d.to_dict().get('subject')} for d in docs]
    else:
        tests = [{"id": t["id"], "test_code": t["test_code"], "subject": t["subject"]} for t in TESTS_DB.values() if t["target_class"] == target_class]
    return {"tests": tests}

@app.get("/student/tests")
def get_student_tests(student_class: str):
    if db:
        docs = db.collection('tests').where('target_class', '==', student_class).stream()
        tests = [{"id": d.id, "test_code": d.to_dict().get('test_code'), "subject": d.to_dict().get('subject'), "total_marks": d.to_dict().get('total_marks')} for d in docs]
    else:
        tests = [{"id": t["id"], "test_code": t["test_code"], "subject": t["subject"], "total_marks": t["total_marks"]} for t in TESTS_DB.values() if t["target_class"] == student_class]
    return {"tests": tests}

@app.post("/student/evaluate")
async def evaluate_student_copy(
    student_name: str = Form(...), school_name: str = Form(...), student_class: str = Form(...),
    test_id: str = Form(...), answer_pages: List[UploadFile] = File(...)
):
    if db:
        doc = db.collection('tests').document(test_id).get()
        if not doc.exists: raise HTTPException(status_code=404, detail="Test not found.")
        test = doc.to_dict()
    else:
        if test_id not in TESTS_DB: raise HTTPException(status_code=404, detail="Test not found.")
        test = TESTS_DB[test_id]

    student_page_urls = []
    for p in answer_pages:
        raw = await p.read()
        student_page_urls.append(image_to_data_url(raw))

    transcription = read_student_copy_pages(student_page_urls)
    evaluation = evaluate_all_questions(test, transcription, student_page_urls)
    pdf_path = build_result_pdf(student_name, student_class, school_name, test, evaluation)

    res_data = {
        "name": student_name, "school": school_name,
        "marks": evaluation["awarded_marks"], "total": test["total_marks"],
        "provisional": evaluation["provisional"],
        "created_at": datetime.now(timezone.utc).isoformat()
    }
    
    if db: db.collection('tests').document(test_id).collection('results').add(res_data)
    else: RESULTS_DB.setdefault(test_id, []).append(res_data)

    return FileResponse(pdf_path, media_type="application/pdf", filename=f"Result_{student_name}.pdf")

@app.get("/teacher/results/excel/{test_id}")
def download_results_excel(test_id: str):
    if db:
        docs = db.collection('tests').document(test_id).collection('results').stream()
        results = sorted([d.to_dict() for d in docs], key=lambda i: i.get("marks", 0), reverse=True)
    else:
        results = sorted(RESULTS_DB.get(test_id, []), key=lambda i: i["marks"], reverse=True)
        
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["Name", "School", "Marks", "Total", "Status"])
    for r in results: ws.append([r["name"], r.get("school",""), r["marks"], r["total"], "Provisional" if r.get("provisional") else "Evaluated"])
    
    fd, path = tempfile.mkstemp(suffix=".xlsx")
    os.close(fd)
    wb.save(path)
    return FileResponse(path, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", filename="MeritList.xlsx")
