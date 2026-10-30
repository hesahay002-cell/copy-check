import os
import io
import re
import base64
import html
import tempfile
from typing import List, Optional

from fastapi import FastAPI, Form, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from openai import OpenAI
from PIL import Image, ImageOps
from pypdf import PdfReader
import openpyxl

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors


# ------------------------------------------------------------
# SAFAL ACADEMY LIVE - AI COPY EVALUATOR BACKEND
# Render Start Command:
# uvicorn main:app --host 0.0.0.0 --port $PORT
# ------------------------------------------------------------

app = FastAPI(title="Safal Academy Live - AI Copy Evaluator", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API key is read ONLY from Render Environment Variables.
# Render variable name must be: GROQ_API_KEY
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b").strip()

client = None
if GROQ_API_KEY:
    client = OpenAI(
        api_key=GROQ_API_KEY,
        base_url="https://api.groq.com/openai/v1",
        timeout=180.0,
        max_retries=1,
    )

# NOTE: These dictionaries are temporary memory storage.
# Render restarts/redeploys can clear them. Re-publish tests after a restart.
TESTS_DB = {}
RESULTS_DB = {}

TEACHER_PIN = "525077"  # Kept as requested. Change it later if needed.


# -------------------------- HELPERS --------------------------

def extract_pdf_text(pdf_bytes: bytes) -> str:
    """Extract selectable text from a PDF. Scanned image-only PDFs may return empty text."""
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        pages = []
        for page in reader.pages:
            pages.append(page.extract_text() or "")
        return "\n".join(pages).strip()
    except Exception:
        return ""


def image_to_data_url(raw_bytes: bytes) -> str:
    """Compress an uploaded image and return a base64 data URL."""
    try:
        with Image.open(io.BytesIO(raw_bytes)) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            image.thumbnail((1400, 1400))
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=72, optimize=True)
            encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
            return "data:image/jpeg;base64," + encoded
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Uploaded answer page is not a valid image: {exc}"
        )


def get_ai_client():
    if not GROQ_API_KEY or client is None:
        raise HTTPException(
            status_code=503,
            detail="GROQ_API_KEY is missing in Render Environment Variables."
        )
    return client


def ask_groq_for_batch(prompt: str, image_urls: list) -> str:
    """Qwen vision accepts a maximum of 3 images in one request."""
    content = [{"type": "text", "text": prompt}]
    for image_url in image_urls:
        content.append({
            "type": "image_url",
            "image_url": {"url": image_url}
        })

    response = get_ai_client().chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": content}],
        max_completion_tokens=3000,
    )
    answer = response.choices[0].message.content
    if not answer:
        raise RuntimeError("AI returned an empty response.")
    return answer


def finalise_marks(notes: str, test: dict) -> str:
    prompt = f"""
You are a careful CBSE/ICSE school examiner for Safal Academy Live, Giridih.

Evaluate the student's complete answer sheet using the question paper and ideal answer
reference below. The page-wise AI observations are also provided.

TEST DETAILS
Subject: {test['subject']}
Class: {test['target_class']}
Test code: {test['test_code']}
Maximum marks: {test['total_marks']}

QUESTION PAPER TEXT:
{test.get('sqp_text') or '[No selectable text was extracted from the question paper PDF.]'}

IDEAL ANSWER TEXT:
{test.get('ideal_text') or '[No selectable text was extracted from the ideal answer PDF.]'}

PAGE-WISE OBSERVATIONS:
{notes[:24000]}

INSTRUCTIONS:
1. Award marks only for answers actually visible in the student's uploaded pages.
2. Compare with the ideal answer where available. Give credit for scientifically/mathematically
   correct equivalent methods and wording.
3. Do not invent answers or assume missing pages.
4. Keep awarded marks between 0 and {test['total_marks']}.
5. Mention question-wise marks, correct points, mistakes, missing steps and suggestions.
6. If reference text is unavailable, clearly disclose that limitation in feedback.
7. Output exactly in this format:
MARKS: <number>
FEEDBACK:
<detailed question-wise evaluation and overall advice>
"""
    response = get_ai_client().chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=4000,
    )
    answer = response.choices[0].message.content
    if not answer:
        raise RuntimeError("AI returned an empty final evaluation.")
    return answer


def parse_marks_and_feedback(ai_reply: str, total_marks: float):
    match = re.search(r"MARKS\s*:\s*(-?\d+(?:\.\d+)?)", ai_reply, re.IGNORECASE)
    if not match:
        raise RuntimeError("AI response did not contain a valid MARKS: value.")
    marks = float(match.group(1))
    marks = max(0.0, min(marks, float(total_marks)))
    feedback_match = re.search(
        r"FEEDBACK\s*:\s*(.*)", ai_reply, re.IGNORECASE | re.DOTALL
    )
    feedback = feedback_match.group(1).strip() if feedback_match else ai_reply.strip()
    return marks, feedback


def safe_pdf_text(value) -> str:
    return html.escape(str(value)).replace("\n", "<br/>")


def make_result_pdf(student_name: str, student_class: str, school_name: str,
                    test: dict, awarded_marks: float, feedback: str) -> str:
    fd, pdf_path = tempfile.mkstemp(suffix=".pdf")
    os.close(fd)

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "MainTitle", parent=styles["Heading1"], fontSize=20,
        textColor=colors.white, alignment=1, spaceAfter=8
    )
    subtitle_style = ParagraphStyle(
        "SubTitle", parent=styles["Heading3"], fontSize=11,
        textColor=colors.white, alignment=1
    )
    feedback_style = ParagraphStyle(
        "FeedbackStyle", parent=styles["Normal"], fontSize=10,
        leading=15, textColor=colors.HexColor("#1F2937")
    )
    story = []

    header = Table(
        [[Paragraph("<b>SAFAL ACADEMY LIVE</b>", title_style)],
         [Paragraph("Barganda, Giridih - Detailed Evaluation Report", subtitle_style)]],
        colWidths=[500]
    )
    header.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#1E3A8A")),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
    ]))
    story.append(header)
    story.append(Spacer(1, 18))

    details = [
        ["Student Name:", student_name, "Class:", student_class],
        ["School Name:", school_name, "Evaluated By:", "RAVISH SIR"],
        ["Test Code:", test["test_code"], "Subject:", test["subject"]],
    ]
    details_table = Table(details, colWidths=[90, 160, 90, 160])
    details_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#F3F4F6")),
        ("BACKGROUND", (2, 0), (2, -1), colors.HexColor("#F3F4F6")),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.7, colors.HexColor("#D1D5DB")),
        ("PADDING", (0, 0), (-1, -1), 8),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(details_table)
    story.append(Spacer(1, 18))

    percentage = (awarded_marks / test["total_marks"] * 100) if test["total_marks"] else 0
    mark_color = colors.HexColor("#16A34A") if percentage >= 40 else colors.HexColor("#DC2626")
    mark_bg = colors.HexColor("#ECFDF5") if percentage >= 40 else colors.HexColor("#FEF2F2")
    marks_table = Table(
        [[Paragraph(
            f"<b>TOTAL MARKS AWARDED: {awarded_marks:g} / {test['total_marks']:g}</b>",
            ParagraphStyle("Marks", fontSize=15, textColor=mark_color, alignment=1)
        )]],
        colWidths=[500]
    )
    marks_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), mark_bg),
        ("BOX", (0, 0), (-1, -1), 1.5, mark_color),
        ("PADDING", (0, 0), (-1, -1), 14),
    ]))
    story.append(marks_table)
    story.append(Spacer(1, 18))
    story.append(Paragraph(
        "<b><font color='#2563EB'>Detailed Evaluation Feedback & Deductions</font></b>",
        styles["Heading3"]
    ))
    story.append(Spacer(1, 6))
    story.append(Paragraph(safe_pdf_text(feedback), feedback_style))
    story.append(Spacer(1, 24))
    story.append(Paragraph(
        "<b>POWERED BY RAVISH SIR</b>",
        ParagraphStyle("Footer", parent=styles["Italic"], alignment=1,
                       textColor=colors.HexColor("#DC2626"))
    ))

    SimpleDocTemplate(pdf_path, pagesize=letter).build(story)
    return pdf_path


# -------------------------- BASIC ROUTES --------------------------

@app.get("/")
def home():
    return {
        "service": "Safal Academy Live AI Copy Evaluator",
        "status": "online",
        "docs": "/docs",
        "evaluation_route": "/student/evaluate"
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "groq_key_configured": bool(GROQ_API_KEY),
        "model": GROQ_MODEL
    }


# -------------------------- TEACHER ROUTES --------------------------

@app.post("/teacher/login")
def teacher_login(password: str = Form(...)):
    if password == TEACHER_PIN:
        return {"status": "success"}
    raise HTTPException(status_code=401, detail="Invalid PIN")


@app.post("/teacher/upload")
async def upload_test(
    test_code: str = Form(...),
    subject: str = Form(...),
    target_class: str = Form(...),
    total_marks: float = Form(...),
    expiration: Optional[str] = Form(None),
    sqp: UploadFile = File(...),
    ideal_answer: UploadFile = File(...)
):
    if total_marks <= 0:
        raise HTTPException(status_code=400, detail="Total marks must be greater than zero.")

    sqp_bytes = await sqp.read()
    ideal_bytes = await ideal_answer.read()

    if not sqp_bytes or not ideal_bytes:
        raise HTTPException(status_code=400, detail="Question paper and ideal answer PDFs are required.")

    test_id = f"{target_class}_{test_code}".strip()
    TESTS_DB[test_id] = {
        "id": test_id,
        "test_code": test_code,
        "subject": subject,
        "target_class": target_class,
        "total_marks": float(total_marks),
        "expiration": expiration,
        "sqp_bytes": sqp_bytes,
        "ideal_bytes": ideal_bytes,
        "sqp_text": extract_pdf_text(sqp_bytes),
        "ideal_text": extract_pdf_text(ideal_bytes),
    }
    RESULTS_DB[test_id] = []
    return {"status": "published", "test_id": test_id}


@app.get("/teacher/tests/{target_class}")
def get_teacher_tests(target_class: str):
    tests = []
    for test in TESTS_DB.values():
        if test["target_class"] == target_class:
            tests.append({
                "id": test["id"],
                "test_code": test["test_code"],
                "subject": test["subject"],
            })
    return {"tests": tests}


@app.get("/teacher/results/pdf/{test_id}")
def download_results_pdf(test_id: str):
    if test_id not in TESTS_DB:
        raise HTTPException(status_code=404, detail="Test not found. Please publish the test again.")

    results = RESULTS_DB.get(test_id, [])
    fd, pdf_path = tempfile.mkstemp(suffix=".pdf")
    os.close(fd)

    styles = getSampleStyleSheet()
    story = [
        Paragraph("<b>Safal Academy Live - Merit List</b>", styles["Title"]),
        Spacer(1, 12)
    ]
    for index, result in enumerate(
        sorted(results, key=lambda item: item["marks"], reverse=True), start=1
    ):
        line = (
            f"{index}. <b>{safe_pdf_text(result['name'])}</b> "
            f"({safe_pdf_text(result['school'])}) - "
            f"Marks: {result['marks']:g} / {result['total']:g}"
        )
        story.append(Paragraph(line, styles["Normal"]))
        story.append(Spacer(1, 6))

    SimpleDocTemplate(pdf_path, pagesize=letter).build(story)
    return FileResponse(pdf_path, media_type="application/pdf", filename="Merit_List.pdf")


@app.get("/teacher/results/excel/{test_id}")
def download_results_excel(test_id: str):
    if test_id not in TESTS_DB:
        raise HTTPException(status_code=404, detail="Test not found. Please publish the test again.")

    results = RESULTS_DB.get(test_id, [])
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "Merit List"
    sheet.append(["Rank", "Student Name", "Email", "Phone", "School", "Marks Obtained", "Total Marks"])

    ordered = sorted(results, key=lambda item: item["marks"], reverse=True)
    for rank, result in enumerate(ordered, start=1):
        sheet.append([
            rank, result["name"], result["email"], result["phone"],
            result["school"], result["marks"], result["total"]
        ])

    fd, excel_path = tempfile.mkstemp(suffix=".xlsx")
    os.close(fd)
    workbook.save(excel_path)
    return FileResponse(
        excel_path,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename="Merit_List.xlsx"
    )


# -------------------------- STUDENT ROUTES --------------------------

@app.get("/student/tests")
def get_student_tests(student_class: str):
    active_tests = []
    for test in TESTS_DB.values():
        if test["target_class"] == student_class:
            active_tests.append({
                "id": test["id"],
                "test_code": test["test_code"],
                "subject": test["subject"],
                "total_marks": test["total_marks"]
            })
    return {"tests": active_tests}


@app.post("/student/evaluate")
async def evaluate_student_copy(
    student_name: str = Form(...),
    student_email: str = Form(...),
    student_phone: str = Form(...),
    school_name: str = Form(...),
    student_class: str = Form(...),
    test_id: str = Form(...),
    answer_pages: List[UploadFile] = File(...)
):
    if test_id not in TESTS_DB:
        raise HTTPException(
            status_code=404,
            detail="Test not found on server. Teacher must publish the test again."
        )

    if not answer_pages:
        raise HTTPException(status_code=400, detail="Please upload at least one answer page.")

    if len(answer_pages) > 20:
        raise HTTPException(status_code=400, detail="Maximum 20 answer pages are allowed.")

    test = TESTS_DB[test_id]

    # Convert all answer images first. Qwen vision supports max 3 images per request.
    image_urls = []
    for page in answer_pages:
        raw = await page.read()
        if not raw:
            raise HTTPException(status_code=400, detail=f"Empty upload: {page.filename or 'answer page'}")
        image_urls.append(image_to_data_url(raw))

    # Ask the vision model to inspect the answer pages in batches of up to 3.
    page_notes = []
    try:
        for start in range(0, len(image_urls), 3):
            batch = image_urls[start:start + 3]
            first_page = start + 1
            last_page = start + len(batch)

            batch_prompt = f"""
You are assisting a strict but fair school examiner.
Inspect only the student's handwritten answer pages {first_page} to {last_page}.
Test: {test['test_code']} | Class: {test['target_class']} | Subject: {test['subject']}.
Maximum test marks: {test['total_marks']}.

Question paper text (if extractable):
{test.get('sqp_text') or '[No selectable text extracted from question paper PDF.]'}

Ideal answer text (if extractable):
{test.get('ideal_text') or '[No selectable text extracted from ideal answer PDF.]'}

Read the handwriting carefully. Transcribe question numbers and visible answers.
Identify correct work, mistakes, missing steps, and any unreadable portions.
Do not guess unclear handwriting. Do not assign the final overall test marks yet.
Return concise, structured page-wise observations.
"""
            page_notes.append(ask_groq_for_batch(batch_prompt, batch))

        final_reply = finalise_marks("\n\n".join(page_notes), test)
        awarded_marks, feedback = parse_marks_and_feedback(
            final_reply, test["total_marks"]
        )

    except HTTPException:
        raise
    except Exception as exc:
        # Do not create a fake zero-mark result when the AI request fails.
        error_text = str(exc)
        print(f"AI evaluation error: {error_text}")
        if "401" in error_text or "invalid_api_key" in error_text.lower():
            detail = "Groq rejected the API key. Check GROQ_API_KEY in Render Environment."
        elif "429" in error_text:
            detail = "Groq rate limit/quota reached. Wait and try again."
        elif "model" in error_text.lower() and ("not found" in error_text.lower() or "decommission" in error_text.lower()):
            detail = f"Groq model is unavailable: {GROQ_MODEL}. Check the current Groq model list."
        else:
            detail = f"AI evaluation failed: {error_text[:500]}"
        raise HTTPException(status_code=502, detail=detail)

    result_record = {
        "name": student_name,
        "email": student_email,
        "phone": student_phone,
        "school": school_name,
        "marks": awarded_marks,
        "feedback": feedback,
        "total": test["total_marks"],
    }
    RESULTS_DB.setdefault(test_id, []).append(result_record)

    pdf_path = make_result_pdf(
        student_name, student_class, school_name,
        test, awarded_marks, feedback
    )
    safe_name = re.sub(r"[^A-Za-z0-9_-]+", "_", student_name).strip("_") or "Student"
    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        filename=f"Result_{safe_name}.pdf"
    )
