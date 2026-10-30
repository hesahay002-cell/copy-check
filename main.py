import os
import io
import re
import json
import html
import tempfile
import base64
from typing import List, Optional

import fitz  # PyMuPDF: render teacher PDFs, including scanned/image PDFs
import openpyxl
from PIL import Image, ImageOps
from fastapi import FastAPI, Form, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from openai import OpenAI
from pypdf import PdfReader

from reportlab.lib.pagesizes import letter
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors


# ============================================================
# SAFAL ACADEMY LIVE - AI COPY EVALUATOR
# Render Start Command:
# uvicorn main:app --host 0.0.0.0 --port $PORT
#
# Render Environment Variables:
# GROQ_API_KEY = your new Groq API key
# GROQ_MODEL   = qwen/qwen3.8-27b (or a currently enabled
#                Groq vision model available in your account)
# ============================================================

app = FastAPI(
    title="Safal Academy Live - AI Copy Evaluator",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

# Temporary memory storage: tests/results are cleared when Render restarts.
TESTS_DB = {}
RESULTS_DB = {}

# Teacher PIN intentionally kept as requested.
TEACHER_PIN = "525077"


# ---------------------------- PDF / IMAGE HELPERS ----------------------------

def extract_pdf_text(pdf_bytes: bytes) -> str:
    """Extract selectable text; scanned PDFs are handled by PDF image rendering too."""
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        return "\n".join((page.extract_text() or "") for page in reader.pages).strip()
    except Exception:
        return ""


def image_bytes_to_data_url(raw_bytes: bytes) -> str:
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
            detail=f"Invalid answer-sheet image ({exc}). Please upload a clear JPG/PNG photo."
        )


def pdf_to_image_data_urls(pdf_bytes: bytes, label: str) -> list:
    """Render PDF pages to images so AI can read scanned and ordinary PDF pages."""
    try:
        document = fitz.open(stream=pdf_bytes, filetype="pdf")
        urls = []
        for page_number in range(len(document)):
            page = document.load_page(page_number)
            pixmap = page.get_pixmap(matrix=fitz.Matrix(1.45, 1.45), alpha=False)
            raw = pixmap.tobytes("png")
            urls.append(image_bytes_to_data_url(raw))
        document.close()
        return urls
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Could not read {label} PDF. Please upload a valid PDF. Details: {exc}"
        )


def get_ai_client():
    if not GROQ_API_KEY or client is None:
        raise HTTPException(
            status_code=503,
            detail="AI service is not configured. Please contact the teacher."
        )
    return client


def call_vision_ai(prompt: str, image_urls: list, max_tokens: int = 3000) -> str:
    content = [{"type": "text", "text": prompt}]
    for url in image_urls:
        content.append({"type": "image_url", "image_url": {"url": url}})

    response = get_ai_client().chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": content}],
        max_completion_tokens=max_tokens,
    )
    answer = response.choices[0].message.content
    if not answer or not answer.strip():
        raise RuntimeError("AI returned an empty response.")
    return answer.strip()


def transcribe_teacher_pdf(pdf_bytes: bytes, pdf_label: str) -> str:
    """
    AI reads the teacher's uploaded PDF itself.
    Text extraction supports searchable PDFs; rendered page images support scanned PDFs.
    """
    extracted_text = extract_pdf_text(pdf_bytes)
    page_images = pdf_to_image_data_urls(pdf_bytes, pdf_label)

    observations = []
    for start in range(0, len(page_images), 3):
        batch = page_images[start:start + 3]
        page_from = start + 1
        page_to = start + len(batch)

        prompt = f"""
You are reading a teacher-uploaded {pdf_label} PDF for a school examination.
These are pages {page_from} to {page_to} of that PDF.

READ THE PDF PAGE IMAGES CAREFULLY. The PDF may be scanned or handwritten.
Transcribe the visible content faithfully. Preserve question numbers, subparts,
marks, equations, chemical formulae, units, diagrams' labels and important wording.
Do not invent missing or unreadable content. Mark unclear content as [unreadable].

For a QUESTION PAPER PDF:
- List every question and sub-question in original order.
- Capture marks allotted to each question if printed.
- Preserve the exact question wording as closely as possible.

For an IDEAL ANSWER PDF:
- Match answer numbers to the question numbers.
- Transcribe the teacher's expected answer, key points, formulae and equations.
- Preserve marking scheme / step marks if shown.

SEARCHABLE PDF TEXT, if available:
{extracted_text[:12000] if extracted_text else '[No selectable text. Read the page images.]'}

Return clear structured text, not a grade and not a student evaluation.
"""
        observations.append(call_vision_ai(prompt, batch, max_tokens=3000))

    return "\n\n".join(observations)


def parse_json_from_ai(text: str) -> dict:
    """Accept plain JSON or JSON enclosed in a Markdown code fence."""
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end <= start:
        raise RuntimeError("AI did not return the required structured evaluation.")
    return json.loads(cleaned[start:end + 1])


# ---------------------------- EVALUATION HELPERS ----------------------------

def evaluate_student_with_ai(test: dict, answer_page_urls: list) -> dict:
    question_reference = test.get("question_reference", "")
    ideal_reference = test.get("ideal_reference", "")

    # First, read/transcribe the student's pages in batches.
    student_page_notes = []
    for start in range(0, len(answer_page_urls), 3):
        batch = answer_page_urls[start:start + 3]
        page_from = start + 1
        page_to = start + len(batch)

        prompt = f"""
You are assisting a school examiner. Read only the student's uploaded answer-sheet
pages {page_from} to {page_to}. Do not award marks yet.

TEACHER'S QUESTION PAPER (transcribed from the teacher's uploaded question PDF):
{question_reference[:14000]}

Transcribe each visible student answer and its question number. Preserve equations,
chemical formulae, units and calculations. State clearly when an answer is missing
or handwriting is unreadable. Do not guess. Include page numbers in your notes.
"""
        student_page_notes.append(
            call_vision_ai(prompt, batch, max_tokens=3000)
        )

    student_transcription = "\n\n".join(student_page_notes)

    # Final evaluation: force each question into the exact requested structure.
    final_prompt = f"""
You are the strict but fair examiner for Safal Academy Live, Giridih.

Evaluate this student's complete answer sheet using ONLY the teacher-uploaded
question paper and teacher-uploaded ideal-answer PDF as the official references.

TEST DETAILS
Class: {test['target_class']}
Subject: {test['subject']}
Test Code: {test['test_code']}
Maximum total marks: {test['total_marks']}

TEACHER-UPLOADED QUESTION PAPER PDF (AI-READ TRANSCRIPTION):
{question_reference[:16000]}

TEACHER-UPLOADED IDEAL ANSWER PDF (AI-READ TRANSCRIPTION):
{ideal_reference[:16000]}

STUDENT ANSWER SHEET TRANSCRIPTION FROM UPLOADED PHOTOS:
{student_transcription[:16000]}

MANDATORY EVALUATION RULES:
1. Evaluate all questions found in the teacher's question paper, in the same order and
   with the same question numbers. Do not invent additional questions.
2. Match each student answer to the correct question number. Do not confuse a student's
   written question label with the printed question number; use the actual content too.
3. Use the teacher's ideal-answer PDF as the primary marking reference. Do not replace
   it with a generic textbook answer. If the ideal PDF does not contain an answer or is
   unreadable, say so explicitly in ideal_answer.
4. For every question show:
   - question: exact question text from teacher's question paper
   - student_answer: what the student actually wrote; if blank, write "No answer found"
   - max_marks: marks allotted in the teacher's question paper
   - awarded_marks: marks earned by the student
   - deduction_reason: specific reason for every lost mark. If full marks, write "No marks deducted."
   - ideal_answer: answer from the teacher's ideal-answer PDF, preserving its method/content.
5. Do not award marks for an answer that is not present. Do not assume unreadable text.
6. Give partial marks only for correct visible points/steps. Do not deduct twice for the
   same mistake. Award mathematically/scientifically equivalent answers when justified
   by the teacher's ideal answer.
7. The sum of question max_marks must not exceed {test['total_marks']}. The sum of
   awarded_marks must not exceed the total. If question-wise marks are absent from the
   question PDF, distribute the total marks fairly across its questions and disclose
   this in overall_feedback.
8. Keep student_answer and ideal_answer readable and sufficiently detailed.
9. Return ONLY valid JSON. No Markdown fences, no text before or after JSON.
10. Use exactly this JSON structure:
{{
  "questions": [
    {{
      "question_number": "1",
      "question": "Full question text",
      "student_answer": "Student's answer or No answer found",
      "max_marks": 2,
      "awarded_marks": 1,
      "deduction_reason": "Specific reason for deduction",
      "ideal_answer": "Teacher's ideal answer"
    }}
  ],
  "overall_feedback": "Short overall feedback and improvement advice"
}}
"""
    raw_result = call_vision_ai(final_prompt, [], max_tokens=5000)
    result = parse_json_from_ai(raw_result)

    questions = result.get("questions")
    if not isinstance(questions, list) or not questions:
        raise RuntimeError("AI did not return question-wise evaluation.")

    total_max = 0.0
    total_awarded = 0.0
    cleaned_questions = []

    for index, item in enumerate(questions, start=1):
        try:
            max_marks = max(0.0, float(item.get("max_marks", 0)))
            awarded = max(0.0, float(item.get("awarded_marks", 0)))
        except (TypeError, ValueError):
            raise RuntimeError(f"Invalid marks returned for question {index}.")

        if awarded > max_marks:
            awarded = max_marks

        total_max += max_marks
        total_awarded += awarded

        cleaned_questions.append({
            "question_number": str(item.get("question_number", index)),
            "question": str(item.get("question", "Question text not readable")),
            "student_answer": str(item.get("student_answer", "No answer found")),
            "max_marks": max_marks,
            "awarded_marks": awarded,
            "deduction_reason": str(item.get("deduction_reason", "Not specified by AI")),
            "ideal_answer": str(item.get("ideal_answer", "Not available in teacher's ideal-answer PDF")),
        })

    # Scale only if AI's question totals exceed the test maximum.
    # Preserve awarded marks proportionally and clearly flag this in feedback.
    overall_feedback = str(result.get("overall_feedback", ""))
    if total_max > float(test["total_marks"]) and total_max > 0:
        factor = float(test["total_marks"]) / total_max
        for item in cleaned_questions:
            item["max_marks"] = round(item["max_marks"] * factor, 2)
            item["awarded_marks"] = round(item["awarded_marks"] * factor, 2)
        overall_feedback += (
            " Note: The AI's question-wise maximum marks exceeded the test maximum; "
            "question marks were proportionally adjusted to the test total."
        )

    total_awarded = sum(item["awarded_marks"] for item in cleaned_questions)
    total_awarded = min(total_awarded, float(test["total_marks"]))

    return {
        "questions": cleaned_questions,
        "overall_feedback": overall_feedback,
        "awarded_marks": round(total_awarded, 2),
    }


# ---------------------------- PDF REPORT ----------------------------

def reportlab_markup(text: str) -> str:
    """Escape text and convert basic Markdown bold/italic to ReportLab markup."""
    safe = html.escape(str(text or ""))
    safe = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", safe, flags=re.DOTALL)
    safe = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", safe)
    safe = safe.replace("\n", "<br/>")
    return safe


def make_result_pdf(student_name: str, student_class: str, school_name: str,
                    test: dict, evaluation: dict) -> str:
    fd, pdf_path = tempfile.mkstemp(suffix=".pdf")
    os.close(fd)

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "MainTitle", parent=styles["Heading1"], fontSize=19,
        textColor=colors.white, alignment=1, spaceAfter=5
    )
    subtitle_style = ParagraphStyle(
        "SubTitle", parent=styles["Normal"], fontSize=10,
        textColor=colors.white, alignment=1
    )
    section_style = ParagraphStyle(
        "QuestionHeading", parent=styles["Heading3"], fontSize=11,
        leading=14, textColor=colors.HexColor("#1E3A8A"), spaceBefore=8, spaceAfter=6
    )
    body_style = ParagraphStyle(
        "BodyTextCustom", parent=styles["Normal"], fontSize=9.5,
        leading=14, textColor=colors.HexColor("#1F2937"), spaceAfter=5
    )
    label_style = ParagraphStyle(
        "LabelCustom", parent=body_style, fontSize=9.5, leading=14
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
        ("TOPPADDING", (0, 0), (-1, -1), 9),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
    ]))
    story.append(header)
    story.append(Spacer(1, 14))

    details = [
        ["Student Name:", student_name, "Class:", student_class],
        ["School Name:", school_name, "Evaluated By:", "RAVISH SIR"],
        ["Test Code:", test["test_code"], "Subject:", test["subject"]],
    ]
    details_table = Table(details, colWidths=[88, 162, 90, 160])
    details_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#F3F4F6")),
        ("BACKGROUND", (2, 0), (2, -1), colors.HexColor("#F3F4F6")),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.6, colors.HexColor("#D1D5DB")),
        ("PADDING", (0, 0), (-1, -1), 7),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    story.append(details_table)
    story.append(Spacer(1, 12))

    total = float(test["total_marks"])
    awarded_total = float(evaluation["awarded_marks"])
    percentage = (awarded_total / total * 100) if total else 0
    mark_color = colors.HexColor("#15803D") if percentage >= 40 else colors.HexColor("#B91C1C")
    marks_box = Table([[
        Paragraph(
            f"<b>TOTAL MARKS: {awarded_total:g} / {total:g} "
            f"({percentage:.1f}%)</b>",
            ParagraphStyle("Marks", fontSize=14, textColor=mark_color, alignment=1)
        )
    ]], colWidths=[500])
    marks_box.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
        ("BOX", (0, 0), (-1, -1), 1, mark_color),
        ("PADDING", (0, 0), (-1, -1), 10),
    ]))
    story.append(marks_box)
    story.append(Spacer(1, 12))

    # Requested format for EVERY question:
    # Question -> Student Answer -> Allotted/Awarded Marks ->
    # Cause of Deduction -> Ideal Answer -> Next Question.
    for item in evaluation["questions"]:
        q_heading = (
            f"Question {html.escape(item['question_number'])} "
            f"({item['awarded_marks']:g} / {item['max_marks']:g} marks)"
        )
        blocks = [
            Paragraph(q_heading, section_style),
            Paragraph(f"<b>Question:</b> {reportlab_markup(item['question'])}", label_style),
            Paragraph(f"<b>Student Answer:</b> {reportlab_markup(item['student_answer'])}", label_style),
            Paragraph(
                f"<b>Allotted Marks:</b> {item['max_marks']:g} "
                f"&nbsp;&nbsp; <b>Marks Awarded:</b> {item['awarded_marks']:g}",
                label_style
            ),
            Paragraph(
                f"<b>Cause of Deduction of Marks:</b> {reportlab_markup(item['deduction_reason'])}",
                label_style
            ),
            Paragraph(f"<b>Ideal Answer:</b> {reportlab_markup(item['ideal_answer'])}", label_style),
            Spacer(1, 8),
        ]
        question_table = Table([[blocks]], colWidths=[500])
        question_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.white),
            ("BOX", (0, 0), (-1, -1), 0.7, colors.HexColor("#CBD5E1")),
            ("LEFTPADDING", (0, 0), (-1, -1), 9),
            ("RIGHTPADDING", (0, 0), (-1, -1), 9),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        story.append(KeepTogether([question_table, Spacer(1, 7)]))

    story.append(Paragraph("<b>Overall Feedback:</b>", section_style))
    story.append(Paragraph(reportlab_markup(evaluation.get("overall_feedback", "")), body_style))
    story.append(Spacer(1, 18))
    story.append(Paragraph(
        "<b>POWERED BY RAVISH SIR</b>",
        ParagraphStyle("Footer", parent=styles["Italic"], alignment=1,
                       textColor=colors.HexColor("#DC2626"))
    ))

    SimpleDocTemplate(
        pdf_path, pagesize=letter,
        rightMargin=42, leftMargin=42, topMargin=36, bottomMargin=36
    ).build(story)
    return pdf_path


# ---------------------------- GENERAL ROUTES ----------------------------

@app.get("/")
def home():
    return {
        "service": "Safal Academy Live AI Copy Evaluator",
        "status": "online",
        "version": "3.0.0",
        "docs": "/docs",
        "evaluation_route": "/student/evaluate",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "groq_key_configured": bool(GROQ_API_KEY),
        "model": GROQ_MODEL,
    }


# ---------------------------- TEACHER ROUTES ----------------------------

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
    ideal_answer: UploadFile = File(...),
):
    if total_marks <= 0:
        raise HTTPException(status_code=400, detail="Total marks must be greater than zero.")

    question_pdf = await sqp.read()
    ideal_pdf = await ideal_answer.read()

    if not question_pdf or not ideal_pdf:
        raise HTTPException(
            status_code=400,
            detail="Teacher must upload both Question Paper PDF and Ideal Answer PDF."
        )

    # Validate and read both teacher PDFs now, before publishing the test.
    question_images = pdf_to_image_data_urls(question_pdf, "Question Paper")
    ideal_images = pdf_to_image_data_urls(ideal_pdf, "Ideal Answer")
    if not question_images or not ideal_images:
        raise HTTPException(status_code=400, detail="Both teacher PDFs must contain at least one page.")

    test_id = f"{target_class}_{test_code}".strip()
    TESTS_DB[test_id] = {
        "id": test_id,
        "test_code": test_code,
        "subject": subject,
        "target_class": target_class,
        "total_marks": float(total_marks),
        "expiration": expiration,
        "sqp_bytes": question_pdf,
        "ideal_bytes": ideal_pdf,
        "sqp_text": extract_pdf_text(question_pdf),
        "ideal_text": extract_pdf_text(ideal_pdf),
        "question_images": question_images,
        "ideal_images": ideal_images,
        "question_reference": "",
        "ideal_reference": "",
        "reference_ready": False,
    }
    RESULTS_DB[test_id] = []

    return {
        "status": "published",
        "test_id": test_id,
        "message": "Question Paper PDF and Ideal Answer PDF uploaded successfully.",
    }


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

    results = sorted(
        RESULTS_DB.get(test_id, []),
        key=lambda item: item["marks"],
        reverse=True
    )
    fd, pdf_path = tempfile.mkstemp(suffix=".pdf")
    os.close(fd)
    styles = getSampleStyleSheet()
    story = [
        Paragraph("<b>Safal Academy Live - Merit List</b>", styles["Title"]),
        Spacer(1, 12),
    ]
    for rank, result in enumerate(results, start=1):
        story.append(Paragraph(
            f"{rank}. <b>{html.escape(result['name'])}</b> "
            f"({html.escape(result['school'])}) - "
            f"Marks: {result['marks']:g} / {result['total']:g}",
            styles["Normal"]
        ))
        story.append(Spacer(1, 6))

    SimpleDocTemplate(pdf_path, pagesize=letter).build(story)
    return FileResponse(pdf_path, media_type="application/pdf", filename="Merit_List.pdf")


@app.get("/teacher/results/excel/{test_id}")
def download_results_excel(test_id: str):
    if test_id not in TESTS_DB:
        raise HTTPException(status_code=404, detail="Test not found. Please publish the test again.")

    results = sorted(
        RESULTS_DB.get(test_id, []),
        key=lambda item: item["marks"],
        reverse=True
    )
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "Merit List"
    sheet.append(["Rank", "Student Name", "Email", "Phone", "School", "Marks Obtained", "Total Marks"])

    for rank, result in enumerate(results, start=1):
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


# ---------------------------- STUDENT ROUTES ----------------------------

@app.get("/student/tests")
def get_student_tests(student_class: str):
    active_tests = []
    for test in TESTS_DB.values():
        if test["target_class"] == student_class:
            active_tests.append({
                "id": test["id"],
                "test_code": test["test_code"],
                "subject": test["subject"],
                "total_marks": test["total_marks"],
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
    answer_pages: List[UploadFile] = File(...),
):
    if test_id not in TESTS_DB:
        raise HTTPException(
            status_code=404,
            detail="This test is not available on the server. Teacher must publish it again."
        )

    if not answer_pages:
        raise HTTPException(status_code=400, detail="Please upload at least one answer page.")
    if len(answer_pages) > 20:
        raise HTTPException(status_code=400, detail="Maximum 20 answer pages are allowed.")

    test = TESTS_DB[test_id]

    try:
        # First read both teacher PDFs through AI. This is done once per test,
        # then cached in memory for later students.
        if not test.get("reference_ready"):
            question_reference = transcribe_teacher_pdf(
                test["sqp_bytes"], "Question Paper"
            )
            ideal_reference = transcribe_teacher_pdf(
                test["ideal_bytes"], "Ideal Answer"
            )
            test["question_reference"] = question_reference
            test["ideal_reference"] = ideal_reference
            test["reference_ready"] = True

        # Read student pages.
        answer_page_urls = []
        for page in answer_pages:
            raw = await page.read()
            if not raw:
                raise HTTPException(
                    status_code=400,
                    detail=f"An uploaded page is empty: {page.filename or 'answer page'}"
                )
            answer_page_urls.append(image_bytes_to_data_url(raw))

        evaluation = evaluate_student_with_ai(test, answer_page_urls)

    except HTTPException:
        raise
    except Exception as exc:
        # Never send an error or fake zero-mark PDF to the student.
        error_text = str(exc)
        print(f"Evaluation failure for test {test_id}: {error_text}")

        lowered = error_text.lower()
        if "401" in error_text or "invalid_api_key" in lowered:
            detail = "Evaluation is temporarily unavailable. Please inform the teacher."
        elif "429" in error_text or "rate limit" in lowered:
            detail = "Evaluation is temporarily busy. Please try again later."
        elif "model" in lowered and ("not found" in lowered or "decommission" in lowered):
            detail = "Evaluation service configuration needs attention. Please inform the teacher."
        else:
            detail = "Evaluation could not be completed. Please try again or contact the teacher."

        raise HTTPException(status_code=502, detail=detail)

    result_record = {
        "name": student_name,
        "email": student_email,
        "phone": student_phone,
        "school": school_name,
        "marks": evaluation["awarded_marks"],
        "feedback": evaluation.get("overall_feedback", ""),
        "total": test["total_marks"],
    }
    RESULTS_DB.setdefault(test_id, []).append(result_record)

    pdf_path = make_result_pdf(
        student_name=student_name,
        student_class=student_class,
        school_name=school_name,
        test=test,
        evaluation=evaluation,
    )

    safe_name = re.sub(r"[^A-Za-z0-9_-]+", "_", student_name).strip("_") or "Student"
    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        filename=f"Result_{safe_name}.pdf",
    )
