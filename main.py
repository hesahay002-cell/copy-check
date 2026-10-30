import os
import io
import re
import json
import html
import uuid
import logging
import tempfile
from datetime import datetime, timezone
from typing import List, Optional

import fitz  # PyMuPDF: render teacher PDFs to images for Vision AI
import openpyxl
from PIL import Image, ImageOps
from fastapi import FastAPI, Form, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from openai import OpenAI

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, KeepTogether
)

# ================================================================
# SAFAL ACADEMY LIVE - VISION AI COPY EVALUATOR
#
# Render Build Command:
#   pip install -r requirements.txt
#
# Render Start Command:
#   uvicorn main:app --host 0.0.0.0 --port $PORT
#
# Render Environment Variables:
#   GROQ_API_KEY = your NEW Groq API key
#   GROQ_MODEL   = qwen/qwen3.8-27b
#   TEACHER_PIN  = optional; defaults to the existing PIN 525077
#
# IMPORTANT:
# - No OCR engine is used.
# - Qwen Vision reads teacher PDFs and student page images directly.
# - Qwen Vision accepts at most 3 images per request; batches are used.
# - API errors never become a fake 0-mark student PDF.
# - Tests/results are held in memory in this version. A Render restart
#   clears them; republish tests after a restart. Use a database for
#   permanent storage in a later upgrade.
# ================================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("safal-copy-check")

app = FastAPI(
    title="Safal Academy Live - Vision AI Copy Evaluator",
    version="4.0.0"
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
TEACHER_PIN = os.getenv("TEACHER_PIN", "525077")

AI_CLIENT = None
if GROQ_API_KEY:
    AI_CLIENT = OpenAI(
        api_key=GROQ_API_KEY,
        base_url="https://api.groq.com/openai/v1",
        timeout=240.0,
        max_retries=1,
    )

# Temporary in-memory storage. Render restarts clear this data.
TESTS_DB = {}
RESULTS_DB = {}

MAX_STUDENT_PAGES = 20
MAX_PDF_PAGES = 40
MAX_UPLOAD_BYTES = 18 * 1024 * 1024
VISION_IMAGES_PER_REQUEST = 3


# ------------------------------ COMMON HELPERS ------------------------------

def require_ai():
    if not GROQ_API_KEY or AI_CLIENT is None:
        raise HTTPException(
            status_code=503,
            detail="AI service is not configured. Please inform the teacher."
        )
    return AI_CLIENT


def safe_filename(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", value or "").strip("_")
    return cleaned[:60] or "Student"


def image_to_data_url(raw: bytes, max_side: int = 1500, quality: int = 76) -> str:
    """Normalize/rotate/compress a student or PDF-rendered page for Vision AI."""
    try:
        with Image.open(io.BytesIO(raw)) as original:
            image = ImageOps.exif_transpose(original).convert("RGB")
            image.thumbnail((max_side, max_side))
            output = io.BytesIO()
            image.save(output, format="JPEG", quality=quality, optimize=True)
            encoded = output.getvalue()
    except Exception as exc:
        raise ValueError(f"Image could not be opened: {exc}")

    if len(encoded) > MAX_UPLOAD_BYTES:
        raise ValueError("Image is too large after compression.")
    import base64
    return "data:image/jpeg;base64," + base64.b64encode(encoded).decode("ascii")


def pdf_pages_as_images(pdf_bytes: bytes, label: str):
    """Render every page of teacher's PDF; works with scanned PDFs too."""
    try:
        document = fitz.open(stream=pdf_bytes, filetype="pdf")
        if document.page_count < 1:
            raise ValueError("PDF has no pages.")
        if document.page_count > MAX_PDF_PAGES:
            raise ValueError(f"{label} PDF has more than {MAX_PDF_PAGES} pages.")

        pages = []
        for page_index in range(document.page_count):
            page = document.load_page(page_index)
            pix = page.get_pixmap(matrix=fitz.Matrix(1.55, 1.55), alpha=False)
            pages.append({
                "page_number": page_index + 1,
                "text": (page.get_text("text") or "").strip(),
                "image": image_to_data_url(pix.tobytes("png"), max_side=1500, quality=76),
            })
        document.close()
        return pages
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Could not read {label} PDF. Please upload a valid PDF. ({exc})"
        )


def call_vision(prompt: str, image_urls: list, max_tokens: int = 5000,
                json_mode: bool = False) -> str:
    """Call Groq Qwen Vision. Never convert provider errors into marks."""
    client = require_ai()
    content = [{"type": "text", "text": prompt}]
    for url in image_urls:
        content.append({
            "type": "image_url",
            "image_url": {"url": url}
        })

    kwargs = {
        "model": GROQ_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a careful visual reader and school examination assistant. "
                    "Read the supplied page images directly. Never pretend unreadable "
                    "handwriting is clear. Do not invent missing answers."
                )
            },
            {"role": "user", "content": content}
        ],
        "temperature": 0.1,
        "max_completion_tokens": max_tokens,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    response = client.chat.completions.create(**kwargs)
    answer = response.choices[0].message.content
    if not answer or not answer.strip():
        raise RuntimeError("Vision AI returned an empty response.")
    return answer.strip()


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

def read_teacher_pdf_pages(pages: list, pdf_kind: str) -> str:
    """
    Read Question Paper or Ideal Answer PDF page images directly with Vision AI.
    Native selectable text is included only as an additional aid, not as OCR.
    """
    all_notes = []
    for start in range(0, len(pages), VISION_IMAGES_PER_REQUEST):
        batch = pages[start:start + VISION_IMAGES_PER_REQUEST]
        image_urls = [p["image"] for p in batch]
        page_details = "\n".join(
            f"PDF page {p['page_number']} selectable text (if present):\n"
            f"{p['text'][:5000] if p['text'] else '[No selectable text; inspect image directly.]'}"
            for p in batch
        )

        if pdf_kind == "question":
            instruction = """
This is the TEACHER'S QUESTION PAPER. Read each page image carefully.
Transcribe every question and sub-question in original order. Preserve question
numbers, exact wording, marks printed beside each question, diagrams and labels,
chemical formulae, mathematical expressions, units and internal choices.
Do not skip a question just because it is near the page edge. If marks are not
printed for a question, write MARKS NOT SHOWN. Do not invent missing text.
"""
        else:
            instruction = """
This is the TEACHER'S IDEAL ANSWER / MARKING SCHEME PDF. Read each page image.
Transcribe answer numbers and match them to question numbers. Preserve expected
steps, formulae, equations, units, key terms, alternate accepted answers and
step-wise marks wherever shown. Do not replace the teacher's answer with your
own textbook answer. If an answer is not present or unreadable, say so.
"""

        prompt = f"""
{instruction}

The images in this request are teacher PDF pages:
{[p['page_number'] for p in batch]}

Additional selectable PDF text (may be incomplete; the image is authoritative):
{page_details}

Return a clear, faithful transcription with page numbers. Do not evaluate any
student and do not award marks.
"""
        notes = call_vision(prompt, image_urls, max_tokens=5000)
        all_notes.append(
            f"--- {pdf_kind.upper()} PDF PAGES {batch[0]['page_number']}-"
            f"{batch[-1]['page_number']} ---\n{notes}"
        )
    return "\n\n".join(all_notes)


# ------------------------ STUDENT COPY VISION READING -----------------------

def read_student_copy_pages(page_urls: list) -> str:
    """
    Vision-read every student page in batches of <=3.
    It deliberately does NOT assign marks yet; it captures every answer,
    including unnumbered answers and loose/inserted pages.
    """
    all_page_notes = []
    total_pages = len(page_urls)

    for start in range(0, total_pages, VISION_IMAGES_PER_REQUEST):
        batch_urls = page_urls[start:start + VISION_IMAGES_PER_REQUEST]
        first_page = start + 1
        last_page = start + len(batch_urls)

        prompt = f"""
Read these student answer-sheet page images directly. They are pages
{first_page} through {last_page} of a complete answer sheet containing
{total_pages} pages.

IMPORTANT READING RULES:
1. Inspect the ENTIRE visible page, including top/bottom margins, side margins,
   crossed-out work, inserted/loose slips, continuation lines and answers written
   below another question.
2. Transcribe all handwriting as faithfully as possible. Preserve equations,
   chemical formulae, units, calculations, diagrams, labels and crossed-out text.
3. Record the page number for every answer/answer fragment.
4. Do NOT assume the student wrote the correct question number. If an answer has
   no number or a possibly wrong number, transcribe its content and say "number
   unclear/unlabelled". The later evaluator will match by meaning and sequence.
5. Do NOT decide that a question is unanswered just because it is absent from
   these particular pages. The complete copy is being read in batches.
6. If handwriting is genuinely unreadable, mark only that portion as
   [unclear handwriting]. Do not guess.
7. Include duplicate/continued answers and loose slips. Do not discard them.
8. Do not award marks and do not judge correctness.

Return the transcription page by page. For each answer fragment include:
- page number
- any written question label (or unlabelled)
- exact readable student writing
- any unclear words/equations
- whether the answer appears continued on another page, if visible
"""
        notes = call_vision(prompt, batch_urls, max_tokens=6000)
        all_page_notes.append(
            f"===== STUDENT COPY: PAGES {first_page}-{last_page} =====\n{notes}"
        )

    return "\n\n".join(all_page_notes)



def audit_evaluation_against_original_images(
    questions: list, student_transcription: str, student_page_urls: list
) -> list:
    """
    Second-pass visual audit: check the evaluator's reading/legibility decisions
    against the actual student page images. This is especially important when
    a readable but scientifically wrong word was mistaken for illegible writing.
    """
    audited = [dict(item) for item in questions]
    if not student_page_urls:
        return audited

    for start in range(0, len(student_page_urls), VISION_IMAGES_PER_REQUEST):
        batch_urls = student_page_urls[start:start + VISION_IMAGES_PER_REQUEST]
        first_page = start + 1
        last_page = start + len(batch_urls)

        relevant = []
        for item in audited:
            page_numbers = {
                int(x) for x in re.findall(r"\d+", str(item.get("pages_found", "")))
            }
            if any(first_page <= n <= last_page for n in page_numbers):
                relevant.append(item)

        if not relevant:
            continue

        prompt = f"""
You are performing a STRICT VISUAL QUALITY-CONTROL AUDIT of an already graded
student answer sheet. The attached images are the ORIGINAL handwritten pages
{first_page} through {last_page}. Inspect the handwriting yourself; do not rely
only on the previous transcription.

PREVIOUS TRANSCRIPTION FOR CONTEXT:
{student_transcription[:18000]}

CURRENT QUESTION EVALUATIONS TO CHECK:
{json.dumps(relevant, ensure_ascii=False)[:14000]}

For each listed question whose answer appears on these attached pages:
1. Check the exact words in the original handwriting, especially words previously
   described as [unclear handwriting] or illegible.
2. If a word is readable, give the actual likely word. Never label a readable
   wrong answer as illegible. If it is genuinely unreadable, keep it unclear.
3. Check whether the scientific statement is correct. Do not confuse the colour
   of a solution with the colour of a deposit/precipitate.
4. Keep marks unchanged unless the visual evidence proves the existing marking
   was based on a misread word or factual mistake. Apply the teacher's ideal
   answer and the stated maximum marks.
5. If the image does not contain that question's answer, do not change it.
6. Do not invent letters or award marks for content that is not visible.

Return ONLY JSON:
{{
 "corrections": [
  {{
   "question_number": "2",
   "student_answer_correction": "Corrected faithful transcription, or empty if unchanged",
   "awarded_marks": 1.5,
   "deduction_reason": "Specific scientific error/missing point, not a legibility claim unless truly unreadable",
   "review_required": false,
   "review_reason": "",
   "visual_evidence": "Brief description of the visible word/line"
  }}
 ]
}}
If no changes are needed, return {{"corrections":[]}}.
"""
        raw = call_vision(prompt, batch_urls, max_tokens=3500, json_mode=True)
        audit_data = parse_json_response(raw)
        corrections = audit_data.get("corrections", [])
        if not isinstance(corrections, list):
            continue

        for correction in corrections:
            qnum = str(correction.get("question_number", "")).strip()
            target = next(
                (item for item in audited
                 if str(item.get("question_number", "")).strip() == qnum),
                None
            )
            if target is None:
                continue

            corrected_answer = str(
                correction.get("student_answer_correction", "")
            ).strip()
            if corrected_answer:
                target["student_answer"] = corrected_answer

            # Marks can only be changed within the question's printed range.
            if correction.get("awarded_marks") is not None:
                try:
                    revised = float(correction["awarded_marks"])
                    if 0 <= revised <= float(target["max_marks"]):
                        target["awarded_marks"] = round(revised, 2)
                except (TypeError, ValueError):
                    pass

            reason = str(correction.get("deduction_reason", "")).strip()
            if reason:
                target["deduction_reason"] = reason

            target["review_required"] = bool(
                correction.get("review_required", target["review_required"])
            )
            target["review_reason"] = str(
                correction.get("review_reason", target.get("review_reason", ""))
            ).strip()

    return audited


# -------------------------- QUESTION-WISE EVALUATION ------------------------

def evaluate_all_questions(test: dict, student_transcription: str, student_page_urls: list = None) -> dict:
    prompt = f"""
You are the lead examiner for Safal Academy Live, Giridih. Evaluate a student's
handwritten answer sheet. You must follow the teacher's two uploaded PDFs.

TEST DETAILS
Class: {test['target_class']}
Subject: {test['subject']}
Test Code: {test['test_code']}
Total test marks: {test['total_marks']}

A. TEACHER'S QUESTION PAPER PDF - VISION-READ TRANSCRIPTION
{test['question_reference'][:24000]}

B. TEACHER'S IDEAL ANSWER / MARKING SCHEME PDF - VISION-READ TRANSCRIPTION
{test['ideal_reference'][:24000]}

C. COMPLETE STUDENT COPY - EVERY UPLOADED PAGE WAS VISION-READ
{student_transcription[:30000]}

MANDATORY PROCESS:
1. Make a complete list of questions from the teacher's QUESTION PAPER. Preserve
   the original question numbers, subparts and order. Evaluate each question once.
2. Match student writing to questions by BOTH content/meaning and sequence.
   Do not rely only on the question number written by the student. Students may
   write a wrong number, omit a number, answer on a later page, continue an answer
   on another page, or use a loose/inserted slip.
3. Search the ENTIRE student transcription before saying "No answer found".
   Answers may be on any page, including later pages and loose slips. Combine
   answer fragments from multiple pages if they clearly belong to one answer.
4. Use the teacher's ideal-answer PDF as the primary marking reference. Do not
   invent a different ideal answer. If it does not contain an answer, say that.
5. For each question provide:
   - question_number: original teacher question number
   - question: full question text
   - student_answer: faithfully transcribed answer, with page number(s)
   - pages_found: page number(s) where answer was found, as a string
   - max_marks: allotted marks from teacher question paper/marking scheme
   - awarded_marks: marks awarded as a number, or null if handwriting prevents
     fair assessment
   - deduction_reason: specific missing/wrong step or "No marks deducted."
   - ideal_answer: teacher's expected answer/steps
   - review_required: true if answer is materially unclear, page matching is
     uncertain, teacher reference is unreadable, or marks cannot be fairly fixed
   - review_reason: concise reason, or empty string
6. Do not give zero merely because a question number is missing or wrong.
   First search all pages and match by answer content.
7. Do not give zero when handwriting is genuinely unclear. Set awarded_marks
   to null, review_required=true, and explain that teacher review is required.
8. Give zero only when the answer is genuinely absent after searching ALL pages,
   or the visible answer contains no creditworthy content. Explain the reason.
9. Partial marks must reflect correct visible points/steps. Do not double-penalize
   the same error. Accept scientifically/mathematically equivalent answers when
   consistent with the teacher's ideal answer.
10. CRITICAL: Separate handwriting legibility from answer correctness. If a word
    can reasonably be read, transcribe the most likely word and assess whether
    it is scientifically correct. Do NOT call a readable but incorrect word
    "illegible". Use [unclear handwriting] only when the actual letters cannot
    reasonably be distinguished from the page image/transcription.
11. For science answers, identify exactly WHAT the student says changes. For
    example, the blue copper sulphate SOLUTION turns green; reddish-brown copper
    DEPOSITS on the iron nail. Do not confuse the solution colour with the metal
    deposit colour. If the student writes that the solution turns brown, mark
    that scientific statement incorrect, not illegible.
12. If the transcription contains [unclear handwriting] but the surrounding
    letters suggest a likely word, do not guess silently. Set review_required=true
    only if the image audit also cannot read it; otherwise transcribe the word
    and mark its scientific correctness.
13. If a question's marks are not shown in the teacher's PDF, infer a fair
    allocation across questions only if needed to reach the test total, and
    clearly flag this in overall_feedback and review_required.
14. The sum of max_marks must not exceed the test total. If the paper's printed
    question marks conflict with the declared test total, preserve the printed
    marks and set overall_review_required=true; do not silently rescale.
15. Never invent a student answer, ideal answer, question, marks allocation,
    or page number.
16. The total awarded score is provisional if any question has null marks or
    review_required=true. The PDF must clearly flag teacher review.
17. Return ONLY a valid JSON object. No Markdown fences or extra commentary.

JSON shape:
{{
  "questions": [
    {{
      "question_number": "1",
      "question": "Full question text",
      "student_answer": "Student's actual answer or No answer found",
      "pages_found": "2, 4",
      "max_marks": 2,
      "awarded_marks": 1.5,
      "deduction_reason": "Specific reason for lost marks",
      "ideal_answer": "Teacher's ideal answer",
      "review_required": false,
      "review_reason": ""
    }}
  ],
  "overall_feedback": "Short useful feedback",
  "overall_review_required": false,
  "overall_review_reason": ""
}}
"""
    raw = call_vision(prompt, [], max_tokens=9000, json_mode=True)
    data = parse_json_response(raw)

    if not isinstance(data.get("questions"), list) or not data["questions"]:
        raise RuntimeError("AI did not return question-wise evaluation.")

    cleaned = []
    max_sum = 0.0
    awarded_sum = 0.0
    pending_count = 0

    for index, q in enumerate(data["questions"], start=1):
        try:
            max_marks = float(q.get("max_marks", 0))
            if max_marks < 0:
                raise ValueError()
        except (TypeError, ValueError):
            raise RuntimeError(f"Invalid maximum marks for question {index}.")

        awarded_raw = q.get("awarded_marks", 0)
        if awarded_raw is None:
            awarded = None
        else:
            try:
                awarded = float(awarded_raw)
            except (TypeError, ValueError):
                awarded = None

        if awarded is not None:
            awarded = max(0.0, min(awarded, max_marks))
            awarded_sum += awarded
        review_required = bool(q.get("review_required", False)) or awarded is None
        if review_required:
            pending_count += 1

        max_sum += max_marks
        cleaned.append({
            "question_number": str(q.get("question_number", index)),
            "question": str(q.get("question", "Question text not readable")),
            "student_answer": str(q.get("student_answer", "No answer found")),
            "pages_found": str(q.get("pages_found", "")),
            "max_marks": round(max_marks, 2),
            "awarded_marks": None if awarded is None else round(awarded, 2),
            "deduction_reason": str(q.get("deduction_reason", "")),
            "ideal_answer": str(q.get("ideal_answer", "Not found in teacher's ideal-answer PDF")),
            "review_required": review_required,
            "review_reason": str(q.get("review_reason", "")),
        })

    # Verify potentially questionable handwriting/legibility decisions against
    # the original page images before calculating the final score.
    cleaned = audit_evaluation_against_original_images(
        cleaned, student_transcription, student_page_urls or []
    )

    # Recalculate totals after the visual audit may have corrected a misread.
    awarded_sum = sum(
        float(item["awarded_marks"])
        for item in cleaned if item["awarded_marks"] is not None
    )
    pending_count = sum(1 for item in cleaned if item["review_required"])

    if max_sum > float(test["total_marks"]) + 0.01:
        data["overall_review_required"] = True
        data["overall_review_reason"] = (
            f"Question-wise maximum marks ({max_sum:g}) exceed the declared "
            f"test total ({float(test['total_marks']):g}). Teacher review required."
        )

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
    """Escape all student/AI text safely for ReportLab and convert basic markdown."""
    value = html.escape(str(value or ""))
    value = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", value, flags=re.DOTALL)
    value = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", value)
    return value.replace("\n", "<br/>")


def build_result_pdf(student_name: str, student_class: str, school_name: str,
                     test: dict, evaluation: dict) -> str:
    file_descriptor, pdf_path = tempfile.mkstemp(prefix="safal_result_", suffix=".pdf")
    os.close(file_descriptor)

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "SafalTitle", parent=styles["Title"], fontName="Helvetica-Bold",
        fontSize=18, leading=22, alignment=TA_CENTER,
        textColor=colors.white, spaceAfter=3
    )
    sub_style = ParagraphStyle(
        "SafalSubtitle", parent=styles["Normal"], fontSize=9,
        alignment=TA_CENTER, textColor=colors.white
    )
    heading_style = ParagraphStyle(
        "QHeading", parent=styles["Heading3"], fontSize=11,
        leading=14, textColor=colors.HexColor("#173B75"),
        spaceBefore=7, spaceAfter=5
    )
    body_style = ParagraphStyle(
        "SafalBody", parent=styles["BodyText"], fontSize=9,
        leading=13, spaceAfter=5, wordWrap="CJK"
    )
    small_style = ParagraphStyle(
        "SafalSmall", parent=body_style, fontSize=8, leading=11
    )

    story = []
    header = Table(
        [[Paragraph("SAFAL ACADEMY LIVE", title_style)],
         [Paragraph("Barganda, Giridih | Detailed AI Copy Evaluation", sub_style)]],
        colWidths=[510]
    )
    header.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#173B75")),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("TOPPADDING", (0, 0), (-1, -1), 9),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
    ]))
    story.append(header)
    story.append(Spacer(1, 12))

    details = [
        ["Student Name", student_name, "Class", student_class],
        ["School", school_name, "Evaluated By", "RAVISH SIR"],
        ["Test Code", test["test_code"], "Subject", test["subject"]],
    ]
    details_table = Table(details, colWidths=[75, 175, 75, 185])
    details_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#EEF2F7")),
        ("BACKGROUND", (2, 0), (2, -1), colors.HexColor("#EEF2F7")),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(details_table)
    story.append(Spacer(1, 10))

    total = float(test["total_marks"])
    provisional_score = float(evaluation["awarded_marks"])
    percentage = provisional_score / total * 100 if total else 0
    label = "PROVISIONAL MARKS" if evaluation["provisional"] else "TOTAL MARKS AWARDED"
    marks_style = ParagraphStyle(
        "ScoreStyle", parent=styles["Heading2"], fontSize=14,
        alignment=TA_CENTER, textColor=colors.HexColor("#166534")
    )
    score_table = Table(
        [[Paragraph(
            f"<b>{label}: {provisional_score:g} / {total:g} ({percentage:.1f}%)</b>",
            marks_style
        )]],
        colWidths=[510]
    )
    score_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F0FDF4")),
        ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#86B99A")),
        ("PADDING", (0, 0), (-1, -1), 10),
    ]))
    story.append(score_table)

    if evaluation["provisional"]:
        story.append(Spacer(1, 6))
        story.append(Paragraph(
            f"<b>TEACHER REVIEW REQUIRED:</b> {evaluation['pending_count']} "
            "question(s) need manual review because the handwriting, matching, "
            "or marking reference was unclear. The score above is provisional.",
            ParagraphStyle("ReviewAlert", parent=body_style,
                           textColor=colors.HexColor("#9A3412"))
        ))

    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "<b>Question-wise Evaluation</b>",
        ParagraphStyle("Section", parent=styles["Heading2"],
                       textColor=colors.HexColor("#173B75"))
    ))

    for item in evaluation["questions"]:
        awarded = "Pending teacher review" if item["awarded_marks"] is None else f"{item['awarded_marks']:g}"
        question_heading = (
            f"Question {html.escape(item['question_number'])} "
            f"({awarded} / {item['max_marks']:g} marks)"
        )
        review_line = ""
        if item["review_required"]:
            review_line = (
                "<br/><b><font color='#9A3412'>Teacher Review:</font></b> "
                + para_text(item["review_reason"] or "Please verify this answer manually.")
            )

        page_line = (
            f"<b>Page(s) Found:</b> {para_text(item['pages_found'])}"
            if item["pages_found"] else "<b>Page(s) Found:</b> Not identified"
        )

        blocks = [
            Paragraph(question_heading, heading_style),
            Paragraph(f"<b>Question:</b> {para_text(item['question'])}", body_style),
            Paragraph(f"<b>Student Answer:</b> {para_text(item['student_answer'])}", body_style),
            Paragraph(page_line, small_style),
            Paragraph(
                f"<b>Allotted Marks:</b> {item['max_marks']:g}"
                f"&nbsp;&nbsp;&nbsp; <b>Marks Awarded:</b> {awarded}",
                body_style
            ),
            Paragraph(
                f"<b>Cause of Deduction of Marks:</b> "
                f"{para_text(item['deduction_reason'] or 'No marks deducted.')}",
                body_style
            ),
            Paragraph(f"<b>Ideal Answer:</b> {para_text(item['ideal_answer'])}", body_style),
        ]
        if review_line:
            blocks.append(Paragraph(review_line, body_style))
        blocks.append(Spacer(1, 5))

        card = Table([[blocks]], colWidths=[510])
        card.setStyle(TableStyle([
            ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#CBD5E1")),
            ("BACKGROUND", (0, 0), (-1, -1), colors.white),
            ("LEFTPADDING", (0, 0), (-1, -1), 9),
            ("RIGHTPADDING", (0, 0), (-1, -1), 9),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]))
        story.append(card)
        story.append(Spacer(1, 7))

    story.append(Spacer(1, 8))
    story.append(Paragraph("<b>Overall Feedback:</b>", heading_style))
    story.append(Paragraph(para_text(evaluation["overall_feedback"]), body_style))

    if evaluation["overall_review_reason"]:
        story.append(Paragraph(
            f"<b>Overall Review Note:</b> {para_text(evaluation['overall_review_reason'])}",
            body_style
        ))

    story.append(Spacer(1, 14))
    story.append(Paragraph(
        "<b>POWERED BY RAVISH SIR</b>",
        ParagraphStyle("Footer", parent=styles["Normal"],
                       alignment=TA_CENTER, textColor=colors.HexColor("#B91C1C"))
    ))

    def add_page_number(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.grey)
        canvas.drawRightString(A4[0] - 35, 18, f"Page {doc.page}")
        canvas.restoreState()

    SimpleDocTemplate(
        pdf_path, pagesize=A4,
        rightMargin=42, leftMargin=42, topMargin=35, bottomMargin=35,
        title="Safal Academy Detailed Evaluation Report"
    ).build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)
    return pdf_path


# -------------------------------- API ROUTES --------------------------------

@app.get("/")
def home():
    return {
        "service": "Safal Academy Live AI Copy Evaluator",
        "status": "online",
        "version": "4.0.0",
        "docs": "/docs",
        "evaluation_route": "/student/evaluate",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "groq_key_configured": bool(GROQ_API_KEY),
        "model": GROQ_MODEL,
        "vision_images_per_request": VISION_IMAGES_PER_REQUEST,
        "max_student_pages": MAX_STUDENT_PAGES,
    }


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

    if len(question_pdf) > 25 * 1024 * 1024 or len(ideal_pdf) > 25 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Each teacher PDF must be 25 MB or smaller.")

    question_pages = pdf_pages_as_images(question_pdf, "Question Paper")
    ideal_pages = pdf_pages_as_images(ideal_pdf, "Ideal Answer")

    test_id = f"{target_class}_{test_code}".strip()
    TESTS_DB[test_id] = {
        "id": test_id,
        "test_code": test_code,
        "subject": subject,
        "target_class": target_class,
        "total_marks": float(total_marks),
        "expiration": expiration,
        "question_pages": question_pages,
        "ideal_pages": ideal_pages,
        "question_reference": None,
        "ideal_reference": None,
        "reference_ready": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    RESULTS_DB[test_id] = []

    logger.info("Teacher published test %s with %s question PDF pages and %s ideal PDF pages",
                test_id, len(question_pages), len(ideal_pages))

    return {
        "status": "published",
        "test_id": test_id,
        "message": "Question Paper PDF and Ideal Answer PDF uploaded successfully.",
    }


@app.get("/teacher/tests/{target_class}")
def get_teacher_tests(target_class: str):
    return {
        "tests": [
            {
                "id": test["id"],
                "test_code": test["test_code"],
                "subject": test["subject"],
            }
            for test in TESTS_DB.values()
            if test["target_class"] == target_class
        ]
    }


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
            detail="This test is not available. Teacher must publish the test again."
        )

    if not answer_pages:
        raise HTTPException(status_code=400, detail="Please upload answer-sheet pages.")
    if len(answer_pages) > MAX_STUDENT_PAGES:
        raise HTTPException(
            status_code=400,
            detail=f"Maximum {MAX_STUDENT_PAGES} answer pages are allowed."
        )

    test = TESTS_DB[test_id]

    try:
        # 1. Read both teacher PDFs directly with Vision AI (once per published test).
        if not test["reference_ready"]:
            logger.info("Vision-reading teacher references for test %s", test_id)
            question_reference = read_teacher_pdf_pages(test["question_pages"], "question")
            ideal_reference = read_teacher_pdf_pages(test["ideal_pages"], "ideal answer")

            if not question_reference.strip() or not ideal_reference.strip():
                raise RuntimeError("Could not read both teacher reference PDFs.")

            test["question_reference"] = question_reference
            test["ideal_reference"] = ideal_reference
            test["reference_ready"] = True

        # 2. Prepare all student pages in their original upload order.
        student_page_urls = []
        for index, page in enumerate(answer_pages, start=1):
            raw = await page.read()
            if not raw:
                raise HTTPException(
                    status_code=400,
                    detail=f"Student page {index} is empty."
                )
            try:
                student_page_urls.append(image_to_data_url(raw))
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=f"Page {index}: {exc}")

        # 3. Vision-read EVERY student page, in groups of max 3 images.
        logger.info("Vision-reading %s student pages for %s", len(student_page_urls), student_name)
        student_transcription = read_student_copy_pages(student_page_urls)

        # 4. Match all answer fragments across the complete copy, then evaluate.
        evaluation = evaluate_all_questions(test, student_transcription, student_page_urls)

        # 5. Build PDF only after successful AI evaluation.
        pdf_path = build_result_pdf(
            student_name=student_name,
            student_class=student_class,
            school_name=school_name,
            test=test,
            evaluation=evaluation,
        )

    except HTTPException:
        raise
    except Exception as exc:
        # Do not make an error PDF or save a fake zero score.
        logger.exception("Evaluation failed for test %s / student %s", test_id, student_name)
        error_text = str(exc).lower()

        if "401" in error_text or "invalid_api_key" in error_text:
            detail = "Evaluation service authentication failed. Please inform the teacher."
        elif "429" in error_text or "rate limit" in error_text:
            detail = "AI service is busy or rate-limited. Please try again later."
        elif "model" in error_text and ("not found" in error_text or "decommission" in error_text):
            detail = "AI model configuration needs attention. Please inform the teacher."
        elif "413" in error_text or "too large" in error_text:
            detail = "Some uploaded pages are too large. Please upload clearer, smaller images."
        else:
            detail = "Evaluation could not be completed. No marks have been recorded. Please try again."
        raise HTTPException(status_code=502, detail=detail)

    # Record result only after evaluation and PDF generation succeeded.
    result_record = {
        "name": student_name,
        "email": student_email,
        "phone": student_phone,
        "school": school_name,
        "marks": evaluation["awarded_marks"],
        "feedback": evaluation["overall_feedback"],
        "total": test["total_marks"],
        "provisional": evaluation["provisional"],
        "pending_review": evaluation["pending_count"],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    RESULTS_DB.setdefault(test_id, []).append(result_record)

    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        filename=f"Result_{safe_filename(student_name)}.pdf",
        background=None,
    )


@app.get("/teacher/results/pdf/{test_id}")
def download_results_pdf(test_id: str):
    if test_id not in TESTS_DB:
        raise HTTPException(status_code=404, detail="Test not found.")

    results = sorted(
        RESULTS_DB.get(test_id, []),
        key=lambda item: item["marks"],
        reverse=True
    )

    fd, path = tempfile.mkstemp(prefix="safal_merit_", suffix=".pdf")
    os.close(fd)
    styles = getSampleStyleSheet()
    story = [
        Paragraph("<b>SAFAL ACADEMY LIVE - MERIT LIST</b>", styles["Title"]),
        Spacer(1, 12),
    ]

    rows = [["Rank", "Student", "School", "Marks", "Status"]]
    for rank, result in enumerate(results, start=1):
        status = "Provisional" if result.get("provisional") else "Evaluated"
        rows.append([
            str(rank), result["name"], result["school"],
            f"{result['marks']:g} / {result['total']:g}", status
        ])

    table = Table(rows, repeatRows=1, colWidths=[40, 130, 130, 100, 80])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#173B75")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(table)
    SimpleDocTemplate(path, pagesize=A4).build(story)
    return FileResponse(path, media_type="application/pdf", filename="Merit_List.pdf")


@app.get("/teacher/results/excel/{test_id}")
def download_results_excel(test_id: str):
    if test_id not in TESTS_DB:
        raise HTTPException(status_code=404, detail="Test not found.")

    results = sorted(
        RESULTS_DB.get(test_id, []),
        key=lambda item: item["marks"],
        reverse=True
    )
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "Merit List"
    sheet.append([
        "Rank", "Student Name", "Email", "Phone", "School",
        "Marks Obtained", "Total Marks", "Evaluation Status", "Pending Review"
    ])

    for rank, result in enumerate(results, start=1):
        sheet.append([
            rank, result["name"], result["email"], result["phone"],
            result["school"], result["marks"], result["total"],
            "Provisional" if result.get("provisional") else "Evaluated",
            result.get("pending_review", 0),
        ])

    fd, path = tempfile.mkstemp(prefix="safal_merit_", suffix=".xlsx")
    os.close(fd)
    workbook.save(path)
    return FileResponse(
        path,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename="Merit_List.xlsx"
    )
