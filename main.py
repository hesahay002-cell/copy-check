import os
import tempfile
import io
import base64
from PIL import Image
from fastapi import FastAPI, Form, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from openai import OpenAI
import openpyxl

# Colorful PDF Libraries
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Groq API Configuration (Using Active Qwen Vision Model)
os.environ["GROQ_API_KEY"] = "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"
client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ.get("GROQ_API_KEY"),
)

# In-memory storage for deployment testing
TESTS_DB = {}
RESULTS_DB = {}

@app.post("/teacher/login")
def teacher_login(password: str = Form(...)):
    if password == "525077":
        return {"status": "success"}
    raise HTTPException(status_code=401, detail="Invalid PIN")

@app.post("/teacher/upload")
async def upload_test(
    test_code: str = Form(...),
    subject: str = Form(...),
    target_class: str = Form(...),
    total_marks: float = Form(...),
    expiration: str = Form(None),
    sqp: UploadFile = File(...),
    ideal_answer: UploadFile = File(...)
):
    test_id = f"{target_class}_{test_code}"
    sqp_bytes = await sqp.read()
    ideal_bytes = await ideal_answer.read()
    
    TESTS_DB[test_id] = {
        "id": test_id,
        "test_code": test_code,
        "subject": subject,
        "target_class": target_class,
        "total_marks": total_marks,
        "expiration": expiration,
        "sqp_bytes": sqp_bytes,
        "ideal_bytes": ideal_bytes
    }
    return {"status": "published", "test_id": test_id}

@app.get("/student/tests")
def get_student_tests(student_class: str):
    active_tests = []
    for tid, t in TESTS_DB.items():
        if t["target_class"] == student_class:
            active_tests.append({
                "id": t["id"],
                "test_code": t["test_code"],
                "subject": t["subject"],
                "total_marks": t["total_marks"]
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
    answer_pages: list[UploadFile] = File(...)
):
    if test_id not in TESTS_DB:
        raise HTTPException(status_code=404, detail="Test not found")
    
    test = TESTS_DB[test_id]
    
    # Image Compression and Preparation for Groq API
    image_contents = []
    for page in answer_pages:
        content = await page.read()
        try:
            # Compress using Pillow to avoid Size Errors
            img = Image.open(io.BytesIO(content))
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img.thumbnail((1200, 1200))
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=65)
            encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
        except Exception:
            encoded = base64.b64encode(content).decode("utf-8")

        image_contents.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{encoded}"
            }
        })

    prompt = f"""
    You are an expert strict examiner for Safal Academy, Giridih. Evaluate the attached student answer sheets for Test Code: {test['test_code']}, Subject: {test['subject']}.
    Total Full Marks: {test['total_marks']}.
    Analyze the handwriting step-by-step. Provide marks awarded out of {test['total_marks']} and detailed feedback/deduction reasons.
    Format your response cleanly:
    MARKS: [Awarded Marks Number only, e.g. 45]
    FEEDBACK: [Detailed evaluation breakdown and feedback]
    """

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                *image_contents
            ]
        }
    ]

    try:
        # Updated to the new active Qwen Multimodal Model on Groq
        completion = client.chat.completions.create(
            model="qwen/qwen3.8-27b",
            messages=messages,
            max_tokens=2000
        )
        ai_reply = completion.choices[0].message.content
    except Exception as e:
        ai_reply = f"MARKS: 0\nFEEDBACK: Error connecting to AI API or parsing response. Details: {str(e)}"

    # Parse Marks & Feedback
    awarded_marks = 0.0
    feedback = ai_reply
    try:
        for line in ai_reply.split('\n'):
            if "MARKS:" in line:
                import re
                nums = re.findall(r"[-+]?\d*\.\d+|\d+", line)
                if nums:
                    awarded_marks = float(nums[0])
    except Exception:
        pass

    # Save Result in DB
    if test_id not in RESULTS_DB:
        RESULTS_DB[test_id] = []
    
    result_record = {
        "name": student_name,
        "email": student_email,
        "phone": student_phone,
        "school": school_name,
        "marks": awarded_marks,
        "feedback": feedback,
        "total": test['total_marks']
    }
    RESULTS_DB[test_id].append(result_record)

    # -----------------------------------------------------
    # Generate Colorful Professional PDF Report Card
    # -----------------------------------------------------
    pdf_path = tempfile.mktemp(suffix=".pdf")
    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    styles = getSampleStyleSheet()
    
    # Custom PDF Styles
    title_style = ParagraphStyle(
        name='MainTitle', parent=styles['Heading1'],
        fontSize=20, textColor=colors.HexColor("#FFFFFF"),
        alignment=1, spaceAfter=8
    )
    subtitle_style = ParagraphStyle(
        name='SubTitle', parent=styles['Heading3'],
        fontSize=12, textColor=colors.HexColor("#FFFFFF"),
        alignment=1, spaceAfter=10
    )
    feedback_style = ParagraphStyle(
        name='FeedbackStyle', parent=styles['Normal'],
        fontSize=11, leading=16, textColor=colors.HexColor("#1F2937")
    )
    footer_style = ParagraphStyle(
        name='FooterStyle', parent=styles['Italic'],
        fontSize=12, alignment=1, textColor=colors.HexColor("#DC2626"), spaceBefore=30
    )

    story = []

    # 1. Header Banner (Deep Blue Background)
    header_data = [
        [Paragraph("<b>SAFAL ACADEMY LIVE</b>", title_style)],
        [Paragraph("<b>Barganda, Giridih - Detailed Evaluation Report</b>", subtitle_style)]
    ]
    header_table = Table(header_data, colWidths=[500])
    header_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#1E3A8A")), 
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ('TOPPADDING', (0, 0), (-1, -1), 10),
    ]))
    story.append(header_table)
    story.append(Spacer(1, 20))

    # 2. Student Details Section (Bordered Table)
    details_data = [
        ["Student Name:", student_name, "Class:", student_class],
        ["School Name:", school_name, "Evaluated By:", "RAVISH SIR"],
        ["Test Code:", test['test_code'], "Subject:", test['subject']]
    ]
    details_table = Table(details_data, colWidths=[100, 150, 90, 160])
    details_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor("#F3F4F6")),
        ('BACKGROUND', (2, 0), (2, -1), colors.HexColor("#F3F4F6")),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor("#111827")),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (2, 0), (2, -1), 'Helvetica-Bold'),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor("#D1D5DB")),
        ('PADDING', (0, 0), (-1, -1), 10),
    ]))
    story.append(details_table)
    story.append(Spacer(1, 20))

    # 3. Marks Awarded Highlight (Green if passed, Red if failed)
    marks_percentage = (awarded_marks / test['total_marks']) * 100 if test['total_marks'] > 0 else 0
    marks_color = colors.HexColor("#16A34A") if marks_percentage >= 40 else colors.HexColor("#DC2626") 
    bg_color = colors.HexColor("#ECFDF5") if marks_percentage >= 40 else colors.HexColor("#FEF2F2")

    marks_data = [[Paragraph(f"<b>TOTAL MARKS AWARDED: {awarded_marks} / {test['total_marks']}</b>", ParagraphStyle(name='marks', fontSize=15, textColor=marks_color, alignment=1))]]
    marks_table = Table(marks_data, colWidths=[500])
    marks_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), bg_color),
        ('BORDER', (0, 0), (-1, -1), 2, marks_color),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('PADDING', (0, 0), (-1, -1), 15),
    ]))
    story.append(marks_table)
    story.append(Spacer(1, 20))

    # 4. Detailed Feedback Section
    story.append(Paragraph("<b><font color='#2563EB'>Detailed Evaluation Feedback & Deductions:</font></b>", styles['Heading3']))
    story.append(Spacer(1, 5))
    clean_feedback = feedback.replace('\n', '<br/>')
    story.append(Paragraph(clean_feedback, feedback_style))
    
    # 5. Footer
    story.append(Spacer(1, 20))
    story.append(Paragraph("<b>POWERED BY RAVISH SIR</b>", footer_style))

    doc.build(story)

    return FileResponse(pdf_path, media_type='application/pdf', filename=f"Result_{student_name}.pdf")

@app.get("/teacher/tests/{target_class}")
def get_teacher_tests(target_class: str):
    tests = []
    for tid, t in TESTS_DB.items():
        if t["target_class"] == target_class:
            tests.append({"id": tid, "test_code": t["test_code"], "subject": t["subject"]})
    return {"tests": tests}

@app.get("/teacher/results/pdf/{test_id}")
def download_results_pdf(test_id: str):
    if test_id not in TESTS_DB:
        raise HTTPException(status_code=404, detail="Test not found")
    
    results = RESULTS_DB.get(test_id, [])
    pdf_path = tempfile.mktemp(suffix=".pdf")
    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    styles = getSampleStyleSheet()
    story = [Paragraph("<b>Safal Academy Live - Merit List</b>", styles['Title']), Spacer(1, 10)]
    
    for r in results:
        story.append(Paragraph(f"<b>{r['name']}</b> ({r['school']}) - Marks: {r['marks']} / {r['total']}", styles['Normal']))
        story.append(Spacer(1, 5))
        
    doc.build(story)
    return FileResponse(pdf_path, media_type='application/pdf', filename="Merit_List.pdf")

@app.get("/teacher/results/excel/{test_id}")
def download_results_excel(test_id: str):
    results = RESULTS_DB.get(test_id, [])
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Merit List"
    ws.append(["Student Name", "Email", "Phone", "School", "Marks Obtained", "Total Marks"])
    
    for r in results:
        ws.append([r["name"], r["email"], r["phone"], r["school"], r["marks"], r["total"]])
        
    excel_path = tempfile.mktemp(suffix=".xlsx")
    wb.save(excel_path)
    return FileResponse(excel_path, media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', filename="Merit_List.xlsx")
