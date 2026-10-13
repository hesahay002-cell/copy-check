import os
import tempfile
import fitz  # PyMuPDF
from fastapi import FastAPI, Form, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from openai import OpenAI
import openpyxl
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Groq API Configuration (Free & Fast Vision Model)
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
    
    # Prepare images for Groq Llama Vision
    import base64
    image_contents = []
    for page in answer_pages:
        content = await page.read()
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
        completion = client.chat.completions.create(
            model="llama-3.2-90b-vision-preview",
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

    # Generate PDF Report Card
    pdf_path = tempfile.mktemp(suffix=".pdf")
    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("<b>Safal Academy Live</b>", styles['Title']))
    story.append(Paragraph("<b>Barganda, Giridih - Detailed Evaluation Report</b>", styles['Heading2']))
    story.append(Spacer(1, 10))
    story.append(Paragraph(f"<b>Student Name:</b> {student_name}", styles['Normal']))
    story.append(Paragraph(f"<b>Class:</b> {student_class} | <b>School:</b> {school_name}", styles['Normal']))
    story.append(Paragraph(f"<b>Test Code:</b> {test['test_code']} | <b>Subject:</b> {test['subject']}", styles['Normal']))
    story.append(Paragraph(f"<b>Evaluated By:</b> RAVISH SIR", styles['Normal']))
    story.append(Spacer(1, 15))
    story.append(Paragraph(f"<b>Marks Awarded:</b> {awarded_marks} / {test['total_marks']}", styles['Heading1']))
    story.append(Spacer(1, 10))
    story.append(Paragraph(<b>Evaluation Feedback & Cause of Deduction:</b>, styles['Heading3']))
    story.append(Paragraph(feedback.replace('\n', '<br/>'), styles['Normal']))
    story.append(Spacer(1, 20))
    story.append(Paragraph("<b>POWERED BY RAVISH SIR</b>", styles['Italic']))

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
    story = [Paragraph(<b>Safal Academy Live - Merit List</b>, styles['Title']), Spacer(1, 10)]
    
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
