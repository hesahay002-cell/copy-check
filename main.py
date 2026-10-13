import os
import json
import base64
import csv
import re
from io import StringIO
from datetime import datetime
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Response
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import pypdf
import shutil
import uuid

app = FastAPI(title="Safal AI Evaluator API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY", "YOUR_OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)

DB_FILE = "tests_db.json"
RESULTS_FILE = "results_db.json"

def load_tests():
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "r") as f: return json.load(f)
    return {}

def save_tests(data):
    with open(DB_FILE, "w") as f: json.dump(data, f, indent=4)

def load_results():
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "r") as f: return json.load(f)
    return {}

def save_results(data):
    with open(RESULTS_FILE, "w") as f: json.dump(data, f, indent=4)

def extract_text_from_pdf(pdf_path):
    try:
        text = ""
        with open(pdf_path, "rb") as f:
            reader = pypdf.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() + "\n"
        return text
    except Exception as e:
        return "Could not extract text from PDF."

@app.post("/teacher/login")
async def teacher_login(password: str = Form(...)):
    if password == "525077": return {"status": "success"}
    raise HTTPException(status_code=401, detail="Invalid Password")

@app.post("/teacher/upload")
async def upload_teacher_files(
    test_code: str = Form(...), subject: str = Form(...),
    target_class: str = Form(...), total_marks: float = Form(...),
    expiration: str = Form(...),
    sqp: UploadFile = File(...), ideal_answer: UploadFile = File(...)
):
    os.makedirs("uploads/teacher", exist_ok=True)
    test_id = str(uuid.uuid4())
    sqp_path = f"uploads/teacher/{test_id}_sqp.pdf"
    ideal_path = f"uploads/teacher/{test_id}_ideal.pdf"
    
    with open(sqp_path, "wb") as buffer: shutil.copyfileobj(sqp.file, buffer)
    with open(ideal_path, "wb") as buffer: shutil.copyfileobj(ideal_answer.file, buffer)
        
    tests = load_tests()
    tests[test_id] = {
        "id": test_id, "test_code": test_code.upper(), "subject": subject,
        "class": target_class, "total_marks": total_marks, "expiration": expiration,
        "sqp_path": sqp_path, "ideal_path": ideal_path, "created_at": datetime.now().isoformat()
    }
    save_tests(tests)
    return {"status": "success", "message": f"Test '{test_code}' published! Total Marks: {total_marks}"}

@app.get("/student/tests")
async def get_active_tests(student_class: str):
    tests = load_tests()
    active_tests = []
    now = datetime.now()
    for tid, test in tests.items():
        if test["class"] == student_class and now < datetime.fromisoformat(test["expiration"]):
            active_tests.append({
                "id": test["id"], "test_code": test.get("test_code", "N/A"), 
                "subject": test["subject"], "total_marks": test.get("total_marks", 50),
                "expiration": test["expiration"]
            })
    return {"tests": active_tests}

@app.get("/teacher/tests/{class_id}")
async def get_teacher_tests(class_id: str):
    tests = load_tests()
    filtered = [{"id": tid, "test_code": t.get("test_code", "N/A"), "subject": t["subject"]} for tid, t in tests.items() if t["class"] == class_id]
    return {"tests": filtered}

@app.post("/student/evaluate")
async def evaluate_student_copy(
    student_name: str = Form(...), student_email: str = Form(...),
    student_phone: str = Form(...), school_name: str = Form(...),
    student_class: str = Form(...), test_id: str = Form(...),
    answer_pages: list[UploadFile] = File(...)
):
    tests = load_tests()
    if test_id not in tests: raise HTTPException(status_code=404, detail="Test not found")
    test_data = tests[test_id]
    total_marks_limit = float(test_data.get("total_marks", 50.0))
    
    ideal_text = extract_text_from_pdf(test_data["ideal_path"])
    
    image_contents = []
    for img in answer_pages:
        img_bytes = await img.read()
        base64_img = base64.b64encode(img_bytes).decode('utf-8')
        image_contents.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}
        })
        
    prompt = f"""You are an expert, strict but fair examiner. Evaluate the student's handwritten answer images based on this IDEAL ANSWER MARKING SCHEME:
{ideal_text[:2000]}

RULES:
1. Maximum marks possible: {total_marks_limit}
2. Apply STEP MARKING. If steps are correct but the final answer is wrong, give partial marks.
3. Grade based on SENSE and CONCEPT. If the student's answer means the same thing as the ideal answer, award full marks. Do NOT deduct marks for not matching exact words.
4. Output MUST be ONLY valid JSON containing 'marks_awarded' (number) and 'feedback' (string explaining deductions or appreciating correct answers). Do not use markdown blocks.
Example JSON: {{"marks_awarded": 45.5, "feedback": "Good attempt. 4.5 marks deducted in Q2 for incorrect calculation at the final step."}}"""

    try:
        response = client.chat.completions.create(
            model="qwen/qwen-2-vl-72b-instruct",
            messages=[
                {"role": "system", "content": "You are a JSON-only evaluation bot."},
                {"role": "user", "content": [{"type": "text", "text": prompt}] + image_contents}
            ]
        )
        ai_response = response.choices[0].message.content.strip()
        
        if ai_response.startswith("```json"): ai_response = ai_response[7:-3]
        elif ai_response.startswith("```"): ai_response = ai_response[3:-3]
        
        result_json = json.loads(ai_response)
        obtained_marks = float(result_json.get("marks_awarded", 0))
        if obtained_marks > total_marks_limit: obtained_marks = total_marks_limit
        feedback = str(result_json.get("feedback", "Evaluated via AI."))
        
    except Exception as e:
        obtained_marks = 0.0
        feedback = "Error connecting to AI API or parsing response. Check your API key and balance."

    results_db = load_results()
    if test_id not in results_db: results_db[test_id] = []
    results_db[test_id].append({
        "student_name": student_name, "student_email": student_email,
        "student_phone": student_phone, "school_name": school_name,
        "student_class": student_class, "marks": obtained_marks,
        "total_marks": total_marks_limit, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")
    })
    save_results(results_db)
    
    os.makedirs("results", exist_ok=True)
    pdf_filename = f"{student_name.replace(' ', '_')}_{test_data.get('test_code', 'Result')}.pdf"
    pdf_path = os.path.join("results", pdf_filename)
    
    c = canvas.Canvas(pdf_path, pagesize=A4)
    width, height = A4
    
    c.setFillColorRGB(0.08, 0.16, 0.36)
    c.rect(0, height - 80, width, 80, stroke=0, fill=1)
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 24)
    c.drawString(40, height - 45, "Safal Academy Live")
    c.setFont("Helvetica", 12)
    c.drawString(40, height - 65, "Barganda, Giridih - Detailed Evaluation Report")
    
    c.setFillColorRGB(0.95, 0.95, 0.97)
    c.roundRect(40, height - 190, width - 80, 90, 8, stroke=1, fill=1)
    c.setFillColor(colors.black)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, height - 125, f"Student Name: {student_name}")
    c.setFont("Helvetica", 10)
    c.drawString(50, height - 145, f"Class: {student_class}  |  School: {school_name}")
    c.drawString(50, height - 165, f"Test Code: {test_data.get('test_code', 'N/A')}  |  Subject: {test_data['subject']}")
    
    c.setFont("Helvetica-Bold", 11)
    c.setFillColorRGB(0.8, 0.1, 0.1)
    c.drawString(width - 250, height - 125, "Evaluated By: RAVISH SIR")
    c.setFillColor(colors.black)
    c.setFont("Helvetica", 10)
    c.drawString(width - 250, height - 145, f"Marks Awarded: {obtained_marks} / {total_marks_limit}")
    c.drawString(width - 250, height - 165, f"Pages Scanned: {len(answer_pages)}")
    
    c.setFillColorRGB(0.85, 0.92, 1.0)
    c.rect(40, height - 230, width - 80, 25, stroke=0, fill=1)
    c.setFillColor(colors.black)
    c.setFont("Helvetica-Bold", 11)
    c.drawString(45, height - 223, "Evaluation Feedback & Cause of Deduction:")
    
    c.setFont("Helvetica", 10)
    from reportlab.lib.utils import simpleSplit
    lines = simpleSplit(feedback, "Helvetica", 10, width - 90)
    y_text = height - 250
    for line in lines:
        c.drawString(45, y_text, line)
        y_text -= 15
    
    c.setFillColorRGB(0.2, 0.2, 0.2)
    c.setFont("Helvetica-Bold", 12)
    c.drawString((width - c.stringWidth("POWERED BY RAVISH SIR", "Helvetica-Bold", 12)) / 2, 30, "POWERED BY RAVISH SIR")
    c.save()
    
    return FileResponse(pdf_path, media_type="application/pdf", filename=pdf_filename)


@app.get("/teacher/results/pdf/{test_id}")
async def download_merit_pdf(test_id: str):
    tests = load_tests()
    if test_id not in tests: raise HTTPException(404, "Test not found")
    test_data = tests[test_id]
    
    results = load_results().get(test_id, [])
    results = sorted(results, key=lambda x: x["marks"], reverse=True)
    
    pdf_filename = f"MeritList_{test_data.get('test_code', 'Test')}.pdf"
    pdf_path = os.path.join("results", pdf_filename)
    
    doc = SimpleDocTemplate(pdf_path, pagesize=A4)
    elements = []
    styles = getSampleStyleSheet()
    
    title_style = styles['Title']
    title_style.textColor = colors.HexColor("#081024")
    elements.append(Paragraph("<b>Safal Academy Live - Merit List</b>", title_style))
    elements.append(Paragraph(f"<b>Test Code:</b> {test_data.get('test_code', 'N/A')} &nbsp;&nbsp;|&nbsp;&nbsp; <b>Subject:</b> {test_data['subject']} &nbsp;&nbsp;|&nbsp;&nbsp; <b>Class:</b> {test_data['class']}", styles['Normal']))
    elements.append(Spacer(1, 20))
    
    data = [["Rank", "Student Name", "School", "Marks", "Total"]]
    for idx, r in enumerate(results):
        data.append([str(idx+1), r["student_name"], r["school_name"], str(r["marks"]), str(r["total_marks"])])
        
    t = Table(data, colWidths=[40, 150, 180, 60, 60])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#142850")),
        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0,0), (-1,0), 10),
        ('BACKGROUND', (0,1), (-1,-1), colors.HexColor("#f8f9fa")),
        ('GRID', (0,0), (-1,-1), 1, colors.HexColor("#dddddd")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#f0f4f8")])
    ]))
    elements.append(t)
    elements.append(Spacer(1, 40))
    elements.append(Paragraph("<b>POWERED BY RAVISH SIR</b>", styles['Normal']))
    
    doc.build(elements)
    return FileResponse(pdf_path, media_type="application/pdf", filename=pdf_filename)

@app.get("/teacher/results/excel/{test_id}")
async def download_merit_excel(test_id: str):
    tests = load_tests()
    if test_id not in tests: raise HTTPException(404, "Test not found")
    
    results = load_results().get(test_id, [])
    results = sorted(results, key=lambda x: x["marks"], reverse=True)
    
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(["Rank", "Student Name", "Email", "Phone Number", "School", "Class", "Marks Obtained", "Total Marks", "Submission Time"])
    
    for idx, r in enumerate(results):
        writer.writerow([idx+1, r.get("student_name",""), r.get("student_email",""), r.get("student_phone",""), r.get("school_name",""), r.get("student_class",""), r.get("marks",0), r.get("total_marks",0), r.get("timestamp","")])
    
    return Response(content=output.getvalue(), media_type="text/csv", headers={"Content-Disposition": f"attachment; filename=StudentData_{tests[test_id].get('test_code', 'Test')}.csv"})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
