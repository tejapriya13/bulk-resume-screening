import os
import json
import re
import shutil
from pathlib import Path
from typing import List

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from pypdf import PdfReader
from docx import Document
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / "uploads"
UPLOAD_FOLDER.mkdir(exist_ok=True)

SKILLS_FILE = BASE_DIR / "skills.json"
if not SKILLS_FILE.exists():
    # Keep the same skill vocabulary supplied with the original project.
    SKILLS_FILE = BASE_DIR / "skills(5).json"

with open(SKILLS_FILE, "r", encoding="utf-8") as file:
    SKILLS_DATABASE = json.load(file)

print("Loading AI Model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("AI Model Loaded Successfully!")

app = FastAPI(title="AI Resume Screening System")


def extract_pdf_text(file_path):
    reader = PdfReader(str(file_path))
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text


def extract_docx_text(file_path):
    document = Document(str(file_path))
    return "\n".join(p.text for p in document.paragraphs) + "\n"


def extract_resume_text(file_path):
    file_path = str(file_path)
    if file_path.lower().endswith(".pdf"):
        return extract_pdf_text(file_path)
    if file_path.lower().endswith(".docx"):
        return extract_docx_text(file_path)
    raise ValueError("Only PDF and DOCX files are supported")


def extract_email(text):
    emails = re.findall(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)
    return emails[0] if emails else "Not Found"


def extract_phone(text):
    phones = re.findall(r"\+?\d[\d\s\-]{8,15}", text)
    return phones[0].strip() if phones else "Not Found"


def extract_name(text):
    lines = text.split("\n")
    for line in lines[:10]:
        line = line.strip()
        if line and len(line.split()) <= 5 and not any(c.isdigit() for c in line) and "@" not in line:
            return line
    return "Unknown Candidate"


def extract_education(text):
    keywords = [
        "b.tech", "btech", "m.tech", "mtech", "bachelor", "master",
        "bca", "mca", "b.sc", "m.sc", "phd", "degree"
    ]
    education = []
    for line in text.split("\n"):
        lower_line = line.lower()
        if any(k in lower_line for k in keywords):
            education.append(line.strip())
    return education[:10]


def extract_experience(text):
    keywords = [
        "experience", "internship", "intern", "developer", "engineer",
        "analyst", "software", "manager"
    ]
    experience = []
    for line in text.split("\n"):
        lower_line = line.lower()
        if any(k in lower_line for k in keywords) and len(line.strip()) > 10:
            experience.append(line.strip())
    return experience[:10]


def extract_skills(text):
    text = text.lower()
    return list({skill for skill in SKILLS_DATABASE if skill.lower() in text})


def semantic_match(resume_text, job_description):
    embeddings = model.encode([resume_text, job_description])
    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    return max(0, min(round(float(similarity * 100), 2), 100))


def tfidf_match(resume_text, job_description):
    try:
        vectorizer = TfidfVectorizer(stop_words="english")
        matrix = vectorizer.fit_transform([resume_text, job_description])
        similarity = cosine_similarity(matrix[0], matrix[1])[0][0]
        return round(float(similarity * 100), 2)
    except Exception:
        return 0.0


def analyze_ats(resume_text, resume_skills, job_skills):
    score = 0
    suggestions = []
    word_count = len(resume_text.split())

    if word_count >= 150:
        score += 20
    else:
        suggestions.append("Add more professional information to your resume.")

    if "@" in resume_text:
        score += 10
    else:
        suggestions.append("Add a professional email address.")

    if len(resume_skills) >= 5:
        score += 20
    else:
        suggestions.append("Add more relevant technical skills.")

    matched_skills = list(set(resume_skills).intersection(set(job_skills)))
    if job_skills:
        score += (len(matched_skills) / len(job_skills)) * 40
    else:
        score += 20

    sections = ["education", "skills", "experience"]
    lower_text = resume_text.lower()
    found_sections = sum(1 for section in sections if section in lower_text)
    score += (found_sections / len(sections)) * 10

    return {
        "ats_score": round(min(score, 100), 2),
        "suggestions": suggestions,
        "matched_skills": matched_skills,
    }


def skill_gap_analysis(resume_skills, job_skills):
    resume_set = {skill.lower() for skill in resume_skills}
    job_set = {skill.lower() for skill in job_skills}
    matched_skills = list(resume_set.intersection(job_set))
    missing_skills = list(job_set.difference(resume_set))
    recommendations = [
        f"Consider learning {skill.title()} or completing a certification."
        for skill in missing_skills
    ]
    return {
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "recommendations": recommendations,
    }


def calculate_final_score(semantic_score, ats_score, skill_match_score):
    semantic_weight = 0.50
    ats_weight = 0.25
    skill_weight = 0.25

    semantic_contribution = semantic_score * semantic_weight
    ats_contribution = ats_score * ats_weight
    skill_contribution = skill_match_score * skill_weight
    final_score = semantic_contribution + ats_contribution + skill_contribution

    explanation = {
        "semantic_matching_contribution": round(semantic_contribution, 2),
        "ats_contribution": round(ats_contribution, 2),
        "skill_match_contribution": round(skill_contribution, 2),
        "final_score": round(final_score, 2),
    }
    return round(final_score, 2), explanation


def generate_strengths(semantic_score, matched_skills):
    strengths = []
    if semantic_score >= 70:
        strengths.append("Strong semantic similarity with the job description.")
    if len(matched_skills) >= 5:
        strengths.append("Candidate has strong technical skill alignment.")
    if len(matched_skills) >= 3:
        strengths.append("Candidate possesses several required job skills.")
    if not strengths:
        strengths.append("Candidate has some relevant qualifications.")
    return strengths


def generate_weaknesses(missing_skills, ats_score):
    weaknesses = []
    if missing_skills:
        weaknesses.append("Some important skills from the job description are missing.")
    if ats_score < 70:
        weaknesses.append("Resume requires improvements for better ATS compatibility.")
    if not weaknesses:
        weaknesses.append("No major weaknesses detected.")
    return weaknesses


def find_related_skills(resume_text, resume_skills, top_n=8):
    if not resume_skills:
        return []
    candidates = [
        s for s in SKILLS_DATABASE
        if s.lower() not in {x.lower() for x in resume_skills}
    ]
    if not candidates:
        return []
    resume_embedding = model.encode(resume_text)
    skill_embeddings = model.encode(candidates)
    scores = cosine_similarity([resume_embedding], skill_embeddings)[0]
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [
        {"skill": skill, "similarity": round(float(score) * 100, 2)}
        for skill, score in ranked[:top_n] if score >= 0.30
    ]


def generate_resume_strengths(resume_text, resume_skills, education, experience):
    strengths = []
    if resume_skills:
        strengths.append(
            f"Demonstrates knowledge of {len(resume_skills)} recognized technical skill(s): "
            + ", ".join(resume_skills[:6]) + "."
        )
    if len(resume_skills) >= 5:
        strengths.append("Shows a broad technical skill set across multiple technologies.")
    if education:
        strengths.append("Includes educational qualifications relevant to the candidate profile.")
    if experience:
        strengths.append("Contains experience or internship information that can support practical skills.")
    if not strengths:
        strengths.append("The resume contains limited structured information for identifying strengths.")
    return strengths[:5]


def generate_resume_weaknesses(resume_text, resume_skills, education, experience):
    weaknesses = []
    lower = resume_text.lower()
    if len(resume_skills) < 3:
        weaknesses.append("The resume lists only a small number of recognized technical skills.")
    if not education:
        weaknesses.append("No clear education qualification was detected.")
    if not experience:
        weaknesses.append("No clear work experience or internship details were detected.")
    if "project" not in lower and "projects" not in lower:
        weaknesses.append("No clearly labelled project section was detected.")
    if "certification" not in lower and "certifications" not in lower:
        weaknesses.append("No certification information was detected.")
    if not weaknesses:
        weaknesses.append("No major structural weakness was detected from the available resume content.")
    return weaknesses[:5]


def generate_resume_improvements(
    resume_text, resume_skills, education, experience,
    missing_skills, related_skills
):
    suggestions = []
    if missing_skills:
        suggestions.append("Prioritize the missing job skills: " + ", ".join(missing_skills[:6]) + ".")
    if related_skills:
        suggestions.append(
            "Consider strengthening related skills such as "
            + ", ".join(x["skill"] for x in related_skills[:4]) + "."
        )
    if not experience:
        suggestions.append("Add specific internship, work, or practical experience details.")
    if "project" not in resume_text.lower() and "projects" not in resume_text.lower():
        suggestions.append("Add projects with the technology used and the result achieved.")
    if not education:
        suggestions.append("Add clear education details such as degree, institution, and year.")
    if len(resume_skills) < 5:
        suggestions.append("Add relevant technical skills that are genuinely supported by your experience.")
    return suggestions[:6]


def generate_interview_questions(skills, experience):
    questions = [f"Can you explain a project where you used {skill}?" for skill in skills[:5]]
    if experience:
        questions.append("Describe a challenging problem you solved during your experience.")
    questions.append("Why do you believe you are suitable for this position?")
    questions.append("How do you approach solving a complex technical problem?")
    return questions


def build_candidate_overall_view(final_score, semantic_score, ats_score, matched_skills, missing_skills):
    """Human-readable explanation; intentionally contains no percentages."""
    fit = (
        "The resume shows strong alignment with the role."
        if final_score >= 70
        else "The resume shows moderate alignment with the role, with some areas that could be strengthened."
        if final_score >= 50
        else "The resume shows limited alignment with the role and would benefit from targeted improvements."
    )
    semantic_text = (
        "The candidate's experience and wording are closely related to the job requirements."
        if semantic_score >= 70
        else "The candidate's background has some meaningful overlap with the job requirements."
        if semantic_score >= 50
        else "The candidate's background has limited direct overlap with the job requirements."
    )
    ats_text = (
        "The resume structure and searchable content are generally suitable for an ATS."
        if ats_score >= 70
        else "The resume has some ATS-friendly content, but its structure or searchable information could be improved."
    )
    skill_text = (
        "Several required skills are represented in the resume."
        if matched_skills
        else "Few or none of the explicitly required skills are represented in the resume."
    )
    gap_text = (
        "The main areas to review are: " + ", ".join(missing_skills[:6]) + "."
        if missing_skills
        else "No explicit job-skill gaps were detected from the supplied skill list."
    )
    return " ".join([fit, semantic_text, ats_text, skill_text, gap_text])


def analyze_candidate(file_path, filename, job_description):
    resume_text = extract_resume_text(file_path)
    if not resume_text.strip():
        raise ValueError("Could not extract text from this resume.")

    candidate_name = extract_name(resume_text)
    email = extract_email(resume_text)
    phone = extract_phone(resume_text)
    education = extract_education(resume_text)
    experience = extract_experience(resume_text)

    resume_skills = extract_skills(resume_text)
    job_skills = extract_skills(job_description)

    semantic_score = semantic_match(resume_text, job_description)
    tfidf_score = tfidf_match(resume_text, job_description)
    ats_result = analyze_ats(resume_text, resume_skills, job_skills)
    skill_gap = skill_gap_analysis(resume_skills, job_skills)

    skill_match_score = (
        (len(skill_gap["matched_skills"]) / len(job_skills)) * 100
        if job_skills else 50
    )

    final_score, explanation = calculate_final_score(
        semantic_score,
        ats_result["ats_score"],
        skill_match_score,
    )

    related_skills = find_related_skills(resume_text, resume_skills, top_n=8)
    resume_strengths = generate_resume_strengths(
        resume_text, resume_skills, education, experience
    )
    resume_weaknesses = generate_resume_weaknesses(
        resume_text, resume_skills, education, experience
    )
    resume_improvements = generate_resume_improvements(
        resume_text, resume_skills, education, experience,
        skill_gap["missing_skills"], related_skills
    )

    strengths = resume_strengths + generate_strengths(
        semantic_score, skill_gap["matched_skills"]
    )
    weaknesses = resume_weaknesses + generate_weaknesses(
        skill_gap["missing_skills"], ats_result["ats_score"]
    )
    interview_questions = generate_interview_questions(resume_skills, experience)

    overall_view = build_candidate_overall_view(
        final_score,
        semantic_score,
        ats_result["ats_score"],
        skill_gap["matched_skills"],
        skill_gap["missing_skills"],
    )

    return {
        "file_name": filename,
        "candidate_name": candidate_name,
        "email": email,
        "phone": phone,
        "education": education,
        "experience": experience,
        "skills": resume_skills,
        "resume_matched_skills": resume_skills,
        "semantic_match_score": semantic_score,
        "tfidf_match_score": tfidf_score,
        "ats_score": ats_result["ats_score"],
        "skill_match_score": round(skill_match_score, 2),
        "final_candidate_score": final_score,
        "matched_skills": skill_gap["matched_skills"],
        "missing_skills": skill_gap["missing_skills"],
        "related_skills": related_skills,
        "resume_missing_skills": [x["skill"] for x in related_skills],
        "strengths": strengths,
        "resume_strengths": resume_strengths,
        "weaknesses": weaknesses,
        "resume_weaknesses": resume_weaknesses,
        "improvement_suggestions": (
            resume_improvements
            + ats_result["suggestions"]
            + skill_gap["recommendations"]
        ),
        "interview_questions": interview_questions,
        "explanation": explanation,
        "candidate_overall_view": overall_view,
    }


async def save_upload(upload: UploadFile):
    if not upload.filename:
        raise ValueError("No resume file selected.")

    allowed_extensions = (".pdf", ".docx")
    if not upload.filename.lower().endswith(allowed_extensions):
        raise ValueError("Only PDF and DOCX files are supported.")

    safe_filename = Path(upload.filename).name
    destination = UPLOAD_FOLDER / safe_filename
    if destination.exists():
        stem, suffix = destination.stem, destination.suffix
        i = 1
        while destination.exists():
            destination = UPLOAD_FOLDER / f"{stem}_{i}{suffix}"
            i += 1

    with open(destination, "wb") as buffer:
        shutil.copyfileobj(upload.file, buffer)
    return destination, safe_filename


@app.get("/", response_class=HTMLResponse)
def home():
    html_file = BASE_DIR / "templates" / "index.html"
    if not html_file.exists():
        return HTMLResponse("<h1>Error: templates/index.html not found</h1>", status_code=500)
    return HTMLResponse(html_file.read_text(encoding="utf-8"))


@app.post("/analyze")
async def analyze_resume(
    resume: UploadFile = File(...),
    job_description: str = Form(...)
):
    try:
        path, filename = await save_upload(resume)
        return analyze_candidate(path, filename, job_description)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/analyze-bulk")
async def analyze_bulk(
    resumes: List[UploadFile] = File(...),
    job_description: str = Form(...)
):
    results = []
    errors = []

    for resume in resumes:
        try:
            path, filename = await save_upload(resume)
            results.append(analyze_candidate(path, filename, job_description))
        except Exception as e:
            errors.append({
                "file_name": resume.filename or "Unknown file",
                "error": str(e),
            })

    # Sort only for presentation; the underlying candidate data remains complete.
    results.sort(key=lambda item: item.get("final_candidate_score", 0), reverse=True)

    return {
        "total_files": len(resumes),
        "successful": len(results),
        "failed": len(errors),
        "results": results,
        "errors": errors,
    }
