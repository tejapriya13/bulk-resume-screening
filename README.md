# Smart Hiring-AI

One AI model powers both Single Resume testing and Bulk Screening.

## Run
```powershell
pip install -r requirements.txt
uvicorn app:app --reload
```
Open http://127.0.0.1:8000

The UI keeps the same purple/green/orange template. The Analysis Summary remains a score-card layout, and Skills Analysis, Strengths, and Improvement Suggestions are shown as three side-by-side visual cards with icons. Candidate Overall View uses wording instead of contribution percentages.
