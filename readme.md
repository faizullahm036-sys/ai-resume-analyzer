# AI Resume Analyzer & Job Matcher

An NLP-based web application that analyzes a resume and compares it
with a job description.

The application extracts text from a PDF resume, calculates a
similarity score using TF-IDF and cosine similarity, and identifies
technical skills that match or are potentially missing from the
target job description.

## Features

- Upload resume in PDF format
- Extract text from PDF
- Compare resume with job description
- Calculate resume-job similarity score
- Detect technical skills
- Identify potentially missing skills
- Interactive web interface
- Simple and easy-to-use design

## Technologies

- Python
- Streamlit
- Scikit-learn
- PyPDF
- Natural Language Processing
- TF-IDF
- Cosine Similarity

## How It Works

1. User uploads a resume in PDF format.
2. Text is extracted from the PDF.
3. Resume and job description text are cleaned.
4. TF-IDF converts the text into numerical vectors.
5. Cosine similarity calculates the similarity percentage.
6. Technical skills are detected from both texts.
7. Matching and potentially missing skills are displayed.

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/ai-resume-analyzer.git