import streamlit as st
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="📄",
    layout="wide"
)


# -----------------------------
# Helper functions
# -----------------------------

def extract_text_from_pdf(uploaded_file):
    """Extract text from an uploaded PDF."""
    try:
        reader = PdfReader(uploaded_file)
        text = ""

        for page in reader.pages:
            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

        return text

    except Exception as error:
        st.error(f"Could not read the PDF: {error}")
        return ""


def clean_text(text):
    """Clean text for comparison."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9+#.\s]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def calculate_similarity(resume_text, job_description):
    """Calculate TF-IDF cosine similarity."""
    documents = [
        clean_text(resume_text),
        clean_text(job_description)
    ]

    vectorizer = TfidfVectorizer(
        stop_words="english"
    )

    matrix = vectorizer.fit_transform(documents)

    similarity = cosine_similarity(
        matrix[0:1],
        matrix[1:2]
    )[0][0]

    return round(similarity * 100, 2)


def find_skills(text):
    """Find common technical skills in text."""

    skills = [
        "python",
        "java",
        "c++",
        "c#",
        "javascript",
        "html",
        "css",
        "react",
        "sql",
        "mysql",
        "postgresql",
        "mongodb",
        "pandas",
        "numpy",
        "scikit-learn",
        "tensorflow",
        "pytorch",
        "opencv",
        "machine learning",
        "deep learning",
        "natural language processing",
        "nlp",
        "computer vision",
        "power bi",
        "tableau",
        "excel",
        "aws",
        "azure",
        "docker",
        "git",
        "github",
        "fastapi",
        "flask",
        "streamlit"
    ]

    cleaned_text = clean_text(text)

    found = []

    for skill in skills:

        skill_cleaned = clean_text(skill)

        if skill_cleaned in cleaned_text:
            found.append(skill)

    return sorted(set(found))


# -----------------------------
# Application
# -----------------------------

st.title("📄 AI Resume Analyzer")
st.write(
    "Upload a resume and compare it with a job description "
    "using Natural Language Processing."
)


st.divider()


# Resume upload
st.subheader("1. Upload your Resume")

uploaded_file = st.file_uploader(
    "Upload a PDF resume",
    type=["pdf"]
)


st.subheader("2. Paste the Job Description")

job_description = st.text_area(
    "Job Description",
    height=250,
    placeholder="Paste the complete job description here..."
)


st.divider()


# Analysis
if st.button(
    "🚀 Analyze Resume",
    use_container_width=True
):

    if uploaded_file is None:
        st.warning("Please upload a PDF resume.")

    elif not job_description.strip():
        st.warning("Please paste a job description.")

    else:

        with st.spinner("Analyzing resume..."):

            resume_text = extract_text_from_pdf(
                uploaded_file
            )

        if not resume_text.strip():

            st.error(
                "No readable text was found in the PDF."
            )

        else:

            score = calculate_similarity(
                resume_text,
                job_description
            )

            resume_skills = find_skills(
                resume_text
            )

            job_skills = find_skills(
                job_description
            )


            matched_skills = sorted(
                set(resume_skills) &
                set(job_skills)
            )

            missing_skills = sorted(
                set(job_skills) -
                set(resume_skills)
            )


            st.success("Analysis completed!")

            st.divider()

            # Score
            st.subheader("Resume Match Score")

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    label="Job Match",
                    value=f"{score}%"
                )

            with col2:
                st.metric(
                    label="Skills Found",
                    value=len(resume_skills)
                )


            st.divider()


            # Skills
            col1, col2 = st.columns(2)


            with col1:

                st.subheader(
                    "✅ Matched Skills"
                )

                if matched_skills:

                    for skill in matched_skills:
                        st.write(f"✓ {skill}")

                else:

                    st.write(
                        "No matching skills detected."
                    )


            with col2:

                st.subheader(
                    "⚠️ Potentially Missing Skills"
                )

                if missing_skills:

                    for skill in missing_skills:
                        st.write(f"• {skill}")

                else:

                    st.write(
                        "No obvious missing skills detected."
                    )


            st.divider()


            # Resume skills
            st.subheader(
                "Technical Skills Detected in Resume"
            )

            if resume_skills:

                st.write(
                    ", ".join(resume_skills)
                )

            else:

                st.write(
                    "No predefined technical skills detected."
                )


            st.divider()

            st.caption(
                "This project uses TF-IDF and cosine similarity. "
                "The result is an automated similarity estimate, "
                "not a hiring decision."
            )