import streamlit as st
import docx
import os
import openai
import re
from dotenv import load_dotenv

# Only import the functions needed for extracting text
from injest_resumes import extract_text_from_pdf, extract_text_from_docx

from RAG_with_Resumes import (
    vector_search_sql,
    generate_completion,
)

load_dotenv()

def extract_keywords(text):
    # Simple keyword extraction: split on non-word chars, remove short/common words
    stopwords = set(["the", "and", "for", "with", "that", "this", "from", "have", "are", "not", "but", "all", "any", "can", "will", "has", "was", "you", "your", "our", "job", "role", "who", "they", "their", "them", "his", "her", "she", "him", "its", "it's", "in", "on", "at", "to", "of", "as", "by", "an", "be", "or", "is", "a"])
    words = re.findall(r'\b\w+\b', text.lower())
    keywords = set([w for w in words if len(w) > 2 and w not in stopwords])
    return keywords

def highlight_keywords(text, keywords):
    def replacer(match):
        word = match.group(0)
        if word.lower() in keywords:
            return f"<mark>{word}</mark>"
        return word
    return re.sub(r'\b\w+\b', replacer, text)


# Sidebar for instructions
st.sidebar.title("ℹ️ How to Use")
st.sidebar.markdown(
    """
    1. Upload a job description (DOCX or PDF).
    2. Click **Search** to find matching resumes.
    3. Ask questions about the results using the AI Q&A box.
    """
)

st.markdown(
    """
    <style>
    .big-font {font-size: 28px !important; font-weight: bold;}
    .section-title {font-size: 22px !important; color: #4F8BF9;}
    .footer {font-size: 14px; color: #888; margin-top: 40px;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="big-font">🔎 Resume Matching Application</div>', unsafe_allow_html=True)
st.info("Find the most suitable candidates based on a job description. Upload a file or paste the description below.")

st.markdown("---")

col1, col2 = st.columns([1, 1])
with col1:
    uploaded_file = st.file_uploader(
        "📄 **Upload a job description file**",
        type=["docx", "pdf"],
        help="Upload a DOCX or PDF file containing the job description."
    )
with col2:
    typed_job_desc = st.text_area(
        "✍️ **Or type/paste the job description here**",
        height=200,
        placeholder="Paste or type the job description if you don't have a file.",
        help="You can paste or type the job description here if you don't want to upload a file."
    )

job_desc = None

if uploaded_file:
    if uploaded_file.type == "application/pdf":
        with open("temp_uploaded.pdf", "wb") as f:
            f.write(uploaded_file.read())
        job_desc = extract_text_from_pdf("temp_uploaded.pdf")
        if os.path.exists("temp_uploaded.pdf"):
            os.remove("temp_uploaded.pdf")
    else:
        job_desc = extract_text_from_docx(uploaded_file)
elif typed_job_desc.strip():
    job_desc = typed_job_desc.strip()

if job_desc:
    job_keywords = extract_keywords(job_desc) 
    st.markdown('<div class="section-title">📝 Job Description</div>', unsafe_allow_html=True)
    st.markdown(
        f"<pre style='white-space: pre-wrap; background-color: #f6f8fa; border-radius: 6px; padding: 10px;'>{job_desc[:2000] + ('...' if len(job_desc) > 2000 else '')}</pre>",
        unsafe_allow_html=True
    )

    st.markdown("---")
    if st.button("🔍 **Search for Candidates**"):
        with st.spinner("Searching for matching resumes..."):
            st.session_state.results = vector_search_sql(job_desc, num_results=5)
            st.session_state.job_keywords = job_keywords  # <-- Now this works

if "results" in st.session_state:
    results = st.session_state.results
    job_keywords = st.session_state.get("job_keywords", set())
    st.markdown('<div class="section-title">📋 Search Results</div>', unsafe_allow_html=True)
    if results:
        for idx, row in enumerate(results, 1):
            filename, chunkid, chunk, similarity_score, distance_score = row
            highlighted_chunk = highlight_keywords(chunk[:1000], job_keywords)
            with st.expander(f"#{idx}: {filename} (Similarity: {similarity_score:.3f})"):
                st.write(f"**Chunk ID:** {chunkid}")
                st.write(f"**Resume Snippet (matched skills highlighted):**")
                st.markdown(
                    f"<pre style='white-space: pre-wrap; background-color: #f6f8fa; border-radius: 6px; padding: 10px;'>{highlighted_chunk + ('...' if len(chunk) > 1000 else '')}</pre>",
                    unsafe_allow_html=True
                )
                st.write(f"**Distance Score:** {distance_score:.3f}")
                # Download button for the resume file
                resume_path = os.path.join("IT", filename)  # Adjust path as needed
                if os.path.exists(resume_path):
                    with open(resume_path, "rb") as f:
                        st.download_button(
                            label="⬇️ Download Full Resume",
                            data=f,
                            file_name=filename,
                            mime="application/octet-stream",
                            key=f"download_{filename}_{chunkid}" 
                        )
        st.success("Results loaded! Ask the AI below for more insights.")
        st.markdown('<div class="section-title">🤖 Ask the AI about these results</div>', unsafe_allow_html=True)
        user_question = st.text_input("Ask a question about the candidates or results:")
        if user_question:
            with st.spinner("Generating answer..."):
                openai.api_version = os.getenv("OPENAI_API_VERSION", "2024-02-15-preview")
                answer = generate_completion(results, user_question)
            st.markdown("**AI Answer:**")
            if isinstance(answer, dict):
                try:
                    content = answer['choices'][0]['message']['content']
                    st.write(content)
                except Exception:
                    st.write(answer)
            else:
                st.write(answer)
    else:
        st.warning("No candidates found.")

