import os
import streamlit as st
from dotenv import load_dotenv

# Import our RAG helper functions
from rag_utility import (
    extract_text_from_pdfs,
    split_documents,
    create_vector_db,
    get_rag_chain,
    query_rag,
    generate_quiz_from_docs,
    generate_interview_questions_from_docs,
    evaluate_interview_answer,
    classify_document
)

# Load environment variables (override to ensure new edits to .env are picked up)
load_dotenv(override=True)

import json

CHAT_HISTORY_FILE = "chat_history.json"

def load_chat_history():
    if os.path.exists(CHAT_HISTORY_FILE):
        try:
            with open(CHAT_HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_chat_history(history):
    try:
        with open(CHAT_HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=4, ensure_ascii=False)
    except Exception:
        pass

# Page configuration
st.set_page_config(
    page_title="StudyResume",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS matching RoastResume layout & colors (Light theme, orange accents)
st.markdown("""
<style>
    /* Google Fonts import */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Outfit:wght@400;500;600;700;800&display=swap');
    
    /* Global Styles */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
        background-color: #FAF9F5 !important;
        color: #1F2937 !important;
    }
    
    /* Top Navbar */
    .navbar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.8rem 2rem;
        background-color: #FFFFFF;
        border-bottom: 1px solid #E5E7EB;
        margin-bottom: 2rem;
    }
    .navbar-logo {
        font-family: 'Outfit', sans-serif;
        font-size: 1.4rem;
        font-weight: 800;
        color: #1F2937;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .navbar-logo span {
        color: #FF5A1F;
    }
    .navbar-badge {
        background-color: #FFF0EB;
        color: #FF5A1F;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        border: 1px solid #FFE0D3;
    }

    /* Left Hero Column Elements */
    .hero-badge {
        background-color: #FFF0EB;
        color: #FF5A1F;
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.4rem 1rem;
        border-radius: 30px;
        font-size: 0.85rem;
        font-weight: 700;
        border: 1px solid #FFE0D3;
        margin-bottom: 1.2rem;
    }
    .hero-heading {
        font-family: 'Outfit', sans-serif;
        font-size: 3.2rem;
        font-weight: 800;
        color: #111827;
        line-height: 1.1;
        letter-spacing: -0.04em;
        margin-bottom: 1rem;
    }
    .hero-heading span.gradient {
        background: linear-gradient(135deg, #FF5A1F 0%, #FF2E00 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-style: italic;
    }
    .hero-subtext {
        font-size: 1.05rem;
        color: #4B5563;
        line-height: 1.5;
        margin-bottom: 1.5rem;
    }
    
    /* Feature pills in columns */
    .pill {
        background-color: #FFFFFF;
        color: #374151;
        border: 1px solid #E5E7EB;
        border-radius: 30px;
        padding: 0.4rem 1rem;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
    }
    
    /* Custom Steps & Cards on Right Column */
    .step-card {
        background-color: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 16px;
        padding: 1.8rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.02);
    }
    .step-badge {
        width: 24px;
        height: 24px;
        background-color: #FFF0EB;
        color: #FF5A1F;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.75rem;
        font-weight: 800;
        margin-bottom: 0.6rem;
        border: 1px solid #FFE0D3;
    }
    .step-title {
        font-family: 'Outfit', sans-serif;
        font-size: 1.2rem;
        font-weight: 700;
        color: #111827;
        margin-bottom: 0.8rem;
    }
    
    /* Custom Dropzone drag block */
    [data-testid="stFileUploader"] {
        border: 2px dashed #FFD4C5 !important;
        background-color: #FFFDFB !important;
        border-radius: 12px !important;
        padding: 1rem !important;
    }
    
    /* Rounded Cards */
    .ui-panel {
        background-color: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.02);
        margin-bottom: 1.5rem;
    }
    
    /* Buttons overriding */
    div.stButton > button {
        background-color: #FFFFFF !important;
        color: #1F2937 !important;
        border: 1px solid #D1D5DB !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        padding: 0.6rem 1.2rem !important;
        transition: all 0.2s ease !important;
    }
    div.stButton > button:hover {
        background-color: #F9FAFB !important;
        border-color: #9CA3AF !important;
        transform: translateY(-1px);
    }
    
    /* Primary buttons override (Orange Accent) */
    div.stButton > button[type="primary"] {
        background-color: #FF5A1F !important;
        color: #FFFFFF !important;
        border: 1px solid #FF5A1F !important;
        box-shadow: 0 4px 12px rgba(255, 90, 31, 0.2) !important;
    }
    div.stButton > button[type="primary"]:hover {
        background-color: #E04E1D !important;
        border-color: #E04E1D !important;
        box-shadow: 0 6px 16px rgba(255, 90, 31, 0.3) !important;
    }

    /* Hide default Streamlit header and footer */
    [data-testid="stHeader"] {
        display: none !important;
    }
    footer {
        visibility: hidden !important;
    }

    /* Compact padding for layout */
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 1rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
    }

    /* Style Streamlit borders to look exactly like step-card */
    div[data-testid="stVerticalBlockBorder"] {
        background-color: #FFFFFF !important;
        border: 1px solid #E5E7EB !important;
        border-radius: 16px !important;
        padding: 1.5rem !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.02) !important;
        margin-bottom: 1rem !important;
    }

    /* Target badge elements inside step cards */
    .step-badge {
        width: 24px;
        height: 24px;
        background-color: #FFF0EB;
        color: #FF5A1F;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.75rem;
        font-weight: 800;
        margin-bottom: 0.6rem;
        border: 1px solid #FFE0D3;
        font-family: 'Outfit', sans-serif;
    }
    .step-title {
        font-family: 'Outfit', sans-serif;
        font-size: 1.2rem;
        font-weight: 700;
        color: #111827;
        margin-bottom: 0.8rem;
    }

</style>
""", unsafe_allow_html=True)

# Initialize Session States
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None
if "raw_docs" not in st.session_state:
    st.session_state.raw_docs = []
if "processed_files" not in st.session_state:
    st.session_state.processed_files = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = load_chat_history()
if "quiz" not in st.session_state:
    st.session_state.quiz = None
if "quiz_user_answers" not in st.session_state:
    st.session_state.quiz_user_answers = {}
if "quiz_submitted" not in st.session_state:
    st.session_state.quiz_submitted = False
if "interview_questions" not in st.session_state:
    st.session_state.interview_questions = None
if "interview_user_answers" not in st.session_state:
    st.session_state.interview_user_answers = {}
if "interview_feedback" not in st.session_state:
    st.session_state.interview_feedback = {}
if "processed_files_metadata" not in st.session_state:
    st.session_state.processed_files_metadata = {}

# Main Grid Layout (Two Columns: Left for Document Details, Right for interaction)
col_left, col_right = st.columns([4, 8], gap="large")

with col_left:
    st.markdown('<div class="hero-heading">Don\'t Let Hard Concepts <br><span class="gradient">Kill Your Grades.</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtext" style="margin-bottom: 0.8rem;">Upload lecture slides, notes, textbooks, or other documents. Instantly interrogate notes using grounded search, test recall accuracy with quizzes, or prepare with mock interview questions.</div>', unsafe_allow_html=True)
    
    # Feature Badges
    st.markdown('<span class="pill">Free to use</span><span class="pill">Context Grounded</span><span class="pill">Auto-Graded Quizzes</span><span class="pill">Interview Prep</span>', unsafe_allow_html=True)
    
    # Ingested Materials Checklist Card
    if st.session_state.processed_files:
        materials_list_html = ""
        for filename in st.session_state.processed_files:
            meta = st.session_state.processed_files_metadata.get(filename, {"doc_type": "Study Notes", "summary": filename})
            doc_type = meta.get("doc_type", "Study Notes")
            summary = meta.get("summary", "")
            
            badge = "📄"
            if doc_type == "Resume":
                badge = "👔"
            elif doc_type == "Job Description":
                badge = "💼"
            elif doc_type in ("Lecture Notes", "Textbook", "Research Paper", "Study Notes"):
                badge = "📚"
                
            materials_list_html += (
                f'<div style="background-color: #FAF9F5; border: 1px solid #E5E7EB; border-radius: 8px; padding: 0.5rem 0.8rem; margin-bottom: 0.5rem; text-align: left;">'
                f'<div style="font-weight: 700; font-size: 0.85rem; color: #1F2937;">{badge} {filename}</div>'
                f'<div style="font-size: 0.7rem; color: #FF5A1F; margin: 0.05rem 0;">Type: {doc_type}</div>'
                f'<div style="font-size: 0.7rem; color: #6B7280; font-style: italic;">{summary}</div>'
                f'</div>'
            )
            
        wrapper_html = (
            f'<div class="ui-panel">'
            f'<div style="font-size: 0.75rem; font-weight: 700; color: #FF5A1F; letter-spacing: 0.05em; margin-bottom: 0.8rem; text-align: left;">📄 ACTIVE MATERIALS</div>'
            f'{materials_list_html}'
            f'</div>'
        )
        st.markdown(wrapper_html, unsafe_allow_html=True)
    else:
        placeholder_html = (
            f'<div class="ui-panel">'
            f'<div style="font-size: 0.75rem; font-weight: 700; color: #9CA3AF; letter-spacing: 0.05em; margin-bottom: 0.8rem; text-align: left;">📄 ACTIVE MATERIALS</div>'
            f'<div style="font-size: 0.8rem; color: #6B7280; line-height: 1.4; text-align: left;">'
            f'No materials uploaded yet. Upload your PDF notes or documents on the right to start preparing!'
            f'</div>'
            f'</div>'
        )
        st.markdown(placeholder_html, unsafe_allow_html=True)

with col_right:

    # STEP 00: API Configuration Check (Only shows up if no valid API key is present in .env)
    env_key = os.getenv("GOOGLE_API_KEY", "")
    if not env_key:
        env_key = os.getenv("google_api_key", "")
    if not env_key:
        try:
            if "GOOGLE_API_KEY" in st.secrets:
                env_key = st.secrets["GOOGLE_API_KEY"]
            elif "google_api_key" in st.secrets:
                env_key = st.secrets["google_api_key"]
        except Exception:
            pass
    has_api_key = env_key and not env_key.startswith("your_actual")
    
    if not has_api_key:
        with st.container(border=True):
            st.markdown('<div class="step-badge">00</div>', unsafe_allow_html=True)
            st.markdown('<div class="step-title">Enter Gemini API Key</div>', unsafe_allow_html=True)
            api_key_input = st.text_input("", type="password", placeholder="AIzaSy...", label_visibility="collapsed")
            api_key = api_key_input
    else:
        api_key = env_key
        
    # STEP 01: Upload Documents Card
    with st.container(border=True):
        st.markdown('<div class="step-badge">01</div>', unsafe_allow_html=True)
        st.markdown('<div class="step-title">Upload your materials</div>', unsafe_allow_html=True)
        
        uploaded_files = st.file_uploader(
            "Upload PDF Notes:",
            type=["pdf"],
            accept_multiple_files=True,
            label_visibility="collapsed"
        )
        
        # Ingestion button
        process_btn = st.button("🚀 Process Documents", type="primary", use_container_width=True)
        
        if process_btn:
            if not api_key or api_key.startswith("your_actual"):
                st.error("Please enter a Google Gemini API Key first!")
            elif not uploaded_files:
                st.error("Please upload at least one PDF note!")
            else:
                with st.spinner("Extracting PDF contents..."):
                    try:
                        docs = extract_text_from_pdfs(uploaded_files)
                    except Exception as e:
                        st.error(f"Error reading PDFs: {e}")
                        docs = []
                        
                if docs:
                    with st.spinner("Splitting text..."):
                        chunks = split_documents(docs)
                    with st.spinner("Building semantic database..."):
                        try:
                            vector_db = create_vector_db(chunks, api_key)
                            
                            # Classify each document
                            metadata = {}
                            for f in uploaded_files:
                                file_docs = [d for d in docs if d.metadata.get("source") == f.name]
                                first_page_content = file_docs[0].page_content if file_docs else ""
                                try:
                                    meta = classify_document(f.name, first_page_content, api_key)
                                except Exception:
                                    meta = {"doc_type": "Study Notes", "summary": f.name}
                                metadata[f.name] = meta
                                
                            # Save state
                            st.session_state.vector_db = vector_db
                            st.session_state.raw_docs = docs
                            st.session_state.processed_files = [f.name for f in uploaded_files]
                            st.session_state.processed_files_metadata = metadata
                            
                            # Reset states
                            st.session_state.chat_history = []
                            st.session_state.quiz = None
                            st.session_state.quiz_user_answers = {}
                            st.session_state.quiz_submitted = False
                            st.session_state.interview_questions = None
                            st.session_state.interview_user_answers = {}
                            st.session_state.interview_feedback = {}
                            
                            st.success("✅ Knowledge base built successfully! Scroll down or proceed to Step 02.")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Vector Indexing Error: {e}")
                else:
                    st.error("No text could be extracted. Scanned/image-only PDFs are not supported.")
    
    # STEP 02: Active Study Mode Panels (Visible once documents are successfully loaded)
    if st.session_state.vector_db:
        with st.container(border=True):
            st.markdown('<div class="step-badge">02</div>', unsafe_allow_html=True)
            st.markdown('<div class="step-title">Select active study mode</div>', unsafe_allow_html=True)
            
            # Segment selection using selectbox or radio button
            study_mode = st.selectbox(
                "Select Mode:",
                options=["💬 Study Chatbot", "📝 Practice Quiz", "👔 Interview Prep"],
                label_visibility="collapsed"
            )
        
        # --------------------- MODE 1: STUDY CHATBOT ---------------------
        if study_mode == "💬 Study Chatbot":
            with st.container(border=True):
                    st.markdown("### **💬 Study Chatbot**")
                    st.caption("Ask questions strictly grounded in your active notes. Footnote citations are shown below answers.")

                    # Clear chat button
                    if st.session_state.chat_history:
                        if st.button("🗑️ Clear Chat History", use_container_width=True):
                            st.session_state.chat_history = []
                            save_chat_history([])
                            st.rerun()

                    # Display chat logs
                    for message in st.session_state.chat_history:
                        with st.chat_message(message["role"]):
                            st.markdown(message["content"])
                            if message.get("citations"):
                                # Render footnotes in a clean block
                                c_text = "  \n".join([f"• Source: `{c['source']}` (p. {c['page']})" for c in message["citations"]])
                                st.caption(f"📚 **Citations:**  \n{c_text}")

                    # Accept user inputs
                    if prompt := st.chat_input("Ask a question about your files..."):
                        with st.chat_message("user"):
                            st.markdown(prompt)
                        st.session_state.chat_history.append({"role": "user", "content": prompt})
                        save_chat_history(st.session_state.chat_history)

                        with st.chat_message("assistant"):
                            message_placeholder = st.empty()
                            try:
                                rag_chain = get_rag_chain(st.session_state.vector_db, api_key)
                                result = query_rag(rag_chain, prompt)
                                answer = result["answer"]
                                source_docs = result["source_documents"]

                                citations = []
                                for doc in source_docs:
                                    citations.append({
                                        "source": doc.metadata.get("source", "Unknown"),
                                        "page": doc.metadata.get("page", "Unknown")
                                    })

                                message_placeholder.markdown(answer)
                                c_text = "  \n".join([f"• Source: `{c['source']}` (p. {c['page']})" for c in citations])
                                st.caption(f"📚 **Citations:**  \n{c_text}")

                                st.session_state.chat_history.append({
                                    "role": "assistant",
                                    "content": answer,
                                    "citations": citations
                                })
                                save_chat_history(st.session_state.chat_history)
                            except Exception as e:
                                error_msg = f"An error occurred: {str(e)}"
                                message_placeholder.error(error_msg)
                                st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
                                save_chat_history(st.session_state.chat_history)

                # --------------------- MODE 2: PRACTICE QUIZ ---------------------
        elif study_mode == "📝 Practice Quiz":
            with st.container(border=True):
                st.markdown("### **📝 Practice Quiz**")

                if st.session_state.quiz is None:
                    st.write("Generate a 5-Question Multiple Choice Quiz from your documents.")
                    if st.button("✨ Generate Quiz", type="primary", use_container_width=True):
                        with st.spinner("Generating..."):
                            try:
                                quiz_data = generate_quiz_from_docs(st.session_state.raw_docs, api_key)
                                if quiz_data:
                                    st.session_state.quiz = quiz_data
                                    st.session_state.quiz_user_answers = {}
                                    st.session_state.quiz_submitted = False
                                    st.rerun()
                                else:
                                    st.error("Not enough text data found to formulate a quiz.")
                            except Exception as e:
                                st.error(f"Failed to generate: {e}")
                else:
                    # Display Quiz Questions
                    for i, q in enumerate(st.session_state.quiz):
                        st.markdown(f"**Q{i+1}: {q['question']}**")
                        current_selection = st.session_state.quiz_user_answers.get(i, None)
                        select_idx = None
                        if current_selection is not None:
                            try:
                                select_idx = q["options"].index(current_selection)
                            except ValueError:
                                select_idx = None

                        user_choice = st.radio(
                            f"q_opt_{i}",
                            options=q["options"],
                            index=select_idx,
                            key=f"q_radio_{i}",
                            disabled=st.session_state.quiz_submitted,
                            label_visibility="collapsed"
                        )
                        st.session_state.quiz_user_answers[i] = user_choice

                        if st.session_state.quiz_submitted:
                            correct_answer = q["answer"]
                            if user_choice == correct_answer:
                                st.success("✅ Correct!")
                            else:
                                st.error(f"❌ Incorrect. Correct answer: **{correct_answer}**")
                            st.info(f"💡 *Explanation:* {q['explanation']}")
                        st.markdown("---")

                    # Submission Controls
                    if not st.session_state.quiz_submitted:
                        if st.button("Submit Answers", type="primary", use_container_width=True):
                            if len(st.session_state.quiz_user_answers) < len(st.session_state.quiz):
                                st.warning("Please answer all questions before submitting!")
                            else:
                                st.session_state.quiz_submitted = True
                                st.rerun()
                    else:
                        # Calculate Score
                        score = sum([1 for idx, q in enumerate(st.session_state.quiz) if st.session_state.quiz_user_answers.get(idx) == q["answer"]])
                        st.markdown(f"### **Score: {score} / 5**")

                        if st.button("🔄 Try Another Quiz", use_container_width=True):
                            st.session_state.quiz = None
                            st.session_state.quiz_user_answers = {}
                            st.session_state.quiz_submitted = False
                            st.rerun()

                # --------------------- MODE 3: INTERVIEW PREP ---------------------
        elif study_mode == "👔 Interview Prep":
            with st.container(border=True):
                st.markdown("### **👔 Interview Prep**")

                if st.session_state.interview_questions is None:
                    st.write("Generate 5 challenging interview questions. Resumes and Job Descriptions are cross-referenced automatically.")
                    if st.button("👔 Generate Mock Interview Session", type="primary", use_container_width=True):
                        with st.spinner("Preparing..."):
                            try:
                                questions_data = generate_interview_questions_from_docs(st.session_state.raw_docs, api_key)
                                if questions_data:
                                    st.session_state.interview_questions = questions_data
                                    st.session_state.interview_user_answers = {}
                                    st.session_state.interview_feedback = {}
                                    st.rerun()
                                else:
                                    st.error("Could not construct questions.")
                            except Exception as e:
                                st.error(f"Failed to generate: {e}")
                else:
                    for i, q in enumerate(st.session_state.interview_questions):
                        st.markdown(f"**Q{i+1}: {q['question']}**")
                        feedback = st.session_state.interview_feedback.get(i, None)

                        if feedback is not None:
                            st.markdown(f"*Your Answer:*")
                            st.write(st.session_state.interview_user_answers.get(i, ""))
                            st.markdown("---")

                            score = feedback["score"]
                            st.markdown(f"##### **AI Score: {score} / 10**")
                            st.progress(score / 10)
                            st.markdown(f"✅ **Strengths:** {feedback['strengths']}")
                            st.markdown(f"⚠️ **Areas to Improve:** {feedback['improvements']}")

                            with st.expander("📝 Suggested Revision & Model Answer"):
                                st.markdown(f"**Ideal Response:**\n{q['model_answer']}")
                                st.markdown(f"**Your Suggested Version:**\n{feedback['suggested_revision']}")

                            if st.button(f"🔄 Re-answer Question {i+1}", key=f"re_ans_btn_{i}"):
                                del st.session_state.interview_feedback[i]
                                st.session_state.interview_user_answers[i] = ""
                                st.rerun()
                        else:
                            current_input = st.session_state.interview_user_answers.get(i, "")
                            user_ans = st.text_area(
                                "Draft your answer:",
                                value=current_input,
                                placeholder="Explain this concept in your own words...",
                                key=f"user_ans_area_{i}",
                                height=100,
                                label_visibility="collapsed"
                            )
                            st.session_state.interview_user_answers[i] = user_ans

                            if st.button(f"📤 Submit Answer {i+1} for Review", key=f"submit_ans_{i}"):
                                if not user_ans.strip():
                                    st.warning("Please type an answer first!")
                                else:
                                    with st.spinner("Evaluating..."):
                                        try:
                                            eval_result = evaluate_interview_answer(q["question"], user_ans, api_key)
                                            st.session_state.interview_feedback[i] = eval_result
                                            st.rerun()
                                        except Exception as e:
                                            st.error(f"Failed: {e}")
                        st.markdown("---")

                    if st.button("🔄 Reset Interview Prep Session", use_container_width=True):
                        st.session_state.interview_questions = None
                        st.session_state.interview_user_answers = {}
                        st.session_state.interview_feedback = {}
                        st.rerun()
