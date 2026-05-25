# 🎓 AI Study Assistant

An interactive web application powered by Retrieval-Augmented Generation (RAG) that helps students study more effectively. Upload multiple PDF files containing lecture notes, books, or summaries, and then chat with a strictly context-grounded AI assistant or take generated multiple-choice practice quizzes with automated grading and explanations.

---

## 🚀 Features

1. **Multiple PDF Upload:** Upload and process multiple PDF study files simultaneously.
2. **Context-Grounded QA:** Ask questions about your study documents. The assistant is strictly instructed to only answer using information from the documents and refuses to hallucinate facts outside the uploaded context.
3. **Source Citations:** View source citations (filename and page numbers) alongside answers to verify information.
4. **Interactive Quiz Practice:** Automatically generate 5 multiple-choice questions (MCQs) from your documents. Practice, submit, view grading, and read educational explanations immediately.
5. **Modern, Minimalist Theme:** Elegant black-and-white styled user interface with micro-interactions, responsive tabs, and clear layouts.

---

## 🛠️ Tech Stack

- **Frontend UI:** [Streamlit](https://streamlit.io/)
- **AI Framework:** [LangChain](https://www.langchain.com/)
- **Vector Index:** [FAISS (Facebook AI Similarity Search)](https://github.com/facebookresearch/faiss)
- **Large Language Model (LLM) & Embeddings:** [Google Gemini API](https://aistudio.google.com/) (`gemini-1.5-flash` and `embedding-001`)
- **PDF Processor:** [PyPDF](https://pypi.org/project/pypdf/)

---

## 📂 Project Structure

```text
AI Study Assistant/
├── .streamlit/
│   └── config.toml       # Streamlit theme configuration (B&W minimal theme)
├── .env.example          # Sample configuration file for environment variables
├── app.py                # Main Streamlit web application & UI state
├── rag_utility.py        # Core RAG engine, document loader, chunker, & quiz generator
├── requirements.txt      # Required python libraries
└── README.md             # Project documentation (this file)
```

---

## 💻 Setup & Installation

### Prerequisites
- Python 3.9, 3.10, or 3.11 installed.
- A Google Gemini API Key. You can get one for free (within limits) at [Google AI Studio](https://aistudio.google.com/).

### Steps

1. **Clone or navigate to the project directory:**
   ```bash
   cd "d:/AI Study Assistant"
   ```

2. **Create a Python Virtual Environment:**
   *On Windows (PowerShell):*
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```
   *On Linux/macOS:*
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Gemini API Key:**
   - Copy `.env.example` to `.env`:
     ```bash
     copy .env.example .env
     ```
   - Open `.env` in a text editor and add your key:
     ```text
     GOOGLE_API_KEY=AIzaSyYourGeminiApiKeyHere
     ```
   *(Note: If you do not configure a `.env` file, the application UI will provide a password field in the sidebar to enter your key directly.)*

---

## 🏃 Running the Application

Start the Streamlit server by executing:
```bash
streamlit run app.py
```

The application will launch and open in your default web browser (usually at `http://localhost:8501`).

---

## 📘 User Guide

1. **Enter API Key:** If your key is not loaded from a `.env` file, input it in the sidebar.
2. **Upload Materials:** Drag and drop or browse to select your PDF notes in the sidebar.
3. **Ingest Notes:** Click **Process Documents**. Wait for the spinner to finish. You should see a list of Active Documents in the sidebar once successful.
4. **Chat (Tab 1):** Type questions in the message bar at the bottom. Read grounded responses, and click on **Citations & Sources** to see the extracted snippets, document source, and page numbers.
5. **Quiz (Tab 2):** Click **Generate 5-Question Quiz** to make practice questions. Make your selections, click **Submit Answers** to see your score, correct/incorrect items, and detail-rich explanations. Click **Try Another Quiz** to reset.
