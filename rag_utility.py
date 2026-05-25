import json
import re
import random
import string
from io import BytesIO
from typing import List, Dict, Any, Tuple
import pypdf
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

def extract_text_from_pdfs(uploaded_files) -> List[Document]:
    """
    Extracts text from a list of uploaded PDF files (Streamlit UploadedFile objects).
    Returns a list of LangChain Document objects with metadata for citations.
    """
    documents = []
    for uploaded_file in uploaded_files:
        try:
            # Read PDF from bytes
            pdf_bytes = uploaded_file.read()
            # Reset file pointer just in case Streamlit needs it later
            uploaded_file.seek(0)
            
            pdf_reader = pypdf.PdfReader(BytesIO(pdf_bytes))
            file_name = uploaded_file.name
            
            for page_num, page in enumerate(pdf_reader.pages):
                text = page.extract_text()
                if text and text.strip():
                    # Add document with metadata
                    documents.append(
                        Document(
                            page_content=text,
                            metadata={
                                "source": file_name,
                                "page": page_num + 1
                            }
                        )
                    )
        except Exception as e:
            raise RuntimeError(f"Error processing PDF '{uploaded_file.name}': {str(e)}")
            
    return documents

def split_documents(documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    """
    Splits the extracted text documents into smaller chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    return text_splitter.split_documents(documents)

def create_vector_db(chunks: List[Document], api_key: str) -> FAISS:
    """
    Creates a FAISS vector database from text chunks using Gemini Embeddings.
    """
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=api_key
    )
    vector_db = FAISS.from_documents(chunks, embeddings)
    return vector_db

def get_rag_chain(vector_db: FAISS, api_key: str):
    """
    Sets up the RAG (Retrieval-Augmented Generation) chain using Gemini.
    Answers are strictly grounded in the retrieved context.
    """
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
        temperature=0.2
    )
    
    retriever = vector_db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )
    
    # Custom system prompt to enforce answering strictly from context
    system_prompt = (
        "You are an AI Study Assistant. Your task is to answer the student's question based strictly on the provided study materials.\n\n"
        "Rules:\n"
        "1. Answer the question using ONLY the retrieved context. Be clear, concise, and educational.\n"
        "2. If the user asks to explain a concept, term, or topic:\n"
        "   - Make the explanation easy to understand, clear, and accessible.\n"
        "   - Provide a highly structured answer with logical bullet points or numbered lists.\n"
        "   - Ensure the explanation is of appropriate length (comprehensive and complete, yet well-organized and readable).\n"
        "3. If the context does not contain the answer, say exactly: \n"
        "   'I cannot find the answer to this question in the uploaded study materials.'\n"
        "   Do not make up any facts or use outside knowledge.\n"
        "4. Provide citations to the source document and page number in your answer if relevant.\n\n"
        "Retrieved Context:\n"
        "---------------------\n"
        "{context}\n"
        "---------------------"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    return rag_chain

def query_rag(rag_chain, question: str) -> Dict[str, Any]:
    """
    Queries the RAG chain and returns the answer and retrieved source documents.
    """
    response = rag_chain.invoke({"input": question})
    return {
        "answer": response["answer"],
        "source_documents": response.get("context", [])
    }

def generate_quiz_from_docs(documents: List[Document], api_key: str, limit_chars: int = 15000) -> List[Dict[str, Any]]:
    """
    Generates 5 MCQs from the study documents using the Gemini API.
    Combines text up to limit_chars to stay within reasonable token/rate usage.
    """
    if not documents:
        return []
        
    # Aggregate text up to limit_chars
    context_text = ""
    for doc in documents:
        if len(context_text) + len(doc.page_content) > limit_chars:
            # Add a partial page or break
            remaining = limit_chars - len(context_text)
            if remaining > 100:
                context_text += "\n" + doc.page_content[:remaining]
            break
        context_text += f"\n--- Source: {doc.metadata.get('source', 'Unknown')} (Page {doc.metadata.get('page', 'Unknown')}) ---\n"
        context_text += doc.page_content
        
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
        temperature=0.7  # Higher temperature for creative/varied question generation
    )
    
    salt = "".join(random.choices(string.ascii_letters + string.digits, k=8))
    prompt = (
        "You are an expert educator. Based on the following study materials, generate exactly 5 multiple-choice questions (MCQs) "
        "to test a student's understanding. Each question must have 4 options and exactly one correct answer.\n\n"
        "CRITICAL REQUIREMENT: To ensure variety, focus on different parts, details, or conceptual aspects of the material than other runs.\n"
        f"Session token for uniqueness: {salt}\n\n"
        "Provide the output strictly in JSON format as a list of 5 objects. Each object must have these keys:\n"
        '- "question": The question text.\n'
        '- "options": An array of 4 strings representing the choices.\n'
        '- "answer": The correct choice (must exactly match one of the items in the "options" array).\n'
        '- "explanation": A brief explanation of why this answer is correct.\n\n'
        "Return ONLY the raw JSON array. Do not wrap it in markdown code blocks or add any other text before or after the JSON. "
        "Make sure the JSON is perfectly formatted and syntax-valid.\n\n"
        "Study Materials:\n"
        "---------------------\n"
        f"{context_text}\n"
        "---------------------"
    )
    
    response = llm.invoke(prompt)
    content = response.content.strip()
    
    # Robustly parse JSON: extract array if present, otherwise strip backticks
    try:
        array_match = re.search(r'(\[\s*\{[\s\S]*\}\s*\])', content)
        if array_match:
            content = array_match.group(1).strip()
        else:
            json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', content)
            if json_match:
                content = json_match.group(1).strip()
            
        quiz_data = json.loads(content)
        
        # Validate format
        validated_quiz = []
        for item in quiz_data:
            if all(k in item for k in ("question", "options", "answer", "explanation")) and len(item["options"]) == 4:
                # Make sure answer is in options
                if item["answer"] not in item["options"]:
                    # Default answer to first option if something went wrong
                    item["options"][0] = item["answer"]
                validated_quiz.append(item)
                
        return validated_quiz[:5]
    except Exception as e:
        raise ValueError(f"Failed to generate or parse quiz JSON: {str(e)}\nRaw Response: {content}")

def generate_interview_questions_from_docs(documents: List[Document], api_key: str, limit_chars: int = 15000) -> List[Dict[str, Any]]:
    """
    Generates 5 challenging interview questions based on the study documents.
    """
    if not documents:
        return []
        
    context_text = ""
    for doc in documents:
        if len(context_text) + len(doc.page_content) > limit_chars:
            remaining = limit_chars - len(context_text)
            if remaining > 100:
                context_text += "\n" + doc.page_content[:remaining]
            break
        context_text += f"\n--- Source: {doc.metadata.get('source', 'Unknown')} (Page {doc.metadata.get('page', 'Unknown')}) ---\n"
        context_text += doc.page_content
        
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
        temperature=0.7
    )
    
    salt = "".join(random.choices(string.ascii_letters + string.digits, k=8))
    prompt = (
        "You are an expert technical interviewer or hiring manager. Based on the following study/career materials, "
        "generate exactly 5 challenging interview questions to test understanding.\n\n"
        "CRITICAL REQUIREMENT: To ensure variety, cover different projects, skills, or knowledge areas than other runs.\n"
        f"Session token for uniqueness: {salt}\n\n"
        "SPECIAL RULE:\n"
        "If you detect both a candidate's Resume/CV and a Job Description (JD) in the materials, tailor the interview questions "
        "specifically as a mock interview for this candidate applying to this job. Ask questions that challenge the candidate to: "
        "1. Map their experience to the key requirements of the JD.\n"
        "2. Address potential skill gaps or architectural tradeoffs related to the role.\n"
        "3. Defend projects in their resume relevant to the job.\n"
        "Otherwise, if it is general study notes, textbooks, or research papers, generate academic/conceptual interview prep questions "
        "designed to test a student's understanding of the subject.\n\n"
        "Provide the output strictly in JSON format as a list of 5 objects. Each object must have these keys:\n"
        '- "question": The interview question.\n'
        '- "model_answer": A comprehensive suggested ideal answer/response outline to the question.\n\n'
        "Return ONLY the raw JSON array. Do not wrap it in markdown code blocks or add any other text before or after the JSON.\n\n"
        "Materials:\n"
        "---------------------\n"
        f"{context_text}\n"
        "---------------------"
    )
    
    response = llm.invoke(prompt)
    content = response.content.strip()
    
    try:
        array_match = re.search(r'(\[\s*\{[\s\S]*\}\s*\])', content)
        if array_match:
            content = array_match.group(1).strip()
        else:
            json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', content)
            if json_match:
                content = json_match.group(1).strip()
            
        interview_data = json.loads(content)
        validated_data = []
        for item in interview_data:
            if all(k in item for k in ("question", "model_answer")):
                validated_data.append(item)
        return validated_data[:5]
    except Exception as e:
        raise ValueError(f"Failed to generate or parse interview JSON: {str(e)}\nRaw Response: {content}")

def evaluate_interview_answer(question: str, user_answer: str, api_key: str) -> Dict[str, Any]:
    """
    Evaluates a candidate's answer to an interview question using the Gemini API,
    providing a score, strengths, improvements, and suggested response.
    """
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
        temperature=0.3
    )
    
    prompt = (
        "You are an expert technical interviewer. Evaluate the candidate's answer to the following interview question.\n\n"
        f"Question: {question}\n"
        f"Candidate's Answer: {user_answer}\n\n"
        "Provide constructive, detailed feedback strictly in JSON format. The JSON object must have exactly these keys:\n"
        '- "score": A score from 0 to 10 (as an integer representing the quality of the answer).\n'
        '- "strengths": A string detailing the strong points of the answer (what was correct or well-explained).\n'
        '- "improvements": A string detailing the areas of improvement (what was missing, incomplete, or incorrect).\n'
        '- "suggested_revision": A short, rewritten exemplary answer that incorporates the improvements.\n\n'
        "Return ONLY the raw JSON object. Do not wrap it in markdown code blocks or add any other text before or after the JSON."
    )
    
    response = llm.invoke(prompt)
    content = response.content.strip()
    
    try:
        object_match = re.search(r'(\{\s*[\s\S]*\s*\})', content)
        if object_match:
            content = object_match.group(1).strip()
        else:
            json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', content)
            if json_match:
                content = json_match.group(1).strip()
            
        feedback_data = json.loads(content)
        # Ensure default values if keys are missing
        feedback_data.setdefault("score", 5)
        feedback_data.setdefault("strengths", "No specific strengths identified.")
        feedback_data.setdefault("improvements", "No specific improvements identified.")
        feedback_data.setdefault("suggested_revision", "N/A")
        return feedback_data
    except Exception as e:
        raise ValueError(f"Failed to parse evaluation feedback JSON: {str(e)}\nRaw Response: {content}")

def classify_document(filename: str, page_content: str, api_key: str) -> Dict[str, str]:
    """
    Classifies the uploaded document type and details (e.g. Resume, Job Description, Study Notes).
    """
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
        temperature=0.2
    )
    
    prompt = (
        "You are an expert document classifier. Analyze the following content snippet from the beginning "
        "of a document and classify it. Determine the document type and key metadata (e.g. Candidate Name for Resumes, "
        "Job Title for Job Descriptions, or Topic/Subject for study materials/textbooks).\n\n"
        "Document Filename: {filename}\n"
        "Content Snippet:\n"
        "---------------------\n"
        "{snippet}\n"
        "---------------------\n\n"
        "Provide your classification strictly in JSON format with exactly these keys:\n"
        '- "doc_type": The classified type (e.g., "Resume", "Job Description", "Lecture Notes", "Textbook", "Research Paper", "Other").\n'
        '- "summary": A very brief 4-8 word description highlighting key detail (e.g., "Software Engineer Role at Google" or "John Doe - React Developer" or "Chapter 5: Neural Networks").\n\n'
        "Return ONLY the raw JSON object. Do not wrap it in markdown code blocks or add any other text."
    )
    
    # Use first 2000 characters for classification
    snippet = page_content[:2000]
    
    try:
        response = llm.invoke(prompt.format(filename=filename, snippet=snippet))
        content = response.content.strip()
        
        # Extract JSON object
        object_match = re.search(r'(\{\s*[\s\S]*\s*\})', content)
        if object_match:
            content = object_match.group(1).strip()
            
        data = json.loads(content)
        data.setdefault("doc_type", "Study Notes")
        data.setdefault("summary", filename)
        return data
    except Exception:
        # Fallback defaults
        return {"doc_type": "Document", "summary": filename}

def extract_resume_profile(filename: str, page_content: str, api_key: str) -> Dict[str, Any]:
    """
    Parses resume text using Gemini to build a structured profile and tailored study/career suggestions.
    """
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
        temperature=0.3
    )
    
    prompt = (
        "You are an expert resume parsing and career coaching assistant. Analyze the candidate's resume content below "
        "and extract a structured profile with personalized learning, study, and career recommendations.\n\n"
        "Provide your response strictly in JSON format with exactly these keys:\n"
        '- "name": The candidate\'s full name (or a logical placeholder if not found).\n'
        '- "headline": A professional headline representing their level/role (e.g. "Full Stack Developer | React & Python").\n'
        '- "summary": A brief 2-3 sentence professional summary highlighting their core experience.\n'
        '- "skills": A list of up to 6 key technical or professional skills extracted from the resume.\n'
        '- "suggestions": A list of 3-5 specific, actionable study recommendations or career suggestions tailored to their resume '
        '(e.g., skill gaps to study, project improvements, certifications, or areas to review for interviews).\n\n'
        "Return ONLY the raw JSON object. Do not wrap it in markdown code blocks or add any other text.\n\n"
        "Resume Content:\n"
        "---------------------\n"
        "{snippet}\n"
        "---------------------"
    )
    
    # Send up to 10000 characters to make sure we get enough context from the resume
    snippet = page_content[:10000]
    
    try:
        response = llm.invoke(prompt.format(snippet=snippet))
        content = response.content.strip()
        
        # Extract JSON object
        object_match = re.search(r'(\{\s*[\s\S]*\s*\})', content)
        if object_match:
            content = object_match.group(1).strip()
            
        data = json.loads(content)
        # Verify required keys
        data.setdefault("name", "Candidate")
        data.setdefault("headline", "Professional")
        data.setdefault("summary", "Resume uploaded successfully.")
        data.setdefault("skills", [])
        data.setdefault("suggestions", ["Review your resume alongside relevant job descriptions.", "Practice technical interview questions."])
        return data
    except Exception as e:
        # Fallback details
        return {
            "name": "Candidate Profile",
            "headline": "Profile Loaded",
            "summary": "Could not parse detailed resume insights due to an error.",
            "skills": ["Python", "Machine Learning", "Development"],
            "suggestions": [
                f"Error parsing resume: {str(e)}",
                "Ensure the PDF contains copyable text.",
                "Try re-uploading the resume."
            ]
        }

