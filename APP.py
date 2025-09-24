# ==============================================================================
# Step 1: Saare zaroori libraries import karein
# ==============================================================================
import numpy as np
import torch
from flask import Flask, request, jsonify, render_template
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
import markdown2

import os
from werkzeug.utils import secure_filename
from langchain_community.document_loaders import PyPDFLoader
import google.generativeai as genai

from pdf2image import convert_from_path
import pytesseract
from PIL import Image

from langchain_community.vectorstores import FAISS

import google.generativeai as genai
genai.configure(api_key="AIzaSyC2jPYqG-_4o2b0LNZvM6jqon8epUS5YqA")

# ==============================================================================
# Step 2: Basic setup (Flask App aur Hugging Face Login)
# ==============================================================================
app = Flask(__name__)

# Hugging Face mein login karein. Token ko yahan hardcode karne se behtar hai
# ki aap isse environment variable mein store karein.
try:
    login("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")
    print("✅ Successfully logged into Hugging Face Hub")
except Exception as e:
    print(f"⚠️ Could not log into Hugging Face Hub. Error: {e}")


# ==============================================================================
# Step 3: RAG Pipeline Class ko define karein

import os
import google.generativeai as genai
from langchain_community.vectorstores import FAISS

class RAGPipeline:
    def __init__(self,
                 gemini_api_key="AIzaSyC2jPYqG-_4o2b0LNZvM6jqon8epUS5YqA",
                 faiss_path="Delhi_vector",
                 embedding_model="sentence-transformers/all-MiniLM-L6-v2",
                 data_path="data/documents"):

        print("🚀 Initializing RAG Pipeline with Gemini 1.5 Flash...")

        # Configure Gemini API
        genai.configure(api_key=gemini_api_key)
        self.model_name = "gemini-1.5-flash"

        # Load embedding model
        print(f"⏳ Loading embedding model: {embedding_model}")
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        print("✅ Embedding model loaded.")

        self.faiss_path = faiss_path
        self.data_path = data_path
        self.load_or_build_vector_store()

    def load_or_build_vector_store(self):
        try:
            print(f"⏳ Loading FAISS index from: {self.faiss_path}")
            self.vector_store = FAISS.load_local(
                self.faiss_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            print("✅ FAISS index loaded successfully.")
        except Exception as e:
            print(f"⚠️ Error loading FAISS index: {e}. Rebuilding...")
            self.rebuild_index()

    def rebuild_index(self, chunk_size=800, chunk_overlap=100):
        if not os.path.exists(self.data_path):
            raise ValueError(f"Data directory not found: {self.data_path}")

        print("🔧 Rebuilding FAISS index...")
        loaders = [
            DirectoryLoader(self.data_path, glob="**/*.pdf", loader_cls=PyPDFLoader),
            DirectoryLoader(self.data_path, glob="**/*.txt", loader_cls=TextLoader)
        ]
        
        documents = []
        for loader in loaders:
            try:
                documents.extend(loader.load())
            except:
                continue

        if not documents:
            raise ValueError("No documents found.")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        splits = text_splitter.split_documents(documents)
        print(f"📄 Created {len(splits)} chunks.")

        self.vector_store = FAISS.from_documents(splits, self.embeddings)
        self.vector_store.save_local(self.faiss_path)
        print(f"✅ FAISS index saved.")

    def retrieve_context(self, query: str, k: int = 15) -> str:
        results = self.vector_store.similarity_search(query, k=k)

        # Merge results into one long context
        merged_context = "\n\n---\n\n".join([doc.page_content for doc in results])

        print(f"✅ Retrieved {len(results)} chunks, total length: {len(merged_context)} chars")
        return merged_context

    def generate_response(self, query: str, context: str) -> str:
        prompt =f"""
You are a highly skilled **Delhi High Court Case Law Expert AI**. Your purpose is to provide clear, accurate, and well-structured answers based ONLY on the provided context from court judgments.

**Your Task:**
Analyze the following `Context` from various Delhi High Court judgments and answer the `Question`.

**Formatting Rules (CRITICAL):**
- **Use Markdown for all formatting.**
- Start with a concise `### Summary` of the answer in 2-3 sentences.
- Use headings like `### Key Legal Principles` or `### Analysis of Relevant Judgments`.
- Use bullet points (`*`) to list legal principles, arguments, or key takeaways.
- Use bold text (`**text**`) to emphasize crucial terms, case names (e.g., **Kesavananda Bharati v. State of Kerala**), and final conclusions.
- If specific judgments are cited in the context, refer to them explicitly.
- Conclude with a clear `### Conclusion` section.
- **DO NOT** include information that is not present in the provided context.

---

### Context:
    {context}

    Question:
    {query}
    """
        try:
            model = genai.GenerativeModel(self.model_name)
            response = model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            print(f"[Gemini API Error] {e}")
            return f"Error generating response: {e}"

    def ask(self, query: str) -> dict:
        context = self.retrieve_context(query)
        answer = self.generate_response(query, context)
        return {
            "answer": answer,
            "status": "success" if not answer.startswith("Error") else "error"
        }


UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


def summarize_pdf_document(file_path):
    """
    Loads a PDF, tries to extract text directly, and falls back to OCR if needed.
    """
    full_text = ""
    try:
        # --- Method 1: Try direct text extraction first ---
        print("Attempting direct text extraction...")
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        full_text = "\n".join([doc.page_content for doc in documents]).strip()
        print("Direct extraction successful.")

    except Exception as e:
        print(f"Direct text extraction failed: {e}. Falling back to OCR.")
        full_text = "" # Ensure text is empty if direct extraction fails

    # --- Method 2: Fallback to OCR if direct extraction yields no text ---
    if not full_text:
        print("Performing OCR on the PDF...")
        try:
            # For Windows, you may need to specify the poppler path:
            # images = convert_from_path(file_path, poppler_path=r"C:\path\to\poppler\bin")
            images = convert_from_path(file_path)
            
            # Use Pytesseract to extract text from each image
            for img in images:
                full_text += pytesseract.image_to_string(img) + "\n"
            print("OCR extraction successful.")
        except Exception as ocr_error:
            print(f"OCR processing failed: {ocr_error}")
            return f"Error: Both direct text extraction and OCR failed. Error: {ocr_error}"

    if not full_text.strip():
        return "Error: Could not extract any text from the document, even with OCR."

    # --- Generate the summary ---
    try:
        print("Generating summary...")
        prompt = f"""
        As a specialized legal AI, provide a structured and detailed summary of the following legal document.
        Format your response as clean, readable plain text. Do not use Markdown syntax (like ### or *).

        Instructions:
        1.  Start with a "Core Summary" section that explains the document's purpose.
        2.  Create a "Key Points" section using hyphens (-) for each bullet point.
        3.  Conclude with a "Conclusion" section.
        4.  Use double line breaks to separate paragraphs and sections for clarity.

        ---
        **Document Text:**
        {full_text}
        ---

        **Structured Plain Text Summary:**
        """
        
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        print("Summary generation successful.")
        return response.text.strip()

    except Exception as e:
        print(f"Error during summary generation: {e}")
        return f"An error occurred while generating the summary: {e}"
    finally:
        # Clean up the uploaded file
        if os.path.exists(file_path):
            os.remove(file_path)


# ==============================================================================
# Step 4: Flask Routes ko define karein
# ==============================================================================

# RAG pipeline ka ek instance banayein (yeh app start hone par ek baar hi banega)
rag_pipeline = RAGPipeline(faiss_path="Delhi_vector")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/Chatbot")
def chatbot_page():
    return render_template("Chatbot.html")

@app.route("/login")
def login_page():  
    return render_template("login.html")

@app.route("/judges")
def judges_page():  
    return render_template("court-tracking.html")

@app.route("/library")
def library_page():
    return render_template("library.html")

@app.route("/document_analysis_page")
def document_analysis_page():  
    return render_template("Document-analysis.html")

@app.route("/query", methods=["POST"])
def handle_query():
    data = request.get_json()
    question = data.get("question", "")

    if not question:
        return jsonify({
            "answer": "Please ask a question.",
            "status": "error"
        }), 400

    try:
        # Get RAG response (this is the raw Markdown text)
        result = rag_pipeline.ask(question)

        # ✨ NEW STEP: Convert the Markdown answer to HTML
        html_answer = markdown2.markdown(result["answer"])

        # Return the HTML answer to the front-end
        return jsonify({
            "question": question,
            "answer": html_answer,  # <-- Send the converted HTML
            "status": result["status"]
        })

    except Exception as e:
        return jsonify({
            "question": question,
            "answer": "I'm currently experiencing technical difficulties. Please try again.",
            "status": "error"
        }), 500

if not os.path.exists('uploads'):
    os.makedirs('uploads')

@app.route('/api/summarize', methods=['POST'])
def summarize_route():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    
    file = request.files['file']
    
    if file.filename == '' or not file.filename.lower().endswith('.pdf'):
        return jsonify({"error": "Please select a valid PDF file"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    # Get the plain text summary
    summary_text = summarize_pdf_document(filepath)
    
    if summary_text.startswith("Error"):
         return jsonify({"error": summary_text}), 500

    # ✨ UPDATED RESPONSE: Send raw text with the key "summary"
    return jsonify({"summary": summary_text})

# ==============================================================================
# Step 5: App ko run karein
# ==============================================================================
if __name__ == "__main__":
    app.run(debug=True)