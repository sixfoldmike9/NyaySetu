# ==============================================================================
# Step 1: Imports
# ==============================================================================
import numpy as np
import torch
import os
import time
import markdown2
import hashlib
from functools import lru_cache
from flask import Flask, request, jsonify, render_template, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_mail import Mail, Message
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import random
import string

# AI & LangChain Imports
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Google Gemini & Image Processing
import google.generativeai as genai
from pdf2image import convert_from_path
import pytesseract
from PIL import Image

# News API
import requests
from datetime import datetime, timedelta

# Load environment variables
load_dotenv()

# ==============================================================================
# Step 2: Configuration & Setup
# ==============================================================================
app = Flask(__name__)

# --- Database & Secret Key Configuration ---
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('SQLALCHEMY_DATABASE_URI', 'sqlite:///database.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- Email Configuration ---
app.config['MAIL_SERVER'] = os.getenv('MAIL_SERVER', 'smtp.gmail.com')
app.config['MAIL_PORT'] = int(os.getenv('MAIL_PORT', 587))
app.config['MAIL_USE_TLS'] = os.getenv('MAIL_USE_TLS', 'True') == 'True'
app.config['MAIL_USERNAME'] = os.getenv('MAIL_USERNAME')
app.config['MAIL_PASSWORD'] = os.getenv('MAIL_PASSWORD')
app.config['MAIL_DEFAULT_SENDER'] = os.getenv('MAIL_DEFAULT_SENDER', os.getenv('MAIL_USERNAME'))

# Initialize Extensions
db = SQLAlchemy(app)
mail = Mail(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# Ensure Upload Folder Exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# --- AI Configuration ---
# Configure Gemini globally
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables")
genai.configure(api_key=GEMINI_API_KEY)

# News API Configuration
NEWS_API_KEY = os.getenv('NEWS_API_KEY')
NEWS_CACHE = {"data": [], "timestamp": None}
NEWS_CACHE_DURATION = timedelta(hours=1)

# Hugging Face Login
HUGGING_FACE_TOKEN = os.getenv('HUGGING_FACE_TOKEN')
try:
    if HUGGING_FACE_TOKEN:
        login(HUGGING_FACE_TOKEN)
        print("✅ Successfully logged into Hugging Face Hub")
    else:
        print("⚠️ HUGGING_FACE_TOKEN not found in environment variables")
except Exception as e:
    print(f"⚠️ Could not log into Hugging Face Hub. Error: {e}")


# ==============================================================================
# Step 3: Database Models & Auth Logic
# ==============================================================================
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)
    subscription_type = db.Column(db.String(50), default='free')  # 'free', 'monthly', 'annual'
    subscription_start = db.Column(db.DateTime, nullable=True)
    subscription_end = db.Column(db.DateTime, nullable=True)
    is_active = db.Column(db.Boolean, default=True)
    is_verified = db.Column(db.Boolean, default=False)  # Email verification status
    otp = db.Column(db.String(6), nullable=True)  # Store current OTP
    otp_created_at = db.Column(db.DateTime, nullable=True)  # OTP creation time

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# ==============================================================================
# OTP Helper Functions
# ==============================================================================
def generate_otp(length=6):
    """Generate a random 6-digit OTP"""
    return ''.join(random.choices(string.digits, k=length))

def send_otp_email(email, otp):
    try:
        msg = Message(
            subject='NyaySetu - Email Verification OTP',
            recipients=[email]
        )
        msg.html = f"""
        <html>
            <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
                <div style="max-width: 600px; margin: 0 auto; padding: 20px; border: 1px solid #ddd; border-radius: 10px;">
                    <h2 style="color: #2c3e50; text-align: center;">NyaySetu Email Verification</h2>
                    <p>Thank you for registering with NyaySetu!</p>
                    <p>Your One-Time Password (OTP) for email verification is:</p>
                    <div style="background-color: #f4f4f4; padding: 15px; text-align: center; font-size: 32px; font-weight: bold; letter-spacing: 5px; border-radius: 5px; margin: 20px 0;">
                        {otp}
                    </div>
                    <p>This OTP is valid for <strong>10 minutes</strong>.</p>
                    <p>If you didn't request this verification, please ignore this email.</p>
                    <hr style="border: none; border-top: 1px solid #ddd; margin: 20px 0;">
                    <p style="font-size: 12px; color: #666; text-align: center;">
                        © 2024 NyaySetu. All rights reserved.
                    </p>
                </div>
            </body>
        </html>
        """
        mail.send(msg)
        return True
    except Exception as e:
        print(f"Error sending email: {e}")
        return False

def verify_otp(user, entered_otp):
    """Verify if the entered OTP is correct and not expired"""
    if not user.otp or not user.otp_created_at:
        return False

    # Check if OTP matches
    if user.otp != entered_otp:
        return False

    # Check if OTP is expired (10 minutes validity)
    time_diff = datetime.now() - user.otp_created_at
    if time_diff.total_seconds() > 600:  # 10 minutes = 600 seconds
        return False

    return True


# ==============================================================================
# Step 4: RAG Pipeline Class
# ==============================================================================
class RAGPipeline:
    def __init__(self, embedding_model="sentence-transformers/all-MiniLM-L6-v2"):
        print("🚀 Initializing Multi-DB RAG Pipeline...")
        self.model_name = "gemini-2.5-flash"

        # Add response cache
        self.response_cache = {}
        self.cache_max_size = 100

        # Load Embedding Model
        print(f"⏳ Loading embedding model: {embedding_model}")
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        print("✅ Embedding model loaded.")

        # Define Database Paths
        self.db_paths = {
            "delhi": "Delhi_vector",
            "constitution": "Constitution_index",
            "allahabad": "allahbaad_vector_db",
            "punjab_haryana": "punjab_vector_db",
            "madras": "madras_vector_db",
            "supreme_court": "sc_vector_db",
        }

        self.vector_stores = {}
        for name, path in self.db_paths.items():
            if os.path.exists(path):
                print(f"⏳ Loading {name} database from {path}...")
                try:
                    self.vector_stores[name] = FAISS.load_local(
                        path, self.embeddings, allow_dangerous_deserialization=True
                    )
                    print(f"✅ {name} index loaded successfully.")
                except Exception as e:
                    print(f"⚠️ Error loading {name}: {e}")
            else:
                print(f"⚠️ Vector DB path not found: {path}")

        self.default_store_name = "delhi"

    def get_store(self, db_selector):
        store = self.vector_stores.get(db_selector)
        if not store:
            store = self.vector_stores.get(self.default_store_name)
            if not store and self.vector_stores:
                store = list(self.vector_stores.values())[0]
        return store

    def search_cases(self, query: str, db_selector: str = "delhi", k: int = 10):
        selected_store = self.get_store(db_selector)
        if not selected_store:
            return []

        # Fetch more results to allow deduplication
        try:
            results = selected_store.similarity_search(query, k=k*3)
        except Exception as e:
            print(f"Search failed: {e}")
            return []

        unique_cases = []
        seen_sources = set()

        for doc in results:
            source_path = doc.metadata.get("source", "Unknown Case")
            if source_path in seen_sources:
                continue
            seen_sources.add(source_path)

            filename = os.path.basename(source_path)
            title = os.path.splitext(filename)[0].replace("_", " ").title()

            unique_cases.append({
                "title": title,
                "summary": doc.page_content[:250].replace("\n", " ") + "...",
                "source": source_path
            })

            if len(unique_cases) >= k:
                break
        
        return unique_cases

    def retrieve_context(self, query: str, db_selector: str = "delhi", k: int = 7) -> str:
        selected_store = self.get_store(db_selector)
        if not selected_store:
            return ""
        results = selected_store.similarity_search(query, k=k)
        context_parts = []
        total_length = 0
        max_length = 4500  

        for doc in results:
            content = doc.page_content[:800]  
            if total_length + len(content) > max_length:
                break
            context_parts.append(content)
            total_length += len(content)

        return "\n\n---\n\n".join(context_parts)

    def generate_response(self, query: str, context: str, conversation_history: list = None) -> str:
        # Build conversation context if history exists
        conversation_context = ""
        if conversation_history:
            conversation_context = "\n\nPrevious conversation:\n"
            for msg in conversation_history[-3:]:  # Include last 3 exchanges
                conversation_context += f"User: {msg['question']}\nAssistant: {msg['answer']}\n\n"

        # Improved prompt for complete answers with conversation context
        prompt = f"""You are a Legal Expert AI. Provide a complete, thorough answer based on the context.

Context: {context}
{conversation_context}
Current Question: {query}

Instructions:
1. If the user refers to "this", "it", "the answer", or similar references, use the previous conversation context
2. Give a comprehensive answer
3. Use clear markdown formatting
4. Include all relevant details
5. Ensure the response is complete and not cut off

Answer:"""
        try:
            # Configure safety settings to be less restrictive
            from google.generativeai.types import HarmCategory, HarmBlockThreshold

            safety_settings = {
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }

            model = genai.GenerativeModel(
                self.model_name,
                generation_config={
                    "temperature": 0.7,
                    "max_output_tokens": 2048,
                    "top_p": 0.95,
                    "top_k": 40,
                },
                safety_settings=safety_settings
            )

            # Generate content with retry logic
            response = model.generate_content(prompt)

            # Debug logging
            print(f"DEBUG - Response object: {response}")
            print(f"DEBUG - Prompt feedback: {response.prompt_feedback if hasattr(response, 'prompt_feedback') else 'None'}")

            # Check for prompt blocking
            if hasattr(response, 'prompt_feedback'):
                if hasattr(response.prompt_feedback, 'block_reason'):
                    block_reason = response.prompt_feedback.block_reason
                    if block_reason:
                        return f"The prompt was blocked by Gemini. Reason: {block_reason}. Please try rephrasing your question."

            # Check if response has candidates
            if not response.candidates or len(response.candidates) == 0:
                print(f"ERROR: No candidates in response")
                return "Gemini API returned no response candidates. Please try again or rephrase your question."

            # Get the first candidate
            candidate = response.candidates[0]
            print(f"DEBUG - Finish reason: {candidate.finish_reason}")
            print(f"DEBUG - Safety ratings: {candidate.safety_ratings if hasattr(candidate, 'safety_ratings') else 'None'}")

            # Extract text from the response
            if not candidate.content or not candidate.content.parts:
                print(f"ERROR: No content parts in candidate")
                return "Gemini API returned an empty response. Please try again."

            # Build the response text from parts
            text_parts = []
            for part in candidate.content.parts:
                if hasattr(part, 'text') and part.text:
                    text_parts.append(part.text)

            if not text_parts:
                return "Unable to extract text from response. Please try again."

            return ''.join(text_parts).strip()

        except Exception as e:
            print(f"ERROR in generate_response: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            return f"Error generating response: {type(e).__name__}: {str(e)}"

    def _get_cache_key(self, query: str, db_selector: str) -> str:
        """Generate a cache key from query and db_selector"""
        combined = f"{query.lower().strip()}_{db_selector}"
        return hashlib.md5(combined.encode()).hexdigest()

    def ask(self, query: str, db_selector: str = "delhi", conversation_history: list = None) -> dict:
        # Check cache first (only for queries without conversation history)
        cache_key = self._get_cache_key(query, db_selector)
        if not conversation_history and cache_key in self.response_cache:
            print("⚡ Cache hit - returning cached response")
            return self.response_cache[cache_key]

        # Generate new response
        context = self.retrieve_context(query, db_selector)
        answer = self.generate_response(query, context, conversation_history)
        result = {"answer": answer, "status": "success" if not answer.startswith("Error") else "error"}

        # Store in cache (with size limit) - only if no conversation history
        if not conversation_history:
            if len(self.response_cache) >= self.cache_max_size:
                # Remove oldest item (simple FIFO)
                self.response_cache.pop(next(iter(self.response_cache)))
            self.response_cache[cache_key] = result

        return result


# Initialize Pipeline Once
rag_pipeline = RAGPipeline()


# ==============================================================================
# Step 5: Helper Functions (PDF Summary & News)
# ==============================================================================

def fetch_live_legal_news():
    """Fetch live legal news from NewsAPI with caching"""
    # Check if cache is valid
    if NEWS_CACHE["data"] and NEWS_CACHE["timestamp"]:
        if datetime.now() - NEWS_CACHE["timestamp"] < NEWS_CACHE_DURATION:
            return NEWS_CACHE["data"]

    # If no API key provided, return fallback news
    if not NEWS_API_KEY:
        return get_fallback_news()

    try:
        # Fetch from NewsAPI
        url = "https://newsapi.org/v2/everything"
        params = {
            "apiKey": NEWS_API_KEY,
            "q": "law OR legal OR court OR supreme court OR judiciary OR legislation",
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": 6,
            "domains": "barandbench.com,livelaw.in,thehindu.com,indianexpress.com,timesofindia.indiatimes.com"
        }

        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        news_items = []
        for article in data.get("articles", [])[:6]:
            # Format the date
            pub_date = article.get("publishedAt", "")
            try:
                date_obj = datetime.fromisoformat(pub_date.replace('Z', '+00:00'))
                formatted_date = date_obj.strftime("%B %d, %Y")
            except:
                formatted_date = "Recent"

            news_items.append({
                "date": formatted_date,
                "title": article.get("title", "Legal News Update"),
                "excerpt": article.get("description", "")[:150] + "..." if article.get("description") else "Click to read more...",
                "url": article.get("url", "#")
            })

        # Update cache
        NEWS_CACHE["data"] = news_items
        NEWS_CACHE["timestamp"] = datetime.now()

        return news_items

    except Exception as e:
        print(f"Error fetching news: {e}")
        return get_fallback_news()

def get_fallback_news():
    """Return fallback news when API is unavailable"""
    return [
        {
            "date": datetime.now().strftime("%B %d, %Y"),
            "title": "Supreme Court Delivers Landmark Judgment on Digital Rights",
            "excerpt": "The Supreme Court has ruled on a critical case concerning digital privacy and data protection...",
            "url": "https://www.barandbench.com/"
        },
        {
            "date": (datetime.now() - timedelta(days=1)).strftime("%B %d, %Y"),
            "title": "New Legal Framework for Technology Sector Announced",
            "excerpt": "Government introduces comprehensive legislation addressing emerging technologies and their legal implications...",
            "url": "https://www.livelaw.in/"
        },
        {
            "date": (datetime.now() - timedelta(days=2)).strftime("%B %d, %Y"),
            "title": "High Court Rules on Environmental Protection Case",
            "excerpt": "Significant ruling sets precedent for environmental law enforcement and corporate accountability...",
            "url": "https://www.barandbench.com/"
        },
        {
            "date": (datetime.now() - timedelta(days=3)).strftime("%B %d, %Y"),
            "title": "Constitutional Amendment Bill Passes Parliament",
            "excerpt": "Parliament approves key amendments affecting fundamental rights and legal procedures...",
            "url": "https://www.livelaw.in/"
        }
    ]

def summarize_pdf_document(file_path):
    full_text = ""
    try:
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        full_text = "\n".join([doc.page_content for doc in documents]).strip()
    except Exception:
        pass # Fallback to OCR

    if not full_text:
        try:
            images = convert_from_path(file_path)
            for img in images:
                full_text += pytesseract.image_to_string(img) + "\n"
        except Exception as e:
            return f"Error reading document: {e}"

    if not full_text.strip():
        return "Error: Document appears empty."

    try:
        from google.generativeai.types import HarmCategory, HarmBlockThreshold
        from google.generativeai.protos import Candidate as CandidateProto

        # Configure less restrictive safety settings
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }

        prompt = f"Summarize this legal document in structured plain text:\n\n{full_text[:30000]}" # Truncate if too huge
        model = genai.GenerativeModel(
            "gemini-2.5-flash",
            generation_config={
                "temperature": 0.5,
                "max_output_tokens": 2048,
            },
            safety_settings=safety_settings
        )
        response = model.generate_content(prompt)

        print(f"DEBUG PDF Summary - Response: {response}")
        print(f"DEBUG PDF - Prompt feedback: {response.prompt_feedback if hasattr(response, 'prompt_feedback') else 'None'}")

        # Check for prompt blocking
        if hasattr(response, 'prompt_feedback'):
            if hasattr(response.prompt_feedback, 'block_reason'):
                block_reason = response.prompt_feedback.block_reason
                if block_reason:
                    return f"The document content was blocked by Gemini. Reason: {block_reason}"

        # Check if response has candidates
        if not response.candidates or len(response.candidates) == 0:
            return "Gemini API returned no response for the PDF summary. Please try again."

        # Get the first candidate
        candidate = response.candidates[0]
        print(f"DEBUG PDF - Finish reason: {candidate.finish_reason}")
        print(f"DEBUG PDF - Safety ratings: {candidate.safety_ratings if hasattr(candidate, 'safety_ratings') else 'None'}")

        # Extract text from the response
        if not candidate.content or not candidate.content.parts:
            return "Gemini API returned an empty summary. Please try again."

        # Build the response text from parts
        text_parts = []
        for part in candidate.content.parts:
            if hasattr(part, 'text') and part.text:
                text_parts.append(part.text)

        if not text_parts:
            return "Unable to extract text from summary response. Please try again."

        return ''.join(text_parts).strip()

    except Exception as e:
        print(f"ERROR in summarize_pdf_document: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return f"Error generating summary: {type(e).__name__}: {str(e)}"


# ==============================================================================
# Step 6: Routes
# ==============================================================================

@app.route("/")
def home():
    return render_template("index.html", user=current_user)

# --- AUTH ROUTES ---

@app.route("/login", methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        user = User.query.filter_by(email=email).first()

        if user and check_password_hash(user.password, password):
            if not user.is_verified:
                flash('Please verify your email first. Check your inbox for the OTP.', 'warning')
                return redirect(url_for('verify_otp_page', email=email))
            login_user(user)
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid email or password', 'error')

    return render_template("login.html")

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        user = User.query.filter_by(email=email).first()
        if user:
            if not user.is_verified:
                flash('Email already registered but not verified. Redirecting to verification.', 'warning')
                return redirect(url_for('resend_otp', email=email))
            flash('Email already registered. Please log in.', 'error')
            return redirect(url_for('login'))

        # Generate OTP
        otp = generate_otp()

        # Create new user with OTP
        new_user = User(
            email=email,
            password=generate_password_hash(password, method='scrypt'),
            otp=otp,
            otp_created_at=datetime.now(),
            is_verified=False
        )
        db.session.add(new_user)
        db.session.commit()

        # Send OTP email
        if send_otp_email(email, otp):
            flash('Registration successful! Please check your email for the OTP to verify your account.', 'success')
            return redirect(url_for('verify_otp_page', email=email))
        else:
            flash('Account created but failed to send OTP. Please contact support.', 'warning')
            return redirect(url_for('verify_otp_page', email=email))

    return render_template('register.html')

@app.route('/verify-otp', methods=['GET', 'POST'])
def verify_otp_page():
    email = request.args.get('email') or request.form.get('email')

    if not email:
        flash('Invalid verification link.', 'error')
        return redirect(url_for('register'))

    if request.method == 'POST':
        otp = request.form.get('otp')

        user = User.query.filter_by(email=email).first()
        if not user:
            flash('User not found.', 'error')
            return redirect(url_for('register'))

        if verify_otp(user, otp):
            # Mark user as verified
            user.is_verified = True
            user.otp = None  # Clear OTP after verification
            user.otp_created_at = None
            db.session.commit()

            flash('Email verified successfully! You can now log in.', 'success')
            return redirect(url_for('login'))
        else:
            flash('Invalid or expired OTP. Please try again.', 'error')

    return render_template('verify_otp.html', email=email)

@app.route('/resend-otp')
def resend_otp():
    email = request.args.get('email')

    if not email:
        flash('Invalid request.', 'error')
        return redirect(url_for('register'))

    user = User.query.filter_by(email=email).first()
    if not user:
        flash('User not found.', 'error')
        return redirect(url_for('register'))

    if user.is_verified:
        flash('Email already verified. Please log in.', 'info')
        return redirect(url_for('login'))

    # Generate new OTP
    otp = generate_otp()
    user.otp = otp
    user.otp_created_at = datetime.now()
    db.session.commit()

    # Send OTP email
    if send_otp_email(email, otp):
        flash('OTP has been resent to your email.', 'success')
    else:
        flash('Failed to send OTP. Please try again later.', 'error')

    return redirect(url_for('verify_otp_page', email=email))

@app.route('/logout')
def logout():
    logout_user()
    flash('You have been logged out successfully.', 'success')
    return redirect(url_for('home'))

@app.route("/dashboard")
@login_required
def dashboard():
    # You can return index.html or a specific dashboard.html
    return render_template("index.html", user=current_user)

# --- APP ROUTES ---

@app.route("/Chatbot")
def chatbot_page():
    return render_template("Chatbot.html")

@app.route("/research", methods=["GET", "POST"])
def research_page():
    results = []
    query = ""
    if request.method == "POST":
        query = request.form.get("query", "")
        if query and rag_pipeline:
            results = rag_pipeline.search_cases(query, k=10)
    return render_template("advance_research.html", results=results, query=query)

@app.route("/judges")
def judges_page():  
    return render_template("court-tracking.html")

@app.route("/library")
def library_page():
    return render_template("library.html")

@app.route("/document_analysis_page")
def document_analysis_page():
    return render_template("Document-analysis.html")

@app.route("/subscription")
def subscription_page():
    return render_template("subscription.html")

# --- API ROUTES ---

@app.route("/query", methods=["POST"])
def handle_query():
    data = request.get_json()
    question = data.get("question", "")
    conversation_history = data.get("conversation_history", [])  # Accept conversation history

    if not question:
        return jsonify({"answer": "Please ask a question.", "status": "error"}), 400

    try:
        result = rag_pipeline.ask(question, conversation_history=conversation_history)
        html_answer = markdown2.markdown(result["answer"])
        return jsonify({
            "question": question,
            "answer": html_answer,
            "status": result["status"],
            "raw_answer": result["answer"]  # Send raw answer for history tracking
        })
    except Exception as e:
        return jsonify({"answer": "Technical Error.", "status": "error"}), 500

@app.route('/api/summarize', methods=['POST'])
def summarize_route():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '' or not file.filename.lower().endswith('.pdf'):
        return jsonify({"error": "Invalid PDF"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    summary_text = summarize_pdf_document(filepath)
    
    # Cleanup
    if os.path.exists(filepath):
        os.remove(filepath)

    if summary_text.startswith("Error"):
         return jsonify({"error": summary_text}), 500
    return jsonify({"summary": summary_text})

@app.route('/search', methods=['POST'])
def search_api():
    data = request.json
    query = data.get('query', '')
    db_selector = data.get('database', 'delhi')
    results = rag_pipeline.search_cases(query, db_selector, k=10)
    return jsonify(results)

@app.route('/api/news', methods=['GET'])
def get_news():
    """API endpoint to fetch live legal news"""
    try:
        news_items = fetch_live_legal_news()
        return jsonify({"news": news_items, "status": "success"})
    except Exception as e:
        return jsonify({"error": str(e), "status": "error"}), 500

# ==============================================================================
# Step 7: Run App
# ==============================================================================
if __name__ == "__main__":
    with app.app_context():
        db.create_all()  # Creates users.db if it doesn't exist
    app.run(debug=os.getenv('FLASK_DEBUG', 'False') == 'True')