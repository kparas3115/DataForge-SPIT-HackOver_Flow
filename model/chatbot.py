# chatbot.py
import os
import tempfile
import re
from pathlib import Path
import fitz  # PyMuPDF
from flask import request, jsonify, Blueprint, current_app
from groq import Groq
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import os

load_dotenv()

# Create a blueprint for chatbot API
chatbot_api = Blueprint('chatbot_api', __name__)

# Initialize Groq client
groq_client = Groq(api_key=os.environ.get("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"))

# Store uploaded PDFs and their content (in a real app, use a database)
PAPER_STORAGE = {}
CURRENT_SESSION_PAPER = None

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file"""
    text = ""
    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            text += page.get_text()
        return text
    except Exception as e:
        current_app.logger.error(f"Error extracting text from PDF: {str(e)}")
        return None

def preprocess_text(text):
    """Remove <think> tags and content between them"""
    # Pattern to match <think> tags and their content
    think_pattern = r'<think>.*?</think>'
    # Remove the matched patterns
    cleaned_text = re.sub(think_pattern, '', text, flags=re.DOTALL)
    return cleaned_text.strip()

@chatbot_api.route('/api/upload-paper', methods=['POST'])
def upload_paper():
    """Upload and process a research paper PDF"""
    if 'file' not in request.files:
        return jsonify({"success": False, "error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"success": False, "error": "No selected file"}), 400
    
    if not file.filename.endswith('.pdf'):
        return jsonify({"success": False, "error": "Only PDF files are allowed"}), 400
    
    try:
        # Create temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            temp_path = temp_file.name
            file.save(temp_path)
        
        # Extract text from PDF
        paper_text = extract_text_from_pdf(temp_path)
        if not paper_text:
            return jsonify({"success": False, "error": "Failed to extract text from PDF"}), 500
        
        # Preprocess the text (remove think tags)
        paper_text = preprocess_text(paper_text)
        
        # Store the paper text in memory
        global CURRENT_SESSION_PAPER
        CURRENT_SESSION_PAPER = {
            'filename': secure_filename(file.filename),
            'text': paper_text
        }
        
        # Clean up temp file
        os.unlink(temp_path)
        
        return jsonify({
            "success": True,
            "message": f"Paper '{file.filename}' uploaded and processed successfully"
        })
    
    except Exception as e:
        current_app.logger.error(f"Error processing PDF: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

@chatbot_api.route('/api/chat', methods=['POST'])
def chat():
    """Process user questions about the uploaded paper"""
    data = request.get_json()
    if not data or 'message' not in data:
        return jsonify({"success": False, "error": "No message provided"}), 400
    
    user_message = data['message']
    
    # Check if a paper has been uploaded
    if not CURRENT_SESSION_PAPER:
        return jsonify({
            "success": False, 
            "response": "Please upload a research paper first."
        }), 400
    
    try:
        # Create a system prompt with context about the paper
        paper_context = CURRENT_SESSION_PAPER['text']
        # Truncate if necessary (Groq models may have token limits)
        if len(paper_context) > 15000:
            paper_context = paper_context[:15000] + "...[text truncated]"
        
        # Create the conversation with the LLM
        completion = groq_client.chat.completions.create(
            model="qwen-qwq-32b",
            messages=[
                {
                    "role": "system", 
                    "content": f"""You are a helpful research assistant. Answer questions based on the following research paper. 
                    Be accurate and cite specific sections when possible. Format your responses using Markdown for better readability.
                    Use headers, bullet points, and emphasis where appropriate.
                    
                    PAPER CONTENT:
                    {paper_context}"""
                },
                {"role": "user", "content": user_message}
            ],
            temperature=0.6,
            max_tokens=4096,
            top_p=0.95,
        )
        
        # Extract the response
        response = completion.choices[0].message.content
        
        return jsonify({
            "success": True,
            "response": response
        })
        
    except Exception as e:
        current_app.logger.error(f"Error in chatbot processing: {str(e)}")
        return jsonify({
            "success": False,
            "response": f"An error occurred: {str(e)}"
        }), 500

def register_chatbot_routes(app):
    app.register_blueprint(chatbot_api)