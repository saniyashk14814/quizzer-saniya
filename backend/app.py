import os
import torch
from flask import Flask, request, jsonify
from transformers import (
    GPT2LMHeadModel, 
    GPT2Tokenizer, 
    DistilBertForQuestionAnswering, 
    DistilBertTokenizer
)
from PIL import Image
import PyPDF2
import pytesseract
from docx import Document
import nltk
from nltk.tokenize import sent_tokenize
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download NLTK data
nltk.download('punkt', quiet=True)

class TextExtractor:
    """Handles text extraction from various file formats"""
    
    def __init__(self, upload_folder):
        self.upload_folder = Path(upload_folder)
        self.upload_folder.mkdir(parents=True, exist_ok=True)
        
    def save_file(self, file):
        """Save uploaded file and return the path"""
        file_path = self.upload_folder / file.filename
        file.save(str(file_path))
        return file_path
    
    def extract_text(self, file):
        """Extract text from different file types"""
        try:
            file_path = self.save_file(file)
            
            if file.filename.endswith('.pdf'):
                text = self._extract_from_pdf(file_path)
            elif file.filename.endswith('.docx'):
                text = self._extract_from_docx(file_path)
            elif file.filename.endswith(('.png', '.jpg', '.jpeg')):
                text = self._extract_from_image(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file.filename}")
            
            # Clean up the temporary file
            file_path.unlink()
            
            return text.strip()
            
        except Exception as e:
            logger.error(f"Error extracting text: {str(e)}")
            raise
    
    def _extract_from_pdf(self, file_path):
        """Extract text from PDF files"""
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            return " ".join(page.extract_text() for page in reader.pages)
    
    def _extract_from_docx(self, file_path):
        """Extract text from DOCX files"""
        doc = Document(file_path)
        return " ".join(paragraph.text for paragraph in doc.paragraphs)
    
    def _extract_from_image(self, file_path):
        """Extract text from image files using OCR"""
        image = Image.open(file_path)
        return pytesseract.image_to_string(image)


class QuestionGenerator:
    """Handles question generation using GPT-2"""

    def __init__(self):
        logger.info("Initializing QuestionGenerator...")
        self.model = GPT2LMHeadModel.from_pretrained("distilgpt2")
        self.tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
        self.max_chunk_length = 300  # Set maximum chunk length for input text
        logger.info("QuestionGenerator initialized successfully")

    def preprocess_text(self, text):
        """Split text into sentences, clean, and filter short sentences."""
        sentences = sent_tokenize(text)
        return [re.sub(r'\s+', ' ', sent.strip()) for sent in sentences if len(sent.split()) >= 5]

    def chunk_text(self, text):
        """Divide text into chunks of manageable length for model processing."""
        words = text.split()
        chunks = []
        chunk = []

        for word in words:
            chunk.append(word)
            if len(chunk) >= self.max_chunk_length:
                chunks.append(' '.join(chunk))
                chunk = []

        # Add any remaining words as the last chunk
        if chunk:
            chunks.append(' '.join(chunk))

        return chunks

    def generate_questions(self, text, num_questions=5):
        """Generate questions from the given text by chunking it into manageable parts."""
        try:
            cleaned_text = " ".join(self.preprocess_text(text))
            text_chunks = self.chunk_text(cleaned_text)
            questions = []

            for chunk in text_chunks[:num_questions]:  # Limit number of questions
                # Create a prompt for question generation
                prompt = f"Generate a question: {chunk}"
                inputs = self.tokenizer.encode(prompt, return_tensors="pt", max_length=self.max_chunk_length, truncation=True)

                # Generate question
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=50,  # Control output length
                    num_return_sequences=1,
                    no_repeat_ngram_size=2,
                    do_sample=True,
                    top_k=50,
                    top_p=0.95,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.eos_token_id
                )

                question = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                questions.append({
                    "question": question,
                    "context": chunk
                })

            return questions

        except Exception as e:
            logger.error("Error generating questions: %s", str(e))
            raise


class AnswerValidator:
    """Handles answer validation using DistilBERT"""
    
    def __init__(self):
        logger.info("Initializing AnswerValidator...")
        self.model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased-distilled-squad")
        self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        logger.info("AnswerValidator initialized successfully")
    
    def validate_answer(self, question, context, user_answer):
        """Validate user's answer against the model's prediction"""
        try:
            # Encode the question and context
            inputs = self.tokenizer(
                question,
                context,
                return_tensors="pt",
                max_length=512,
                truncation=True
            )
            
            # Get the model's prediction
            outputs = self.model(**inputs)
            start_scores = outputs.start_logits
            end_scores = outputs.end_logits
            
            # Find the most likely answer span
            start_idx = torch.argmax(start_scores)
            end_idx = torch.argmax(end_scores)
            
            # Get the predicted answer
            predicted_answer = self.tokenizer.decode(
                inputs.input_ids[0][start_idx:end_idx+1],
                skip_special_tokens=True
            )
            
            # Calculate confidence score
            confidence = float(torch.max(start_scores) * torch.max(end_scores))
            
            # Compare with user's answer (simple string matching for now)
            is_correct = predicted_answer.lower() in user_answer.lower() or user_answer.lower() in predicted_answer.lower()
            
            return {
                "is_correct": is_correct,
                "predicted_answer": predicted_answer,
                "confidence": confidence
            }
            
        except Exception as e:
            logger.error(f"Error validating answer: {str(e)}")
            raise

# Initialize Flask app
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"

# Initialize components
text_extractor = TextExtractor(app.config["UPLOAD_FOLDER"])
question_generator = QuestionGenerator()
answer_validator = AnswerValidator()

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and text extraction"""
    try:
        if 'file' not in request.files:
            return jsonify({"status": "error", "error": "No file provided"}), 400
        
        file = request.files['file']
        if not file.filename:
            return jsonify({"status": "error", "error": "Empty filename"}), 400
        
        extracted_text = text_extractor.extract_text(file)
        return jsonify({
            "status": "success",
            "extracted_text": extracted_text
        })
        
    except Exception as e:
        logger.error(f"Error in upload_file: {str(e)}")
        return jsonify({"status": "error", "error": str(e)}), 500

@app.route('/generate-questions', methods=['POST'])
def generate_questions():
    """Generate questions from provided text"""
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({"status": "error", "error": "No text provided"}), 400
        
        num_questions = data.get('num_questions', 5)
        questions = question_generator.generate_questions(data['text'], num_questions)
        
        return jsonify({
            "status": "success",
            "questions": questions
        })
        
    except Exception as e:
        logger.error(f"Error in generate_questions: {str(e)}")
        return jsonify({"status": "error", "error": str(e)}), 500

@app.route('/validate-answer', methods=['POST'])
def validate_answer():
    """Validate user's answer"""
    try:
        data = request.get_json()
        if not all(k in data for k in ['question', 'context', 'user_answer']):
            return jsonify({"status": "error", "error": "Missing required fields"}), 400
        
        result = answer_validator.validate_answer(
            data['question'],
            data['context'],
            data['user_answer']
        )
        
        return jsonify({
            "status": "success",
            "result": result
        })
        
    except Exception as e:
        logger.error(f"Error in validate_answer: {str(e)}")
        return jsonify({"status": "error", "error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)