import os
import torch
from flask import Flask, request, jsonify
from transformers import (
    T5ForConditionalGeneration, 
    T5Tokenizer,
    BertForQuestionAnswering,
    BertTokenizer,
    pipeline
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
    """Handles question generation using T5"""
    
    def __init__(self):
        logger.info("Initializing QuestionGenerator...")
        # Using T5 instead of GPT-2 as it's specifically fine-tuned for question generation
        self.model = T5ForConditionalGeneration.from_pretrained("t5-base")
        self.tokenizer = T5Tokenizer.from_pretrained("t5-base")
        
        # Set up QG pipeline
        self.qg_pipeline = pipeline(
            "text2text-generation", 
            model=self.model, 
            tokenizer=self.tokenizer
        )
        
        self.max_length = 512
        logger.info("QuestionGenerator initialized successfully")
    
    def generate_questions(self, text, num_questions=5):
        """Generate questions from the given text"""
        try:
            sentences = self.preprocess_text(text)
            questions = []
            
            for sentence in sentences[:num_questions]:
                # T5 specific prompt format
                prompt = f"generate question: {sentence}"
                
                try:
                    # Generate question using T5
                    output = self.qg_pipeline(
                        prompt,
                        max_length=64,
                        num_return_sequences=1,
                        do_sample=True,
                        top_p=0.95,
                        temperature=0.7
                    )
                    
                    question = output[0]['generated_text']
                    
                    # Clean up and format the question
                    question = question.strip()
                    if not question.endswith("?"):
                        question += "?"
                    
                    if len(question) < 10 or not any(word in question.lower() for word in ['what', 'when', 'where', 'who', 'how', 'why']):
                        question =(sentence)
                    
                    questions.append({
                        "question": question,
                        "context": sentence
                    })
                    
                except Exception as e:
                    logger.error(f"Error generating specific question: {str(e)}")
                    questions.append({
                        "question": self.generate_default_question(sentence),
                        "context": sentence
                    })
            
            return questions
            
        except Exception as e:
            logger.error(f"Error generating questions: {str(e)}")
            raise

class AnswerValidator:
    """Handles answer validation using BERT"""
    
    def __init__(self):
        logger.info("Initializing AnswerValidator...")
        # Using BERT instead of DistilBERT for better accuracy
        self.model = BertForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
        self.tokenizer = BertTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
        
        # Set up QA pipeline
        self.qa_pipeline = pipeline(
            "question-answering",
            model=self.model,
            tokenizer=self.tokenizer
        )
        logger.info("AnswerValidator initialized successfully")
    
    def validate_answer(self, question, context, user_answer):
        """Validate user's answer against the model's prediction"""
        try:
            # Use the QA pipeline for prediction
            result = self.qa_pipeline(
                question=question,
                context=context
            )
            
            predicted_answer = result['answer']
            confidence = result['score']
            
            # Compare with user's answer using more sophisticated matching
            user_answer = user_answer.lower().strip()
            predicted_answer = predicted_answer.lower().strip()
            
            # Check for exact match or significant overlap
            is_correct = (
                predicted_answer in user_answer or 
                user_answer in predicted_answer or
                self._calculate_overlap(user_answer, predicted_answer) > 0.7
            )
            
            return {
                "is_correct": is_correct,
                "predicted_answer": predicted_answer,
                "confidence": confidence
            }
            
        except Exception as e:
            logger.error(f"Error validating answer: {str(e)}")
            raise
    
    def _calculate_overlap(self, text1, text2):
        """Calculate text overlap ratio"""
        words1 = set(text1.split())
        words2 = set(text2.split())
        overlap = words1.intersection(words2)
        return len(overlap) / max(len(words1), len(words2))
    
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