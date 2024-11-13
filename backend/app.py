import os
from flask import Flask, request, jsonify
from transformers import pipeline
from PIL import Image
import PyPDF2
import pytesseract
from docx import Document
import nltk
from nltk.tokenize import sent_tokenize

# Download NLTK data
nltk.download('punkt')

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Initialize ML models
try:
    question_generator = pipeline(
        "text2text-generation",
        model="valhalla/t5-small-qg-hl",
        device=-1  # CPU mode
    )
    qa_model = pipeline(
        "question-answering",
        model="distilbert-base-uncased-distilled-squad",
        device=-1
    )
    print("Models initialized successfully")
except Exception as e:
    print(f"Error initializing models: {e}")
    question_generator = None
    qa_model = None

def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    return " ".join(page.extract_text() for page in reader.pages)

def extract_text_from_docx(file):
    doc = Document(file)
    return " ".join(paragraph.text for paragraph in doc.paragraphs)

def extract_text_from_image(file):
    image = Image.open(file)
    return pytesseract.image_to_string(image)

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files['file']
        if not file.filename:
            return jsonify({"error": "Empty filename"}), 400

        # Extract text based on file type
        if file.filename.endswith('.pdf'):
            text = extract_text_from_pdf(file)
        elif file.filename.endswith('.docx'):
            text = extract_text_from_docx(file)
        elif file.filename.endswith(('.jpg', '.jpeg', '.png')):
            text = extract_text_from_image(file)
        else:
            return jsonify({"error": "Unsupported file type"}), 400

        return jsonify({
            "success": True,
            "extracted_text": text
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/generate-questions', methods=['POST'])
def generate_questions():
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({"error": "No text provided"}), 400

        text = data['text']
        sentences = sent_tokenize(text)
        qa_pairs = []
        
        for i, sentence in enumerate(sentences[:5]):  # Limit to the first 5 sentences
            if len(sentence.strip()) < 10:  # Skip short sentences
                continue

            # Generate question using question generator
            question_prompt = f"Generate a question: {sentence.strip()}"
            question = question_generator(question_prompt, max_length=50, num_return_sequences=1)[0]['generated_text']
            
            # Get answer using QA model
            answer = qa_model(question=question, context=sentence)
            
            qa_pairs.append({
                "question": question,
                "answer": answer['answer'],
                "context": sentence
            })

        return jsonify({
            "success": True,
            "qa_pairs": qa_pairs
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/answer', methods=['POST'])
def validate_answer():
    try:
        data = request.get_json()
        if not data or 'question' not in data or 'context' not in data:
            return jsonify({"error": "Missing question or context"}), 400

        answer = qa_model(
            question=data['question'],
            context=data['context']
        )

        return jsonify({
            "success": True,
            "answer": answer['answer'],
            "confidence": answer['score']
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)