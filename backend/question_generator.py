from transformers import GPT2LMHeadModel, GPT2Tokenizer

class QuestionGenerator:
    def __init__(self):
        self.model = GPT2LMHeadModel.from_pretrained("distilgpt2")
        self.tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")

    def generate_questions(self, text, num_questions=5):
        # Encode the input text
        input_ids = self.tokenizer.encode(text, return_tensors="pt")

        # Generate questions
        output_ids = self.model.generate(
            input_ids,
            max_length=50,
            num_return_sequences=num_questions,
            num_beams=4,
            early_stopping=True,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            length_penalty=2.0,
            diversity_penalty=1.0,
            repetition_penalty=3.0
        )

        # Decode the generated questions
        questions = [self.tokenizer.decode(output_id, skip_special_tokens=True) for output_id in output_ids]
        return questions