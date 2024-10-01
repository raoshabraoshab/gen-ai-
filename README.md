# gen-ai-
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Load the pre-trained model and tokenizer
model = AutoModelForSeq2SeqLM.from_pretrained("codegen-2m-mono")
tokenizer = AutoTokenizer.from_pretrained("codegen-2m-mono")

# Define a function to generate code from natural language descriptions
def generate_code(prompt):
    # Tokenize the input prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    # Generate code using the model
    output = model.generate(input_ids, max_length=512, num_beams=4)

    # Decode the generated code
    code = tokenizer.decode(output[0], skip_special_tokens=True)

    return code

# Test the function
prompt = "Create a Python function to calculate the area of a rectangle"
code = generate_code(prompt)
print(code)
