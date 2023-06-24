import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load the saved model's state dictionary
state_dict = torch.load("pytorch_model.bin", map_location=device)

# Create a new instance of the GPT2LMHeadModel class
model = GPT2LMHeadModel.from_pretrained('gpt2', state_dict=state_dict)

# Set the model to evaluation mode and move it to the CPU
model = model.eval().to(torch.device('cpu'))

# Use the model to generate text
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
prompt = "The quick brown fox"
input_ids = tokenizer.encode(prompt, return_tensors='pt')
output = model.generate(input_ids, max_length=50, num_return_sequences=1)
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

# Print the generated text
print(generated_text)