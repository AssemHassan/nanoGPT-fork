import argparse
import pickle
import ggml
import torch
from transformers import AutoConfig

# Define the command-line arguments
parser = argparse.ArgumentParser(description='Determine the type of a model file.')
parser.add_argument('--model_path', type=str, help='path to the model file')

# Parse the command-line arguments
args = parser.parse_args()

# If the model path is not passed as a command-line argument, prompt the user to enter it
if not args.model_path:
    args.model_path = input("Please enter the path to the model file: ")

# Set the device to cuda if available, otherwise cpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the file as a pickle model
try:
    with open(args.model_path, 'rb') as f:
        pickle.load(f)
    print(f"{args.model_path} is a pickle model.")
except:
    pass

# Load the file as a PyTorch model
try:
    model = torch.load(args.model_path, map_location=device)
    if isinstance(model, dict):
        if all(k.startswith('module.') for k in model.keys()):
            print(f"{args.model_path} is a pre-trained PyTorch model.")
        elif any(k.startswith('transformer.') for k in model.keys()):
            print(f"{args.model_path} is a transformer-based PyTorch model.")
        else:
            print(f"{args.model_path} is a PyTorch model.")
        if isinstance(model, torch.nn.Module):
            print(f"Model type: {type(model)}")
except:
    pass

try:
    model = ggml.load(args.model_path)
    print(f"GGML Model type: {type(model)}")
except Exception as e:
    print(e)
    pass


# Load the file as a Hugging Face Transformers model
try:
    config = AutoConfig.from_pretrained(args.model_path)
    print(f"{args.model_path} is a Hugging Face Transformers pre-trained model.")
    print(f"Model type: {config.model_type}")
except:
    pass

# If none of the above work, print an error message
if not any((locals().get('model'), locals().get('pickle'), locals().get('config'))):
    print(f"Could not determine the type of {args.model_path}.")