import os
import torch
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel

name_model = input('char:')

# Special Tokens
Q_TKN = "<usr>"
A_TKN = "<sys>"
BOS = "</s>"
EOS = "</s>"
MASK = "<unused0>"
SENT = "<unused1>"
PAD = "<pad>"

# GPT2 Tokenizer & Model Initialization
koGPT2_TOKENIZER = PreTrainedTokenizerFast.from_pretrained(
    "skt/kogpt2-base-v2",
    bos_token=BOS,
    eos_token=EOS,
    unk_token="<unk>",
    pad_token=PAD,
    mask_token=MASK,
)
model = GPT2LMHeadModel.from_pretrained("skt/kogpt2-base-v2")

# Load model weights
def load_model_weights(model, path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=True)  # weights_only=True 추가
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

# Generate Response
def generate_response(input_text, model, tokenizer, max_len=40, num_beams=5):
    input_ids = tokenizer.encode(Q_TKN + input_text + SENT, return_tensors="pt")
    input_ids = input_ids[:, -max_len:]
    input_ids = input_ids.to(model.device)

    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            max_length=max_len,
            num_beams=num_beams,
            no_repeat_ngram_size=2,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

# Main Function
if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load Trained Model
    model_save_path = "saved_models/" + name_model
    if os.path.exists(model_save_path):
        print("Loading trained model...")
        load_model_weights(model, model_save_path, device)
    else:
        raise FileNotFoundError(f"Model file not found at {model_save_path}")

    # Start chatbot
    print("챗봇을 시작합니다. 'exit'을 입력하면 종료됩니다.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("챗봇을 종료합니다.")
            break
        response = generate_response(user_input, model, koGPT2_TOKENIZER)
        print(f"Chatbot: {response}")
