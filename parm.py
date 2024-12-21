import torch
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel
import os

your_model = input('model:')

# Load Tokenizer & Model
koGPT2_TOKENIZER = PreTrainedTokenizerFast.from_pretrained(
    "skt/kogpt2-base-v2", 
    bos_token="</s>", 
    eos_token="</s>", 
    unk_token="<unk>", 
    pad_token="<pad>", 
    mask_token="<unused0>"
)
model = GPT2LMHeadModel.from_pretrained("skt/kogpt2-base-v2")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 저장된 모델 불러오기
checkpoint_path = "saved_models/" + your_model  # 저장된 모델 경로
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print("Checkpoint loaded successfully!")

# 파라미터 출력 및 저장 함수
def save_model_parameters(model, output_file="model_parameters.txt"):
    with open(output_file, "w", encoding="utf-8") as f:
        for name, param in model.named_parameters():
            f.write(f"Parameter name: {name}\n")
            f.write(f"Parameter shape: {param.shape}\n")
            f.write(f"Parameter values: {param.data}\n\n")
    print(f"Model parameters saved to: {output_file}")

# 전체 파라미터 확인
def print_model_parameters(model):
    for name, param in model.named_parameters():
        print(f"Parameter name: {name}")
        print(f"Parameter shape: {param.shape}")
        print(f"Parameter values: {param.data}\n")

# 사용 예시
if __name__ == "__main__":
    # 파일로 파라미터 저장
    save_model_parameters(model, "model_parameters.txt")
    
    # 터미널에 출력
    print_model_parameters(model)
