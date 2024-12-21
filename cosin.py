import torch
import json
from itertools import product  # 모든 q와 a 조합을 만듭니다.
from torch.nn.functional import cosine_similarity

def load_vocab(vocab_path):
    with open(vocab_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_embeddings(model_path):
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    
    # transformer.wte.weight에서 임베딩 추출
    if 'transformer.wte.weight' in state_dict:
        print("Using embedding key: transformer.wte.weight")
        return state_dict['transformer.wte.weight']
    else:
        raise ValueError("transformer.wte.weight not found in the state_dict.")

def compute_q_a_similarity(embeddings, vocab):
    q_words = vocab['q']
    a_words = vocab['a']

    for q_word in q_words.keys():
        print(f"\n단어 <{q_word}>와의 유사도:")
        idx_q = q_words[q_word]

        # 각 a 단어와의 유사도 계산
        similarities = []
        for a_word in a_words.keys():
            idx_a = a_words[a_word]
            if idx_q < embeddings.size(0) and idx_a < embeddings.size(0):
                vec_q = embeddings[idx_q].unsqueeze(0)
                vec_a = embeddings[idx_a].unsqueeze(0)
                sim = cosine_similarity(vec_q, vec_a).item()
                similarities.append((a_word, sim))
            else:
                print(f"Skipping invalid indices: {q_word} ({idx_q}), {a_word} ({idx_a})")
        
        # 유사도가 높은 순으로 정렬
        similarities = sorted(similarities, key=lambda x: -x[1])
        
        # 정렬된 결과 출력
        for a_word, sim in similarities:
            print(f"{a_word}: {sim:.4f}")

def main():
    files = [
        ("0vocab.json", "model_0.pth"),
        ("1vocab.json", "model_1.pth"),
        ("25vocab.json", "model_25.pth"),
        ("50vocab.json", "model_50.pth"),
        ("75vocab.json", "model_75.pth"),
        ("100vocab.json", "model_100.pth")
    ]
    
    for vocab_path, model_path in files:
        print(f"\nProcessing {vocab_path} and {model_path}")
        
        # 단어장 및 임베딩 불러오기
        vocab = load_vocab(vocab_path)
        embeddings = load_embeddings(model_path)
        
        # 모든 q 단어에 대해 a 단어와 유사도 계산
        compute_q_a_similarity(embeddings, vocab)

if __name__ == "__main__":
    main()
