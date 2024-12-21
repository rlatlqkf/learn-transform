import pandas as pd
import json
from collections import Counter

your_file = input("file:")

def create_vocab_from_csv(csv_file, output_vocab_path="vocab.json"):
    """
    CSV 파일의 Q와 A 데이터를 바탕으로 단어장을 생성하고 저장합니다.

    Args:
        csv_file (str): CSV 파일 경로
        output_vocab_path (str): 저장될 단어장 JSON 파일 경로
    """
    # Step 1: CSV 파일 읽기
    df = pd.read_csv(csv_file)
    if "Q" not in df.columns or "A" not in df.columns:
        raise ValueError("CSV 파일에 'Q'와 'A' 열이 필요합니다.")
    
    # Step 2: Q와 A 열의 텍스트 데이터를 결합
    corpus = " ".join(df["Q"].dropna()) + " " + " ".join(df["A"].dropna())
    words = corpus.split()
    
    # Step 3: 단어 빈도수 계산 및 단어장 생성
    word_counts = Counter(words)
    vocab = {word: idx for idx, (word, _) in enumerate(word_counts.items(), start=1)}
    vocab["<PAD>"] = 0  # 패딩 토큰 추가

    # Step 4: 단어장 저장
    with open(output_vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=4)
    print(f"단어장이 '{output_vocab_path}'에 저장되었습니다.")

    return vocab

def tokenize_sentence(sentence, vocab):
    """
    문장을 단어장에 맞춰 인덱스로 변환합니다.

    Args:
        sentence (str): 입력 문장
        vocab (dict): 단어장 (단어 -> 인덱스)

    Returns:
        list: 토큰화된 인덱스 리스트
    """
    return [vocab.get(word, vocab["<PAD>"]) for word in sentence.split()]

if __name__ == "__main__":
    # 입력 파일 및 설정
    csv_file = "data/"+your_file+".csv"  # CSV 파일 경로
    output_vocab_path = "vocab.json"

    # Step 1: 단어장 생성
    vocab = create_vocab_from_csv(csv_file, output_vocab_path)

    # Step 2: 단어장 활용 예시
    test_sentence = "안녕하세요 답변을 확인합니다"
    tokenized_sentence = tokenize_sentence(test_sentence, vocab)
    print("토큰화된 문장:", tokenized_sentence)
