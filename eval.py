import torch
from transformers import GPT2LMHeadModel, GPT2Config
from torch.utils.data import Dataset, DataLoader
from custom_tokenizer import CustomTokenizer
from datasets import load_dataset

# 1. GPT-2 모델 구성
def create_gpt2_model():
    model = GPT2LMHeadModel.from_pretrained("./my_finetuned_model")
    return model

# 2. 텍스트 생성 함수
def generate_text(model: GPT2LMHeadModel, tokenizer, prompt, max_length=50, num_return_sequences=1):
    model.eval()
    input_ids = torch.tensor([tokenizer.encode(prompt, add_special_tokens=True)]).to(model.device)
    
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=max_length,
            num_return_sequences=num_return_sequences,
            no_repeat_ngram_size=2,
            repetition_penalty=2.0,
            top_p=0.95,
            temperature=1.0,
            do_sample=True,
            top_k=50,
            eos_token_id=tokenizer.vocab["[EOS]"],
            pad_token_id=tokenizer.pad_token_id
        )
    
    generated_texts = [tokenizer.decode(output_seq.tolist()) for output_seq in output]
    return generated_texts

# 3. 학습 실행 예제 및 텍스트 생성
if __name__ == "__main__":

    # device 선택
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # Kiwi 초기화 및 커스텀 토크나이저 생성
    tokenizer = CustomTokenizer()

    print("1. loading dataset")
    dataset = load_dataset("csv", data_files="formatted_data.csv")["train"] # .select(range(10))
    print(f"Len: {len(dataset)}")
    print(" - complete")

    texts = [d['text'] for d in dataset]

    # 어휘 사전 생성
    print("2. building vocab")
    tokenizer.build_vocab(texts)
    # print(tokenizer.vocab)
    print(" - complete")

    # 모델, 데이터 로더 준비
    model = create_gpt2_model()

    # 학습된 모델로 텍스트 생성
    prompt = "[BOS] 입냄새 안나나? [PAD]"
    generated_texts = generate_text(model, tokenizer, prompt, max_length=50, num_return_sequences=3)
    
    # 생성된 텍스트 출력
    for i, text in enumerate(generated_texts):
        print(f"\nGenerated Text {i+1}: {text}")
