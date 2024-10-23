import torch
from transformers import GPT2LMHeadModel, GPT2Config
from torch.utils.data import Dataset, DataLoader
from custom_tokenizer import CustomTokenizer
from datasets import load_dataset


# 2. 데이터셋 준비 (간단한 예시)
class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        input_ids = self.tokenizer.encode(text, add_special_tokens=True)

        # 패딩
        input_ids = input_ids[:self.max_length] + [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
        attention_mask = [1 if id != self.tokenizer.pad_token_id else 0 for id in input_ids]

        return {"input_ids": torch.tensor(input_ids), "attention_mask": torch.tensor(attention_mask)}

# 3. GPT-2 모델 구성
def create_gpt2_model(vocab_size):
    configuration = GPT2Config(vocab_size=vocab_size)
    model = GPT2LMHeadModel(configuration)
    return model

# 4. 데이터 로더 설정
def create_data_loader(texts, tokenizer, max_length, batch_size):
    dataset = TextDataset(texts, tokenizer, max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

# 5. 학습 함수 정의
def train_gpt2(model, dataloader, optimizer, device, num_epochs):
    model = model.to(device)
    model.train()
    for epoch in range(num_epochs):
        for batch in dataloader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss
            print(f"loss: {loss}")
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

# 6. 텍스트 생성 함수
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

# 7. 학습 실행 예제 및 텍스트 생성
if __name__ == "__main__":
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
    model = create_gpt2_model(vocab_size=len(tokenizer.vocab))
    dataloader = create_data_loader(texts, tokenizer, max_length=50, batch_size=2)

    # Optimizer 및 학습 설정
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # 학습 시작
    print("3. begin training")
    train_gpt2(model, dataloader, optimizer, device, num_epochs=1)
    print(" - complete")

    # 학습된 모델로 텍스트 생성
    prompt = "입냄새 안나나?"
    generated_texts = generate_text(model, tokenizer, prompt, max_length=50, num_return_sequences=3)
    
    # 생성된 텍스트 출력
    for i, text in enumerate(generated_texts):
        print(f"\nGenerated Text {i+1}: {text}")

    model.save_pretrained("./my_finetuned_model")