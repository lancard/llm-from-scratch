import torch
from transformers import GPT2LMHeadModel, GPT2Config
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


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
        input_ids = input_ids[:self.max_length] + \
            [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
        attention_mask = [
            1 if id != self.tokenizer.pad_token_id else 0 for id in input_ids]

        return {"input_ids": torch.tensor(input_ids), "attention_mask": torch.tensor(attention_mask)}


class LlmUtil:
    # GPT2 모델 생성
    def create_gpt2_model(vocab_size):
        configuration = GPT2Config(vocab_size=vocab_size)
        model = GPT2LMHeadModel(configuration)
        return model

    # GPT2 모델 로드
    def load_gpt2_model(dir_path):
        model = GPT2LMHeadModel.from_pretrained(dir_path)
        return model

    # 데이터 로더 설정
    def create_data_loader(texts, tokenizer, max_length, batch_size):
        dataset = TextDataset(texts, tokenizer, max_length)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        return dataloader

    # 옵티마이저 생성
    def create_optimizer(model, lr=5e-5):
        return torch.optim.AdamW(model.parameters(), lr=lr)

    # 학습 함수 정의
    def train_gpt2(model, dataloader, optimizer, device, num_epochs):
        model = model.to(device)
        model.train()
        for epoch in range(num_epochs):
            for batch in tqdm(dataloader):
                optimizer.zero_grad()
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)

                outputs = model(input_ids=input_ids,
                                attention_mask=attention_mask, labels=input_ids)
                loss = outputs.loss
                loss.backward()
                optimizer.step()

            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

    # 텍스트 생성 함수
    def generate_text(model: GPT2LMHeadModel, tokenizer, prompt, max_length=50, num_return_sequences=1):
        model.eval()
        input_ids = torch.tensor(
            [tokenizer.encode(prompt, add_special_tokens=True)]).to(model.device)

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

        generated_texts = [tokenizer.decode(
            output_seq.tolist()) for output_seq in output]
        return generated_texts
