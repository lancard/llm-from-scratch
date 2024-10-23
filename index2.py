from transformers import AutoTokenizer, RobertaConfig, RobertaModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
import torch

import kiwipiepy.transformers_addon  # do not delete this statement



# device 선택
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Device: {device}")



tokenizer = AutoTokenizer.from_pretrained("kiwi-farm/roberta-base-64k")
print(f"vocab size: {len(tokenizer)}")

model = RobertaModel(RobertaConfig(vocab_size=len(tokenizer)))
model.resize_token_embeddings(len(tokenizer))

model_size = sum(t.numel() for t in model.parameters())
print(f"LLM token size: {model_size/1000**2:.1f}M parameters")










dataset = load_dataset("csv", data_files="formatted_data.csv")["train"].select(range(10))
print(f"Len: {len(dataset)}")

# test
text = "한우 고기가 먹고 싶어"
tokens = tokenizer.tokenize(text)
print("Original text:", text)
print("Tokens:", tokens)


# 텍스트 데이터를 토큰화
def tokenize_function(examples):
    ret = tokenizer(examples["text"], return_tensors="pt", padding="max_length", truncation=True, max_length=512)
    return ret

tokenized_dataset = dataset.map(tokenize_function, batched=True)



epochs = 1
batch_size = 2
learning_rate = 5e-5
warmup_steps = 100


def collate_fn(batch):
    # 텍스트 데이터의 예시에서는 input_ids를 텐서로 변환
    input_ids = torch.tensor([item['input_ids'] for item in batch])
    return {'input_ids': input_ids}

# 데이터셋 전처리
dataloader = DataLoader(tokenized_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

# 옵티마이저와 스케줄러 설정
optimizer = AdamW(model.parameters(), lr=learning_rate)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=len(dataloader) * epochs)


for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    for batch in dataloader:
        # 입력 데이터를 GPU로 이동
        inputs = batch["input_ids"].to(device)
        
        # 모델의 기울기 초기화
        optimizer.zero_grad()
        
        # 모델 출력 및 손실 계산
        outputs = model(inputs)
        loss = outputs.loss
        
        # 역전파를 통해 기울기 계산
        loss.backward()
        
        # 옵티마이저와 스케줄러로 가중치 업데이트
        optimizer.step()
        scheduler.step()
        
        # 손실 출력
        print(f"Loss: {loss.item()}")

# 모델 저장
model.save_pretrained("./my_finetuned_model")
# tokenizer.save_pretrained("./my_finetuned_model")

# 모델 평가 예시
model.eval()
test_text = "비트코인을 디지털 화폐로 인정한 나라는?"
input_ids = tokenizer.encode(test_text, return_tensors="pt").to(device)
output = model.generate(input_ids, max_length=50, num_return_sequences=1)
decoded = tokenizer.decode(output[0], skip_special_tokens=True)
print(decoded)