import torch
from transformers import GPT2LMHeadModel
from custom_tokenizer import CustomTokenizer
from llm_util import LlmUtil


# device 선택
# 생성은 그냥 cpu를 쓰도록 설계한다.
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
print(f"device: {device}")

# 커스텀 토크나이저 로드
tokenizer = CustomTokenizer('./my_finetuned_tokenizer')

# 모델 준비
model = GPT2LMHeadModel.from_pretrained("./my_finetuned_model")

# 학습된 모델로 텍스트 생성
print("generating text...")
prompt = "한우는 어느나라꺼야? 짧게 대답해줘."
generated_texts = LlmUtil.generate_text(
    model, tokenizer, prompt, max_length=100, num_return_sequences=1, skip_special_tokens=False)

# 생성된 텍스트 출력
for i, text in enumerate(generated_texts):
    print(f"\nGenerated Text {i+1}: {text}")
