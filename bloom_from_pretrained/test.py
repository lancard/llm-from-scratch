from transformers import BloomForCausalLM, BloomTokenizerFast

# 모델과 토크나이저 로드
model_name = "bigscience/bloom-560m"
tokenizer = BloomTokenizerFast.from_pretrained(model_name)
model = BloomForCausalLM.from_pretrained(model_name)

# 텍스트 입력
input_text = "한국말도 할 줄 알아?"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids

# 텍스트 생성
output = model.generate(input_ids, max_length=100, num_return_sequences=1)

# 생성된 텍스트 디코딩
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

print(generated_text)
