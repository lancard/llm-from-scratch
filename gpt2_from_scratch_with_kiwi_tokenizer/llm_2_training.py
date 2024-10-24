import torch
from custom_tokenizer import CustomTokenizer
from datasets import load_dataset
from llm_util import LlmUtil

if __name__ == '__main__':
    # device 선택
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # 커스텀 토크나이저 생성 및 config 로드
    tokenizer = CustomTokenizer('./my_finetuned_tokenizer')

    # 토크나이저 시에 썼던 texts 로딩
    texts = []

    # load data file when used by tokenizer
    with open("./my_finetuned_tokenizer/data.txt", 'r', encoding='utf-8') as data_file:
        str = data_file.read()
        texts = str.split("ζ")

    # 모델, 옵티마이저 준비
    # if os.path.exists('./my_finetuned_model'):
    #    model = LlmUtil.load_gpt2_model('./my_finetuned_model')
    # else:
    model = LlmUtil.create_gpt2_model(vocab_size=len(tokenizer.vocab))
    optimizer = LlmUtil.create_optimizer(model)
    dataloader = LlmUtil.create_data_loader(
        texts, tokenizer, max_length=1000, batch_size=3)

    # 학습 시작
    print("begin training")
    num_epochs = 3
    for epoch in range(num_epochs):
        loss = LlmUtil.train_gpt2_once(model, dataloader, optimizer, device)
        print(f" - saved epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")
        model.save_pretrained("./my_finetuned_model")
    print(" - complete")