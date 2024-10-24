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
        texts, tokenizer, max_length=1000, batch_size=2)

    # 학습 시작
    print("1. begin training")
    LlmUtil.train_gpt2(model, dataloader, optimizer, device, num_epochs=3)
    print(" - complete")

    print("2. saving model")
    model.save_pretrained("./my_finetuned_model")
    print(" - complete")
