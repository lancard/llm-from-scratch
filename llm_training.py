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

    print("1. loading dataset")
    dataset = load_dataset("csv", data_files="formatted_data.csv")[
        "train"]  # .select(range(10))
    texts = [d['text'] for d in dataset]
    print(f"Len: {len(texts)}")
    print(" - complete")

    # 어휘 사전 생성
    print("2. merging vocab")
    tokenizer.build_vocab(texts)
    print(" - complete")

    # 모델, 데이터 로더, 옵티마이저 준비
    model = LlmUtil.create_gpt2_model(vocab_size=len(tokenizer.vocab))
    dataloader = LlmUtil.create_data_loader(
        texts, tokenizer, max_length=50, batch_size=2)
    optimizer = LlmUtil.create_optimizer(model)

    # 학습 시작
    print("3. begin training")
    LlmUtil.train_gpt2(model, dataloader, optimizer, device, num_epochs=10)
    print(" - complete")

    print("4. saving model")
    tokenizer.save_config('./my_finetuned_tokenizer')
    model.save_pretrained("./my_finetuned_model")
    print(" - complete")