from custom_tokenizer import CustomTokenizer
from datasets import load_dataset

if __name__ == '__main__':
    # 커스텀 토크나이저 생성 및 config 로드
    tokenizer = CustomTokenizer('./my_finetuned_tokenizer')

    print("1. loading dataset")
    dataset = load_dataset("csv", data_files="./data/train2.csv")["train"]
    texts = [d['text'] for d in dataset]

    dataset = load_dataset("json", data_files="./data/train.jsonl")["train"]
    texts += [f"[BOS] {d['instruction']} [PAD] {d['output']} [EOS]" for d in dataset]

    print(f" - data length: {len(texts)}")
    print(" - complete")

    # 어휘 사전 생성
    print("2. build vocab")
    tokenizer.build_vocab(texts)
    print(f" - token length: {len(tokenizer.vocab)}")
    print(" - complete")

    # texts 파일 저장
    tokenizer.save_config('./my_finetuned_tokenizer', texts)
