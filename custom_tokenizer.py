from kiwipiepy import Kiwi
import os
import json


class CustomTokenizer:
    def __init__(self, dir_path=''):
        self.kiwi = Kiwi()
        self.load_config(dir_path)

    def _generate_related_data(self):
        self.special_tokens_set = set(self.special_tokens)
        self.inverse_vocab = {self.vocab[token]: token for token in self.vocab}

    def load_config(self, dir_path, special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"]):
        if dir_path == '':
            for st in special_tokens:
                self.kiwi.add_user_word(st)
            self.special_tokens = special_tokens
            self.vocab = {token: idx for idx,
                          token in enumerate(special_tokens)}
            self._generate_related_data()
            return

        # load config
        file_path = os.path.join(dir_path, "config.json")

        with open(file_path, 'r', encoding='utf-8') as json_file:
            json_data = json.load(json_file)
            self.special_tokens = json_data['special_tokens']
            self.vocab = json_data['vocab']
            self._generate_related_data()
            

    def save_config(self, dir_path):
        os.makedirs(dir_path, exist_ok=True)

        file_path = os.path.join(dir_path, "config.json")

        with open(file_path, 'w', encoding='utf-8') as json_file:
            json.dump({"special_tokens": self.special_tokens, "vocab": self.vocab}, json_file, ensure_ascii=False, indent=4)

    def build_vocab(self, texts):
        """텍스트 목록으로부터 어휘 사전을 생성합니다."""
        for text in texts:
            tokens = self.tokenize(text)
            for token in tokens:
                if token not in self.vocab:
                    idx = len(self.vocab)
                    self.vocab[token] = idx
                    self.inverse_vocab[idx] = token

    def tokenize(self, text):
        """텍스트를 형태소 분석하여 토큰으로 분할합니다."""
        tokens = []
        for token in self.kiwi.tokenize(text):
            if token.form in self.special_tokens_set:
                tokens.append(token.form)
            else:
                tokens.append(token.form+"/"+token.tag)
        return tokens

    def encode(self, text, add_special_tokens=True):
        """텍스트를 인덱스 리스트로 변환합니다."""
        tokens = self.tokenize(text)
        token_ids = [self.vocab.get(token, self.vocab["[UNK]"])
                     for token in tokens]

        if add_special_tokens:
            token_ids = [self.vocab["[BOS]"]] + \
                token_ids + [self.vocab["[EOS]"]]

        return token_ids

    def decode(self, token_ids, skip_special_tokens=True):
        """인덱스 리스트를 텍스트로 변환합니다."""
        tokens = [self.inverse_vocab.get(idx, "[UNK]") for idx in token_ids]
        if skip_special_tokens:
            tokens = [t for t in tokens if t not in self.special_tokens]
        
        tuple_list = []
        for t in tokens:
            tuple_list.append((t.split("/")[0], t.split("/")[1]))
        return self.kiwi.join(tuple_list)

    @property
    def pad_token_id(self):
        return self.vocab["[PAD]"]
