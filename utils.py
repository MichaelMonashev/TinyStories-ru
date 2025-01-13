from typing import Union

class Vocab():
    def __init__(self, vocab:dict[str,int] = {}):
        self.vocab = vocab # str -> int
        return

    def __len__(self) -> int:
        return len(self.vocab)

    def __contains__(self, token: str) -> bool:
        return self.vocab.__contains__(token)

    def __getitem__(self, token: str) -> int:
        return self.vocab[token]

    def __str__(self) -> str:
        return str(self.vocab)

    def items(self) -> list[tuple[str, int]]:
        return self.vocab.items()

    def make(self, text: str) -> int:
        chars = set(text)
        for char in chars:
            if char not in self.vocab:
                self.vocab[char] = len(self.vocab)

class Tokenizer():
    def __init__(self, vocab):
        self.vocab = vocab
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        return

    # text -> tokens
    def __call__(self, text: str, length:int=0, padding_side:str = 'right', pad_token:Union[str, int] = ' ', truncation_side:str = 'right') -> list[int]:
        #делим на символы
        chars = list(text)

        # преобразуем в токены
        tokens = [self.vocab[char] for char in chars]

        if length>0:
            tokens =  self.pad(tokens, length, padding_side, pad_token)
            tokens =  self.truncate(tokens, length, truncation_side)

        return tokens

    # tokens -> tokens
    def pad(self, tokens: list[int], length: int, padding_side:str = 'right', pad_token:Union[str, int] = ' ') -> list[int]:
        if type(pad_token) is str:
            pad_token = self.vocab[pad_token]

        # дополняем проблемами
        if len(tokens)<length:
            pad_len = length-len(tokens)
            if padding_side == 'right':
                tokens += [pad_token]*pad_len
            else:
                tokens = [pad_token]*pad_len + tokens

        return tokens

    # tokens -> tokens
    def truncate(self, tokens: list[int], length: int, truncation_side:str = 'right') -> list[int]:
        # обрезаем лишнee
        if len(tokens)>length:
            if truncation_side == 'right':
                tokens = tokens[:length]
            else:
                tokens = tokens[-length:]

        return tokens

    # tokens -> text
    def decode(self, tokens:list[int]) -> str:
        text = [self.inv_vocab[t] for t in tokens]
        return ''.join(text)