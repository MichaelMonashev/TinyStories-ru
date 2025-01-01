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
    def __call__(self, text: str, length: int) -> list[int]:
        # дополняем справа проблемами
        text += ' '*length

        # обрезаем лишний текст
        text = text[:length]

        #делим на символы
        chars = list(text)

        # преобразуем в токены
        tokens = [self.vocab[char] for char in chars]

        return tokens

    # tokens -> text
    def decode(self, tokens:list[int]) -> str:
        text = [self.inv_vocab[t] for t in tokens]
        return ''.join(text)