import torch

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

    def make(self, text: str) -> int:
        chars = set(text)
        for char in chars:
            if char not in self.vocab:
                self.vocab[char] = len(self.vocab)
