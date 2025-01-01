import torch
import os
import glob
import json

from utils import Vocab

def main(args = {}):
    print(args)

    # путь к директории с датасетом
    dataPath = 'data'

    # путь к словарю
    vocabPath = 'vocab.json'

    # загружаем/создаём словарь
    try:
        with open(vocabPath, 'r') as f:
            data = json.load(f)
            vocab = Vocab(data)

    except (FileExistsError, FileNotFoundError):
        vocab = Vocab()
        files = glob.glob(dataPath + "/*.json")
        for filename in files:
            print("Loading", filename,"\r")
            with open(filename, 'r') as f:
                data = json.load(f)

            text = data['story']
            vocab.make(text)

        with open(vocabPath, 'w') as f:
            json.dump(vocab.vocab, f, ensure_ascii=False)

    print(vocab)

if __name__ == '__main__':
    main()