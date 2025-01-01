# python3 -m pip install bitsandbytes accelerate transformers optimum

import re
import argparse
import os
import hashlib
import random
import time
import sys
import json
import uuid

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Gemma2ForCausalLM

def read_words(filepath):
    words = []
    with open(filepath, 'r', encoding='UTF-8') as file:
        for line in file:
            if line.startswith("#"):
                continue

            words.append(line.strip())

    return words

def main(args):

    print(args)

    result_filename = None

    adjectives = read_words(os.path.join(args.dataset_path,'adjectives.txt'))
    nouns = read_words(os.path.join(args.dataset_path,'nouns.txt'))
    verbs = read_words(os.path.join(args.dataset_path,'verbs.txt'))

    print("Adjectives:", len(adjectives))
    print("Nouns:", len(nouns))
    print("Verbs:", len(verbs))

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    #model_id = "Qwen/Qwen2.5-72B-Instruct"
    #model_id = "Qwen/Qwen2.5-32B-Instruct"
    model_id = "google/gemma-2-27b-it"

    dtype = torch.bfloat16
    device =  torch.device('cuda')

    # load model in 4-bit
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=dtype
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=args.huggingface_assecc_token)
    #model = AutoModelForCausalLM.from_pretrained(
    model = Gemma2ForCausalLM.from_pretrained(
                                                model_id,
                                                quantization_config=quantization_config,
                                                torch_dtype=dtype,
                                                token=args.huggingface_assecc_token,
                                                ).to(device)

    model.eval()

    model_name = model_id+" bnb_4bit"

    number_stories = 0
    while True:

        adjective = random.choice(adjectives)
        noun = random.choice(nouns)
        verb = random.choice(verbs)

        dialog = random.choice(['', 'содержать диалог'])

        nun_of_personages = ''
        if len(dialog)>0:
            nun_of_personages = random.choice(['', 'иметь двух персонажей', 'иметь трёх персонажей'])
        else:
            nun_of_personages = random.choice(['', 'иметь одного персонажа', 'иметь двух персонажей', 'иметь трёх персонажей'])

        end_type = random.choice(['', 'плохо заканчиваться', 'иметь хороший конец'])
        plot_twist = random.choice(['', 'иметь неожиданный поворот сюжета'])
        moral_value = random.choice(['', 'быть поучительным'])
        fairy_tale = random.choice(['', 'быть похожим на сказку'])

        features_list = []
        for feature in [dialog, nun_of_personages, end_type, plot_twist, moral_value, fairy_tale]:
            if len(feature)>0:
                features_list.append(feature)

        random.shuffle(features_list)

        features = ", ".join(features_list)

        if len(features)>0:
            features = " Рассказ должен иметь следующие особенности: "+features+"."

        #print(features)

        text = '''<start_of_turn>user
Напишите небольшой рассказ размером 3–5 абзацев, в котором используются очень простые слова, которые, скорее всего, поймет трехлетний ребенок. В рассказе следует использовать глагол "'''+verb+'''", существительное "'''+noun+'''" и прилагательное "'''+adjective+'''".'''+features+''' Не забывайте использовать только простые слова, которые маленькие дети легко понимают!
<end_of_turn>
<start_of_turn>model
'''

        #print(text)

        inputs = tokenizer(text, return_tensors="pt")
        input_len = inputs['input_ids'].shape[-1]
        #print(input_len)
        #print(inputs)
        inputs = inputs.to(model.device)

        start = time.perf_counter()
        with torch.inference_mode():
            outputs = model.generate(**inputs, max_new_tokens=512-input_len-10)
        end = time.perf_counter()
        print ('Time:', end-start)

        # удаляем промт
        tokens = outputs[0][input_len:]
        #print(tokens)

        # ищем конец рассказа <end_of_turn>
        end_of_turn_id = tokenizer.convert_tokens_to_ids('<end_of_turn>') # 107
        end_of_turn_pos = (tokens == end_of_turn_id).nonzero()
        #print(end_of_turn_pos)
        #print(end_of_turn_pos.shape[0])

        if end_of_turn_pos.shape[0]==0:
            print("No <end_of_turn>")
            continue

        end_of_turn_pos = end_of_turn_pos[0,0].item()
        #print(end_of_turn_pos)
        tokens = tokens[:end_of_turn_pos]
        #print(tokens)

        story = tokenizer.decode(tokens, skip_special_tokens=False)
        #print(story)
        story = story.strip()
        #print(story)

        # заменеям некоторые символы
        story = story.replace('«', '"')
        story = story.replace('»', '"')
        story = story.replace('…', '...')

        # проверяем, что не попал английский и маркдаун.
        if not bool(re.search('^[\n0123456789абвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ —–!",.:;?-]+$', story)):
            print("Wrong simbols in story:", story)
            continue

        digest = hashlib.sha3_256(story.encode('utf-8'))
        digest = digest.hexdigest()
        #print(digest)

        data = {
            "features": features_list,
            "sha3_256": digest,
            "story": story,
            "adjective": adjective,
            "noun": noun,
            "verb": verb,
            "model": model_name,
        }

        number_stories += 1
        if number_stories > 50000:
            number_stories = 0
            result_filename = None

        if result_filename is None:
            result_filename = str(uuid.uuid4())

        filename = os.path.join(args.dataset_path,'data', result_filename+'.jsonl')

        try:
            with open(filename, 'a+', encoding='utf-8') as f:
                json.dump(data, f, sort_keys=True, ensure_ascii=False)
                f.write('\n')
        except:
            print("Unexpected error", filename, sys.exc_info())
            continue



if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Generate TinyStories-ru dataset.',
        epilog="""Examples:
python3 generate.py --dataset ~/TinyStories-ru
"""
    )
    parser.add_argument(
        '--dataset-path',
        help='Path to dataset. Example: --dataset ~/TinyStories-ru/',
        default='~/TinyStories-ru/')

    parser.add_argument(
        '--huggingface-assecc-token',
        help='Hugging Face token for assecc to models. https://huggingface.co/settings/tokens. Example: --huggingface-assecc-token hf_hjgFSKDRKGVBhfkdSDKBKsdk',
        required=True)

    main(parser.parse_args())