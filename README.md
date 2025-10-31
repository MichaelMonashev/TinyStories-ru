# TinyStories-ru

TinyStories-ru - это русскоязычная версия датасета TinyStories (https://arxiv.org/pdf/2305.07759). Своими глазами проще оценить текст на родном языке, чем на изученном.

Датасет состоит из коротких историй. Все они сгенерированны LLM. Написаны простым детским языком, что позволяет обучать на датасете небольшие языковые модели на доступных GPU.

# Ошибки

Датасет сгенерённый и потому содержит ошибки. Иногда путаются мужской и женский род, склонения, падежи, слова употребляются не к месту и т.д.

Пример сложно обнаруживаемой ошибки "так не говорят":

---
Маша, Коля и маленький щенок Дружок играли в саду. Маша шевелила руками, будто пчела летит. Коля бегал по траве, а Дружок пытался его
догнать.

— Смотри, Дружок, **мой глаз видит бабочку**! — крикнула Маша. Бабочка сидела на цветке и шевелила крыльями.

— Я тоже вижу! — сказал Коля. Он медленно подходил к бабочке, чтобы не спугнуть ее.

Вдруг Дружок залаял и начал быстро бегать по кругу.

— Что случилось, Дружок? — спросила Маша.

Щенок остановился и посмотрел на дерево. На ветке сидел кот, а в его лапах была… бабочка!

— Ой! — воскликнул Коля. — Кот украл бабочку!

Дружок начал лаять еще громче, а Маша и Коля побежали к дереву, чтобы спасти бабочку.

---
Если Вы знаете, как это найти и/или исправить автоматически, напишите, пожалуйста, об этом тут в issues. Pull request с исправлениями или кодом, который находит и/или исправляет ошибки, также приветствуется.

# Вариант использования

```
lass MyDataset(torch.utils.data.Dataset):
    def __init__(self, dir, tokenizer, num_of_predicted_tokens):
        super().__init__()

        self.tokenizer = tokenizer
        self.num_of_predicted_tokens = num_of_predicted_tokens

        self.max_len = 0
        self.min_len = float("inf")

        self.files = glob.glob(dir + "/*.jsonl")
        print("Number of files:", len(self.files))

        self.index = []
        for i, file in enumerate(self.files):
            with open(file, 'r', encoding='utf-8') as f:
                while True:
                    offset = f.tell()
                    line = f.readline()
                    if len(line)==0:
                        break

                    data = json.loads(line)
                    self.index.append([i, offset])

                    text = data['story']
                    text = text.strip()

                    self.max_len = max(self.max_len , len(text))
                    self.min_len = min(self.min_len , len(text))

        print("Number of samples:", len(self.index))
        print("Max text lenght:", self.max_len)
        print("Min text lenght:", self.min_len)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):

        i, offset = self.index[index]
        file = self.files[i]

        with open(file, 'r', encoding='utf-8') as f:
            f.seek(offset)
            data = json.loads(f.readline())

        text = data['story']
        text = text.strip()

        # обрезаем длинные тексты
        text = text[:1500]

        split_pos = len(text)-self.num_of_predicted_tokens

        input = text[split_pos-self.num_of_predicted_tokens:split_pos]
        output = text[split_pos:split_pos+self.num_of_predicted_tokens]

        input_tokens = self.tokenizer(input, length=self.num_of_predicted_tokens) #max_len)
        output_tokens = self.tokenizer(output, length=self.num_of_predicted_tokens)

        # преобразуем в тензор
        input_tokens = torch.LongTensor(input_tokens)
        output_tokens = torch.LongTensor(output_tokens)

        return input_tokens, output_tokens
```
# ToDo

Автоматические поиск и исправление ошибок (например, как описано тут https://t.me/natural_language_processing/125733 , https://t.me/natural_language_processing/125749 ).

Добавить для каждой истории краткое содержание.

Код для валидации пунктуации, грамматики, связности и последовательности текста.

Дедупликация по порогу cos_sim 0.975 с помошью intfloat/multilingual-e5-large (взято из https://huggingface.co/datasets/Vikhrmodels/GrandMaster-PRO-MAX ).
