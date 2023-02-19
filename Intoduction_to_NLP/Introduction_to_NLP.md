# Обобщённая формулировка задания
По ссылке находится произведение русской классики. Задача состоит в том, чтобы применить RNNMorph для анализа произведения. Необходимо будет вывести колчество предложений, количество состоящих только из букв токенов и несколько других метрик. \
Перед выполнением индивидуального задания необходимо разобраться с обучающим примером и только потом переходить к индивидуальному заданию. Комметарии составлены мной.

## Задание

### Импортируем все необходимые библиотеки
```python
from rnnmorph.predictor import RNNMorphPredictor  # позволяет использовать библиотеку pymorphy2 для русского языка, работает на рекурентных нейронных сетях
import urllib.request  # помогает открыть URL
from bs4 import BeautifulSoup  # библиотека для извлечения данных из HTML и XML файлов, удаления тегов
from nltk.tokenize import sent_tokenize, word_tokenize  # функции позволяют проводить токенизацию по словам и по предложениям соответственно
import nltk  # Natural Language Toolkit
from tqdm import tqdm  # задание полоски ожидания

from nltk import FreqDist  # функция частотного распределения для эксперимента 
from nltk.corpus import stopwords  # некоторые слова "a", "the" встречаются часто, их можно игнорировать
```
### Задаём ссылку на источник с тектом и создаём объект морфологического анализатора
```python
DATA_URL = "http://az.lib.ru/t/tolstoj_a_k/text_0180.shtml"

predictor = RNNMorphPredictor(language="ru")  # объект анализатора
```
### Скачиваем текст, по которому будет дано задание
```python
opener = urllib.request.URLopener({})  # базовый класс для открытия и чтения URLs
resource = opener.open(DATA_URL)  

raw_text = resource.read().decode(resource.headers.get_content_charset())  # Получаем текст заголовка с html тегами

# можно вывести raw_text и увидеть, что текст содержит html теги, от которых необходимо избавиться. Поможет библиотека bs4
print(raw_text)  # теги с заголовка страницы
```
### Удаляем теги
```python
soup = BeautifulSoup(raw_text, features="html.parser")  # создаём объект, содержащий документ как вложенную структуру данных

# kill all script and style elements
for script in soup(["script", "style"]):  
    script.extract()    # удаляет тег

# get text
cleaned_text = soup.get_text()  # возвращает только обычный текст из документа или тега
print(cleaned_text[:200])
```
### С помощью библиотеки NLTK разбиваем текст на предложения и токены
```python
nltk.download('punkt')  # скачиваем пакет с пунктуацией для модуля nltk

tokenized_sentences = [word_tokenize(sentence) for sentence in sent_tokenize(cleaned_text)]  # токенизируем на предложения, а затем каждое на слова
```
### С помощью стокового метода .isalpha из стандартной библиотеки сделаем так, чтобы остались только буквенные токены
```python
predictions = [[pred.normal_form for pred in sent] 
               for sent in tqdm(predictor.predict_sentences(sentences=tokenized_sentences), "sentences") ]  # используем словарный подход для слов в предложении
print(predictions[-11:-10])  # Сейчас видно, что токены типа "точка", "запятая" и тд пока присутствуют в предложениях. От них нужно избавиться
# создаём 2 списка, и 2 цикла, проходим по предложению, по словам в нём и проверяем, является ли слово словом
print(len(predictions))
non_uniq_tokens = [word for sentence in predictions for word in sentence]  # сделаем из двумерного массива одномерный
y = list(filter(str.isalpha, non_uniq_tokens))  # с помощью встроенной функции filter отфильтруем слова. Поскольку это итератор, то необходимо превратить в список для удобства работы
print(len(y))
```
### Используя ```y```, стоп-слова для русского языка из библиотеки `nltk`, `nltk.FreqDist` вычислить какую долю среди 100 самых частотных токенов в произведении занимают токены, не относящиеся к стоп словам.
```python
nltk.download("stopwords")  # скачиваем пакет со стоп-словами и создаём из них множество
STOPWORDS = set(stopwords.words("russian"))
stopwords.words("russian")[:5] #Пример стоп слов

fdist = FreqDist()  # задаём объект
lenght = 100  # сколько самых частостных слов надо
for word in y:  # из оф. документации создаём словарь, где ключ - слово, а значение - количество повторений
    fdist[word.lower()] += 1
List = fdist.most_common()[0:lenght]  # определяем lenght самых частовстречающихся слов, получаем кортежи
count = 0  # в цикле проверяем, входит ли первый элемент, т.е. слово, из каждого кортежа во множество стоп слов, считаем долю
for elem in List:
  if elem[0] in STOPWORDS:
    count += 1
(lenght - count) / lenght

```

### Вычислить, сколько токенов встречается в тексте строго больше 50 раз
```python
u = list(filter(lambda x: x[1] > 50, List))
len(u)
```


### Теперь необходимо выполнить задание для своего варианта и с другими числовыми данными. Алгоритм тот же, потому чистый код приведён ниже
Ссылка http://az.lib.ru/l/leskow_n_s/text_0246.shtml \
Вывести количество предложений, токенов. Доля 150 самых частотных слов, не входящих в стоп-слова. Количество токенов, встречающихся больше 10 раз
```python
DATA_URL = "http://az.lib.ru/l/leskow_n_s/text_0246.shtml"

from rnnmorph.predictor import RNNMorphPredictor
predictor = RNNMorphPredictor(language="ru")
import urllib.request

opener = urllib.request.URLopener({})
resource = opener.open(DATA_URL)
raw_text = resource.read().decode(resource.headers.get_content_charset()) 

from bs4 import BeautifulSoup
soup = BeautifulSoup(raw_text, features="html.parser")


for script in soup(["script", "style"]):
    script.extract()    


cleaned_text = soup.get_text()


from nltk.tokenize import sent_tokenize, word_tokenize
import nltk
nltk.download('punkt')


from tqdm import tqdm
predictions = [[pred.normal_form for pred in sent] 
               for sent in tqdm(predictor.predict_sentences(sentences=tokenized_sentences), "sentences") ]
predictions[-11:-10] 

print(len(predictions))

non_uniq_tokens = [word for sentence in predictions for word in sentence]
y = list(filter(str.isalpha, non_uniq_tokens))
print(len(y))


import nltk
from nltk import FreqDist
from nltk.corpus import stopwords
nltk.download("stopwords")
STOPWORDS = set(stopwords.words("russian"))


fdist = FreqDist()
lenght = 150
for word in y:
    fdist[word.lower()] += 1

List = fdist.most_common()[0:lenght]
count = 0
for elem in List:
  if elem[0] in STOPWORDS:
    count += 1
print((lenght - count) / lenght)


u = list(filter(lambda x: x[1] > 10, List))
len(u)
```
