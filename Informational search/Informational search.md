# Задание. Использовать в качестве документов тексты запросов из набора текста об аэронавтике CRANFIELD и найти пара наиболее близких к указанному запросу документа.
Текст запроса: <b>electronic computer</b>\
Необходимо выполнить следующие подпункты:
+ Ввести индекс наиболее близкого документа согласно мере Жаккара
+ Ввести значение коэффициента Жаккара для наиболее близкого документа
+ Ввести индекс второго по близости документа согласно мере Жаккара и его значение

Для второго запроса <b>surface heat</b> необхожимо:
+ Вывести индекс 2х наиболее близких документов относительно косинусного расстояния и значения расстояния для них

### Для начала скачиваем классический набор данных - набор текстов об аэронавтике CRANFIELD
Воспользуемся модулем `wget` для скачивания набора. \
С помощью команды для работы со сжатыми файлами `tar` и методом этой команды `xvf`, где х - извлечение архима, v - отображение подробной информации, f - указание имени файла, извлечём файл `cran.tar.gz`. \

```python
! wget -q http://ir.dcs.gla.ac.uk/resources/test_collections/cran/cran.tar.gz
! tar -xvf cran.tar.gz
! rm cran.tar.gz*  # удаляем архив после извлечения файлов
```

### Берём только сами запросы (это будут наши документы)
С помощью `grep`, которая ищет заданный шаблон в файле и печатает соответствующие заданным критериям строки
```python
! grep -v "^\." cran.qry > just.qry
! head -3 just.qry  # вывести первые три строки только что созданного файла с нужными строками
```

### Теперь все строки просто объединим в один список
```python
raw_query_data = [line.strip() for line in open("just.qry", "r").readlines()]  # читаем файл, c помошью r и readlines, получаем список, для каждого элемента которого убираем ненужные пробелы
query_data = [""]

for query_part in raw_query_data:
  query_data[-1] += query_part + ""  # добавляем в конец списка элемент
  if query_part.endswith("."):  # если предложение закончено, то добавляем пустой элемент
    query_data.append("")

query_data[:2] #Выведем пару документов для примера
```
### Составим запросы к нашим документам
```python
QUERIES = ['electronic computer', 'surface heat']
```
### Boolean retrieval
Представим каждый документ как "битовую маску": вектор размером со словарь, в котором на каждой позиции единица, если в документе есть соответсвующий терм, и ноль, если терма нет
```python
# для совпадения ответов с проверяющей системой рекомендуется установить scikit-learn версии 0.22.2.post1
! pip install -q scikit-learn==0.22.2.post1

from  sklearn.feature_extraction.text import CountVectorizer

encoder = CountVectorizer(binary=True)  # создаём сущность объекта класса CountVectorizer
encoded_data = encoder.fit_transform(query_data)  # обучаем алгоритм на тренировочных данных и задаём размерность под наши данные
encoded_queries = encoder.transform(QUERIES)  # применяем алгоритм к уже тестовым данным с тренировочными параметрами
list(encoder.vocabulary_)[:3]  # метод для маппинга, где ключами становятся термы, а значениями - индексы
```
### Посмотрим на представление первого предложения
```python
id2term = {idx: term for term, idx in encoder.vocabulary_.items()}  # вытаскиваем из словаря индекс и терм, создаём словарь с ключами в виде индексов, значениями - словами

non_zero_values_ids = encoded_data[0].nonzero()[1]  # получить индексы ненулевых элементов, поскольку encoded_data возвращает кортеж (0, N), где N - индекс элемента

terms = [id2term[idx] for idx in non_zero_values_ids]  # по ключу через индекс создаём список из слов
terms

```
## Задание 0
Теперь для каждого из данных запросов `QUERIES` найдём ближайший для него документ из `query_data` по сходству Жаккара. Есть более эффективные способы это сделать, но вам требуется реализовать расстояние Жаккара и далее применить его к нашим данным.
```python
import numpy as np 

def jaccard_sim(vector_a: np.array, vector_b: np.array) -> float:
   """
     Сходство или коэффициент Жаккара: отношение мощности пересечения
    к мощности объединения
   """
    ins = np.logical_and(vector_a, vector_b)  # вычисляем пересечение
    uni = np.logical_or(vector_a,vector_b)  # вычисляем объединение
    simil = ins.sum() / float(uni.sum())  # считаем коэф. Жаккара
    return simil
#Проверка, что функция работает правильно
assert jaccard_sim(np.array([1, 0, 1, 0, 1]), np.array([0, 1, 1, 1, 1])) == 0.4
print(jaccard_sim(np.array([1, 1, 1, 0, 1]), np.array([0, 1, 1, 1, 1])))
```
Здесь документы представлены так же, как строки в матрице термов-документов. Каждая ячейка вектора отвечает за наличие/отсутствие конкретного элемента (например, слова-терма, когда у нас в словаре всего 5 слов). В первом случае их три, во втором — четыре. Объединение — все пять возможных элементов. Пересечение — два. Отсюда и 0.4.

## Задание 1. Вычислите для каждого запроса самые близкие документы
```python
for q_id, query in enumerate(encoded_queries):  # кортеж с индексом и значением с помощью enumerate
  # приводим к нужному типу
  query = query.todense().A1  # получим матрицу-строку
  docs = [doc.todense().A1 for doc in encoded_data]  # аналогично для документов из данных
  # вычисляем коэфф. Жаккара
  id2doc2similarity = [(doc_id, doc, jaccard_sim(query, doc)) for doc_id, doc in enumerate(docs)]  
  # сортируем по коэффициенту Жаккара
  closest = sorted(id2doc2similarity, key=lambda x: x[2], reverse=True)
  
  print("Q: %s\nFOUND:" % QUERIES[q_id])  # выводим запрос
  # выводим по 3 наиболее близких документа для каждого запроса
  for closest_id, _, sim in closest[:3]:
    print("    %d\t%.2f\t%s" %(closest_id, sim, query_data[closest_id]))
```
Если выполнить код, то будет видно, что для первого запроса `electronic computer` будет два близких документа под индесами 15 и 128 с коэф. Жаккара 0.12 и 0.08, что является не очень высоким показателем. Для запроса `surface heat` будут документы 45, 8, 94 с коэффициентами 0.14, 0.11, 0.10, что является уже результатом лучше.
## VSM
Попробуем теперь сделать то же, но с tf-idf и косинусным расстоянием. Мы сделаем всё опять "руками", но "в реальной жизни" лучше использоватьесть эффективные реализации cosine distance, например, из библиотеки scipy.
```python
from  sklearn.feature_extraction.text import TfidfVectorizer

tfidf_encoder = TfidfVectorizer()  # снова создаём экземпляр класса
tfidf_encoded_data = tfidf_encoder.fit_transform(query_data)  # возвращает матрицу термов-документов
tfidf_encoded_queries = tfidf_encoder.transform(QUERIES)

list(tfidf_encoder.vocabulary_)[:3]
```
## Задание 2. Реализоват косинусное расстояние
```python
from numpy import linalg, dot

def cosine_distance(vector_a: np.array, vector_b: np.array) -> float:
    """
    Косинусное расстояние: единица минус отношение скалярного произведения
    на произведение L2-норм (подсказка: в numpy такие нормы есть)
    """


    return  1 - dot(vector_a,vector_b) / (linalg.norm(vector_a) * linalg.norm(vector_b))
#Проверка, что функция работает правильно
assert cosine_distance(np.array([1, 0, 1, 1, 1]), np.array([0, 0, 1, 0, 0])) == 0.5
```
### Теперь вычислим ближайшие по косинусному расстоянию между векторными представлениями документов и запросов
```python
for q_id, query in enumerate(tfidf_encoded_queries):
  
  # приводим к нужному типу
  query = query.todense().A1
  docs = [doc.todense().A1 for doc in tfidf_encoded_data]
  # Косинусное расстояние
  id2doc2similarity = [(doc_id, doc, cosine_distance(query, doc)) \
                       for doc_id, doc in enumerate(docs)]
  # сортируем по нему
  closest = sorted(id2doc2similarity, key=lambda x: x[2])
  
  print("Q: %s\nFOUND:" % QUERIES[q_id])
  
  for closest_id, _, sim in closest[:3]:
    print("    %d\t%.2f\t%s" %(closest_id, sim, query_data[closest_id]))
```
Для запроса `surface heat` ближайшие документы 45, 44 и 127 с величинами косинусного расстояния 0.56, 0.76 и 0.76 соответственно, а для `electronic computer` 15, 128 с величинами 0.53, 0.76
## Вывод

Если сравнить оба метода, то для `electronic computer` ближайшие документы совпали - 15 и 128. А вот для `surface heat` совпал только самый ближайший документ - 45. Полагаю, такую разницу можно объяснить редкостью запроса electronic computer для данного сета документов. Поскольку у нас дата сет с текстами, связанными с аэронавтикой, то `surface heat` просто встречает в большем количестве документов
