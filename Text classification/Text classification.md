# Задание
На примере классического датасета Spambase Datasrt создать свой спам-фильм с помощью библиотеки scikit-learn
### Загружаем библиотеки
```python
import pandas as pd  # для работы с датасетом
import nltk  # уже знакомый тулбокс
from nltk import stem  # стеммеры
from nltk.corpus import stopwords  # стопслова
import re  # для удаления лишних символов
from sklearn.model_selection import train_test_split  # для создание модели на основе метода опорных векторов
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report  # удобное представление метрик модели
```

### Читаем датасет, подготавливаем его оси
```python
df = pd.read_csv('data.csv', encoding='latin-1')  # считываем датасет
df = df[['v1', 'v2']].rename(columns={'v1': 'label', 'v2': 'text'})  # оставляем только колонки с текстами смс и метками
print(df.head(10))  # выведем первые 10 значений
df = df.drop_duplicates('text')  # удаляем дубликаты
df['label'] = df['label'].map({'ham': 0, 'spam': 1})  # заменяем метки на бинарные
```


### Создаём объект стеммера, загружаем стоп-слова
```python
stemmer = stem.SnowballStemmer('english')  # поиск основы слова
nltk.download('stopwords')
stopwords = set(stopwords.words('english'))  # добавляем множество английских стоп-слов
```

### Функция для преобработки текста
Удаляем знаки препинания, сводим всё к строчным буквам, удаляем стоп слова, проводим стемминг
```python
def preprocess(text):
    text = re.sub(r'[^\w\s]', '', text)  # возвращает строку в соответствии с шаблоном
    text = [el.lower() for el in text.split() if el not in stopwords]  # каждый элемент, не относящийся к стоп-словам
    # делаем маленьким и возвращаем в список
    text = ' '.join([stemmer.stem(el) for el in text])  # применяем стеммер к каждому слову, делаем предложение
    return text
# проверка, что всё верно работает
assert preprocess("I'm gonna be home soon and i don't want to talk about this stuff anymore tonight, k? I've cried enough today.") == "im gonna home soon dont want talk stuff anymor tonight k ive cri enough today"
assert preprocess("Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat...") == "go jurong point crazi avail bugi n great world la e buffet cine got amor wat"


df['text'] = df['text'].apply(preprocess)  # применим написанную функцию ко всем текстам
```


### Делим нашу выборку, векторизуем данные
```python
Y = df['label'].values  # создаём массив откликов
X_train, X_test, Y_train, Y_test = train_test_split(df['text'], Y, test_size=0.3, random_state=51)
# извлекаем признаки из текстов
vectorizer = TfidfVectorizer(decode_error='ignore')  # создаём объект класса TfidVectorizer
X_train = vectorizer.fit_transform(X_train)  # собрираем стастику по обучающему множеству
X_test = vectorizer.transform(X_test)  # применяем к тестовым данным

```

### Обучаем SVM
```python
# обучаем модель SVM

model = LinearSVC(random_state=51, C=1)
model.fit(X_train, Y_train)
predictions = model.predict(X_test)

print(classification_report(Y_test, predictions, digits=3))
```

### Определяем категорию нескольких писем
```python
s1 = "Enlightening A great overview of U.S. foreign policy."
s2 = "If you are interested, please write back and I will provide further instructions."
s3 = "This is a fine collection of articles by Bernard Lewis about the Middle East."
s4 = "I think this book is a must read for anyone who wants an insight into the Middle East."
s5 = "As a valued customer, I am pleased to advise you that following recent review of your Mob No. you are awarded with a å£1500 Bonus Prize, call 09066364589"
Lis = [s1, s2, s3, s4, s5]
for txt in Lis:
    txt = preprocess(txt)
    txt = vectorizer.transform([txt])
    print(model.predict(txt))
```
