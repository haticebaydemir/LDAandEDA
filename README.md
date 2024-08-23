Projenin kodlarına [buraya tıklayarak](https://colab.research.google.com/drive/1B76ARLIlXXvPQGok5H90fwebmAxMbS9h#scrollTo=HGm5YA9vpLGc) ulaşabilirsiniz.
Dataset: 70000+ Turkish News - Kaggle
# LDAandEDA
LDA - Gizli Dirichlet Ayrımı (Latent Dirichlet Allocation)
En popüler konu modelleme algoritmalarından biridir.

Exploratory Data Analysis (EDA)

## Gerekli Kütüphaneler
```
!pip install pyLDAvis
!pip install pyLDAvis gensim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import re
import string
import nltk
from nltk.corpus import stopwords

import gensim
import pyLDAvis.gensim # for visualization
from gensim.models import CoherenceModel
```
## Veri Yükleme
```
from google.colab import drive
drive.mount('/content/drive')
```
```
# Dosya yolu:
file_path = '/content/drive/MyDrive/dataset/turkish_news.zip'
df = pd.read_csv(file_path)
```
```
import zipfile
import io
from google.colab import drive

# Google Drive'ı bağlayın
drive.mount('/content/drive')

# ZIP dosyasının yolu
zip_file_path = '/content/drive/MyDrive/dataset/turkish_news.zip'

# ZIP dosyasını açın ve CSV dosyasını Pandas DataFrame'e yükleyin
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    # ZIP dosyasındaki CSV dosyalarının isimlerini al
    csv_file_name = [name for name in zip_ref.namelist() if name.endswith('.csv')][0]

    # CSV dosyasını bellekten okuma
    with zip_ref.open(csv_file_name) as my_file:
        df = pd.read_csv(my_file)
```
```
# Veri serindeki ilk üç satır
df.head(3)
```
```
df['text'] = df['text'].astype(str)
#Bu satır, bir pandas DataFrame'deki text sütunundaki tüm verileri string (metin) veri türüne dönüştürmek için kullanılır.
```
```
news_df = df[['text']]
news_df.head()
#Bu satır, pandas DataFrame'inden sadece belirli bir sütunu seçmek ve yeni bir DataFrame oluşturmak için kullanılır.
```
```
import nltk

# 'stopwords' verisini indirin
nltk.download('stopwords')

```
```
import string
from nltk.corpus import stopwords

# Noktalama işaretleri setini oluşturun
punctuation_set = set(string.punctuation)

# Türkçe stopwords'leri yükleyin ve genişletin
stopwords_set = set(stopwords.words('turkish'))
stopwords_set.update(['bir', 'kadar', 'sonra'])

# Örnek metin
text = "Bu bir örnek metindir. Şu an 2024 yılındayız."

# Metni temizleme işlevi
def clean_dataset(text):
    # Küçük harfe çevir
    text = text.lower()

    # Noktalama işaretlerini kaldır
    text = ''.join([char for char in text if char not in punctuation_set])

    # Sayıları kaldır
    text = ''.join([char for char in text if not char.isdigit()])

    # Stopwords'leri kaldır
    text = ' '.join(word for word in text.split() if word not in stopwords_set)

    return text
```
```
news_df['text'] = news_df['text'].apply(lambda x: clean_dataset(x))
```
```
news_df['cleaned_text'] = news_df.iloc[5].text
```
```
news_df['cleaned_text'].iloc[5]
```
```
news_df['cleaned_text_token'] = news_df['cleaned_text'].apply(lambda x: x.split())
```
```
news_df.drop(['text'], axis=1, inplace=True)
```
```
news_df.head(5)
```
LDA MODEL TRAINING
```
tokenized_text = news_df['cleaned_text_token']
word_list = gensim.corpora.Dictionary(tokenized_text)
```
```
# vectorized terms
document_term_matrix = [word_list.doc2bow(term) for term in tokenized_text]
```
```
lda_model = gensim.models.ldamodel.LdaModel(
                                            corpus= document_term_matrix,
                                            id2word= word_list,
                                            num_topics = 15,
                                            passes = 10
                                           )
```
```
# Most repeated words in created topics
topics = lda_model.print_topics(num_words=7)

for topic in topics:
    print(topic)
```
MODEL VISUALIZATION
```
pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(lda_model, document_term_matrix, dictionary=lda_model.id2word, mds='mmds')
pyLDAvis.display(vis)
```
Choosing best topic by Coherence Score
```
topic_number_range_list = range(9,30,3)

coherence_score_list = list()
topic_number_list = list()

for topic_number in topic_number_range_list:
    lda_model = gensim.models.ldamodel.LdaModel(
                                                corpus= document_term_matrix,
                                                id2word= word_list,
                                                num_topics = topic_number,
                                                passes = 10
                                                )

    coherence_model_lda = CoherenceModel(
                                        model = lda_model,
                                        texts = tokenized_text,
                                        dictionary = word_list,
                                        coherence='c_v'
                                        )

    temp_coherence_score_lda = coherence_model_lda.get_coherence()
    coherence_score_list.append(temp_coherence_score_lda)
    topic_number_list.append(topic_number)

```
```
plt.plot(topic_number_list, coherence_score_list, "-")
plt.xlabel("Topic Numbers")
plt.ylabel("Coherence Scores")

plt.show()
```
```
topics = lda_model.print_topics(num_words=7)
topics = sorted(topics, key = lambda x: x[0])

for topic in topics:
    print(topic)
```
```
pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(lda_model, document_term_matrix, dictionary=lda_model.id2word, mds='mmds')
pyLDAvis.display(vis)
```
