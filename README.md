Projenin kodlarına [buraya tıklayarak](https://colab.research.google.com/drive/1B76ARLIlXXvPQGok5H90fwebmAxMbS9h#scrollTo=HGm5YA9vpLGc) ulaşabilirsiniz.
Dataset: 70000+ Turkish News - Kaggle
# LDAandEDA
# LDA ve EDA ile Türkçe Haberler Üzerinde Konu Modelleme (NLP Projesi)
Bu proje, Türkçe haber metinlerini kullanarak **Gizli Dirichlet Ayrımı (Latent Dirichlet Allocation - LDA)** ve **Keşifsel Veri Analizi (Exploratory Data Analysis - EDA)** yöntemleriyle doğal dil işleme (NLP) süreçlerini kapsamaktadır. Amacımız, metinlerdeki gizli konu dağılımlarını keşfetmek ve bu konuları modelleyerek anlamlı sonuçlar çıkarmaktır. Proje ayrıca **tutarlılık (coherence) skoru** ile model değerlendirmesi ve görselleştirme yöntemlerini içermektedir.

## Proje Özeti

- **Konu Modelleme**: Türkçe haber metinlerinde gizli konuların çıkarılması.
- **Tutarlılık Ölçümü**: Modelin performansını tutarlılık metrikleri ile değerlendirme.
- **Görselleştirme**: LDA modelini PyLDAvis kullanarak görselleştirme.
- **Dil İşleme Teknikleri**: Stopwords temizliği, metin normalizasyonu ve tokenizasyon işlemleri.

## İçindekiler

- Gereksinimler
- Kurulum ve Ortam Ayarları
- Veri Ön İşleme
- LDA Model Eğitimi
- Tutarlılık Skoru ile Model Değerlendirme
- Model Görselleştirme
- Sonuçlar ve Yorum


## Gereksinimler

Bu proje için aşağıdaki Python kütüphaneleri gereklidir:

- **numpy**: Sayısal işlemler için.
- **pandas**: Veri manipülasyonu ve analizi.
- **matplotlib**: Grafik ve görselleştirme.
- **nltk**: Doğal dil işleme için stopwords ve metin temizleme.
- **gensim**: LDA modelleme ve metin işleme.
- **pyLDAvis**: LDA modelinin görselleştirilmesi.

Kütüphaneleri yüklemek için:

```bash
pip install numpy pandas matplotlib nltk gensim pyLDAvis
```
Ayrıca nltk stopwords verisini indirmeyi unutmayın:
```bash
nltk.download('stopwords')
```
## Kurulum ve Ortam Ayarları
1. Depoyu Klonlayın:
```bash
git clone https://github.com/kullanici_adi/LDAandEDA.git
cd LDAandEDA
```
2. Gerekli Python Kütüphanelerini Kurun:
```bash
pip install -r requirements.txt
```
3. Google Colab Üzerinde Çalışma: Proje, Google Colab ortamında çalışmak üzere optimize edilmiştir. Google Drive’ınızı bağlayarak veri setini kullanabilirsiniz:
```bash
from google.colab import drive
drive.mount('/content/drive')
```

## Veri Ön İşleme
Veri işleme, metin verilerinin analiz için uygun hale getirilmesini sağlar. Bu projede, Türkçe haber metinleri işlenmektedir. Verilerin işlenmesi, stopwords çıkarma, küçük harfe dönüştürme, sayıları ve noktalama işaretlerini kaldırma gibi adımları içerir.
1. Stopwords ve Noktalama Temizliği
```bash
import string
from nltk.corpus import stopwords

# Stopwords ve noktalama işaretlerini yükleyin
stopwords_set = set(stopwords.words('turkish'))
punctuation_set = set(string.punctuation)

def clean_dataset(text):
    text = text.lower()  # Küçük harfe çevir
    text = ''.join([char for char in text if char not in punctuation_set])  # Noktalama işaretlerini kaldır
    text = ''.join([char for char in text if not char.isdigit()])  # Sayıları kaldır
    text = ' '.join(word for word in text.split() if word not in stopwords_set)  # Stopwords'leri çıkar
    return text

# Uygulama
df['cleaned_text'] = df['text'].apply(clean_dataset)
```
Bu adım, NLP modellerinin daha verimli çalışmasını sağlar ve gereksiz bilgileri temizler.

2. Tokenizasyon
Metinler, LDA modeli için tokenize edilmelidir:
```bash
tokenized_text = df['cleaned_text'].apply(lambda x: x.split())
```
## LDA Model Eğitimi
Latent Dirichlet Allocation (LDA), dokümanlardaki gizli konuları çıkarmak için kullanılan bir modeldir. Modeli eğitmek için, her bir dokümanı kelime frekansları üzerinden vektörize ederiz.
```bash
import gensim

# Kelime listesi ve doküman terim matrisi
word_list = gensim.corpora.Dictionary(tokenized_text)
document_term_matrix = [word_list.doc2bow(text) for text in tokenized_text]

# LDA Modeli
lda_model = gensim.models.ldamodel.LdaModel(corpus=document_term_matrix, id2word=word_list, num_topics=15, passes=10)
```
### Konu Modelleme Çıktıları
LDA modelinin ürettiği konular şu şekilde incelenebilir:
```bash
topics = lda_model.print_topics(num_words=7)
for topic in topics:
    print(topic)
```
Bu aşamada, her bir konu içindeki en sık geçen kelimeleri ve konu dağılımlarını inceleyebilirsiniz.

## Tutarlılık Skoru ile Model Değerlendirme
LDA modelinin başarısı, konu modellemede yaygın olarak kullanılan Coherence Score ile ölçülür. Bu metrik, konuların tutarlılığını değerlendirmeye yardımcı olur ve daha anlamlı konular oluşturulmasını sağlar.
```bash
from gensim.models import CoherenceModel

# Konu sayısı aralığını tanımlayın
coherence_score_list = []
topic_number_range_list = range(9, 30, 3)

for topic_number in topic_number_range_list:
    lda_model = gensim.models.ldamodel.LdaModel(corpus=document_term_matrix, id2word=word_list, num_topics=topic_number, passes=10)
    coherence_model_lda = CoherenceModel(model=lda_model, texts=tokenized_text, dictionary=word_list, coherence='c_v')
    coherence_score_list.append(coherence_model_lda.get_coherence())

# Skorları görselleştirin
plt.plot(topic_number_range_list, coherence_score_list, "-")
plt.xlabel("Konu Sayısı")
plt.ylabel("Tutarlılık Skoru")
plt.show()
```
Bu grafik, en uygun konu sayısını seçmeye yardımcı olur.

## Model Görselleştirme
LDA modelinin PyLDAvis ile görselleştirilmesi, konuların dağılımını ve ilişkilerini daha iyi anlamaya yardımcı olur. Bu, özellikle modelin doğruluğunu görsel olarak değerlendirmek için kullanışlıdır.
```bash
import pyLDAvis.gensim

vis = pyLDAvis.gensim.prepare(lda_model, document_term_matrix, dictionary=lda_model.id2word, mds='mmds')
pyLDAvis.display(vis)
```
Bu görselleştirme, interaktif olarak konuları incelemenizi sağlar.

## Sonuçlar ve Yorum
Bu projede, LDA modeli kullanarak Türkçe haber metinlerinde gizli konuları ortaya çıkardık. Coherence Score ve görselleştirme ile modeli değerlendirdik. Modelin performansı, veri temizleme ve ön işleme süreçlerinin kalitesine bağlıdır. Daha iyi sonuçlar için, veri setindeki çeşitlilik ve metin uzunlukları dikkate alınmalıdır.



## İletişim
Sorularınız veya önerileriniz için aşağıdaki iletişim bilgilerini kullanabilirsiniz:

E-posta: baydemirhatice@hotmail.com

Linkedln: https://www.linkedin.com/in/haticebaydemir/

