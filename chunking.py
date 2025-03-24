from razdel import sentenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
import numpy as np

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

# nltk.download("stopwords")
nltk.download("punkt")
nltk.download("punkt_tab")

# stop_words = set(stopwords.words("russian"))
punctuation = set(string.punctuation)

# Удаляет пунктуацию перед обработкой TF-IDF
def preprocess_text(text):
    words = word_tokenize(text.lower())
    filtered_words = [word for word in words if word not in punctuation]
    return " ".join(filtered_words)

# Автоматически выбирает количество смысловых чанков в зависимости от длины текста
def dynamic_n_topics(text_length, min_chunks=3, max_chunks=10):
    return min(max_chunks, max(min_chunks, text_length // 300))  # 300 символов = 1 смысловой чанк

# Разбивает текст на чанки по предложениям (sentence-based chunking)
def split_by_sentences(text, max_length=512):
    sentences = [s.text for s in sentenize(text)]
    chunks = []
    current_chunk = []

    for sentence in sentences:
        if sum(len(s) for s in current_chunk) + len(sentence) < max_length:
            current_chunk.append(sentence)
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

# Разделяет текст на смысловые блоки с помощью TF-IDF
def split_by_topics(text):
    sentences = [s.text for s in sentenize(text)]
    text_length = len(text)

    n_topics = dynamic_n_topics(text_length)
    sentences = [preprocess_text(s) for s in sentences] 
    
    tfidf_vectorizer = TfidfVectorizer(max_features=1000)
    tfidf_matrix = tfidf_vectorizer.fit_transform(sentences).toarray()

    # Нормализация матрицы TF-IDF по длине документа
    tfidf_matrix_normalized = normalize(tfidf_matrix, norm='l2', axis=1)
    
    # Вычисляем "важность" предложений как сумму TF-IDF весов
    sentence_scores = np.sum(tfidf_matrix_normalized, axis=1)
    
    # Определяем точки разрыва - места, где разница важности максимальна
    split_points = np.argsort(np.diff(sentence_scores))[-(n_topics - 1):] + 1
    split_points = np.sort(split_points)

    # split_points = np.linspace(0, len(sentences), n_topics + 1, dtype=int)[1:-1]
    
    topic_chunks = []
    start = 0
    for end in split_points:
        topic_chunks.append(" ".join(sentences[start:end]))
        start = end
    topic_chunks.append(" ".join(sentences[start:]))

    return topic_chunks

# Гибридное разбиение: сначала по темам (TF-IDF), затем по предложениям
def hybrid_chunking(text, chunk_size=512, overlap=100):
    topic_chunks = split_by_topics(text)

    final_chunks = []
    for topic_chunk in topic_chunks:
        sentence_chunks = split_by_sentences(topic_chunk, max_length=chunk_size)
        
        # Добавляем overlap (перекрытие между чанками)
        for i, chunk in enumerate(sentence_chunks):
            if i > 0:
                chunk = sentence_chunks[i-1][-overlap:] + " " + chunk
            final_chunks.append(chunk)

    return final_chunks

# Объединяет короткие чанки с соседними (Эмбеддинг-модель Qwen плохо работает с короткими предложениями)
def merge_short_chunks(chunks, min_length=80):
    merged_chunks = []
    buffer = ""

    for chunk in chunks:
        if len(chunk) < min_length:
            buffer += " " + chunk
        else:
            if buffer:
                merged_chunks.append(buffer.strip() + " " + chunk)
                buffer = ""
            else:
                merged_chunks.append(chunk)

    if buffer:
        merged_chunks[-1] += " " + buffer

    return merged_chunks