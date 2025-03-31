import string
from typing import List

import nltk
import numpy as np
from nltk.tokenize import word_tokenize
from razdel import sentenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

nltk.download("punkt")
nltk.download("punkt_tab")

punctuation = set(string.punctuation)

def preprocess_text(text: str) -> str:
    """
    Очищает текст от пунктуации и приводит слова к нижнему регистру.
    Используется для подготовки текста перед построением TF-IDF матрицы.
    """
    words = word_tokenize(text.lower())
    filtered_words = [word for word in words if word not in punctuation]
    return " ".join(filtered_words)

def dynamic_n_topics(text_length: int, min_chunks: int = 3, max_chunks: int = 30) -> int:
    """
    Определяет количество смысловых чанков на основе длины текста.
    Позволяет гибко масштабировать количество блоков при разбиении.
    """
    return min(max_chunks, max(min_chunks, text_length // 300))

def split_by_sentences(text: str, max_length: int = 512) -> List[str]:
    """
    Разбивает текст на чанки по предложениям, сохраняя длину каждого чанка не больше max_length.
    Использует жадный подход с накоплением предложений в текущем чанке.
    """
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

def split_by_topics(text: str) -> List[str]:
    """
    Делит текст на смысловые части с помощью TF-IDF анализа.
    Определяет наиболее значимые точки разрыва по изменению важности предложений.
    """
    sentences = [s.text for s in sentenize(text)]
    text_length = len(text)

    n_topics = dynamic_n_topics(text_length)
    sentences = [preprocess_text(s) for s in sentences]

    tfidf_vectorizer = TfidfVectorizer(max_features=1000)
    tfidf_matrix = tfidf_vectorizer.fit_transform(sentences).toarray()

    tfidf_matrix_normalized = normalize(tfidf_matrix, norm='l2', axis=1)
    sentence_scores = np.sum(tfidf_matrix_normalized, axis=1)

    split_points = np.argsort(np.diff(sentence_scores))[-(n_topics - 1):] + 1
    split_points = np.sort(split_points)

    topic_chunks = []
    start = 0
    for end in split_points:
        topic_chunks.append(" ".join(sentences[start:end]))
        start = end
    topic_chunks.append(" ".join(sentences[start:]))

    return topic_chunks

def hybrid_chunking(text: str, chunk_size: int = 512, overlap: int = 100) -> List[str]:
    """
    Сначала делит текст на смысловые блоки с помощью TF-IDF, 
    затем каждый блок дополнительно разбивает по предложениям.
    Добавляет перекрытие между соседними чанками для лучшего сохранения контекста.
    """
    topic_chunks = split_by_topics(text)

    final_chunks = []
    for topic_chunk in topic_chunks:
        sentence_chunks = split_by_sentences(topic_chunk, max_length=chunk_size)

        for i, chunk in enumerate(sentence_chunks):
            if i > 0:
                chunk = sentence_chunks[i - 1][-overlap:] + " " + chunk
            final_chunks.append(chunk)

    return final_chunks

def merge_short_chunks(chunks: List, min_length: int = 80) -> List[str]:
    """
    Объединяет слишком короткие чанки с соседними, чтобы избежать потерь информации и
    повысить эффективность эмбеддинговой модели Qwen. Используется как постпроцессинг после разбиения.
    """
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
