import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import time
import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from multiprocessing import Pool

# Thời điểm bắt đầu
start_time = time.time()

def preprocess_text(text):
    # Tách từ và loại bỏ stop words
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text.lower())
    tokens = [token for token in tokens if token.isalnum() and token not in stop_words]
    return tokens
    
def word_embeddings_cosine_similarity(text1,text2):
    # text1, text2 = args
    # Create word embeddings for each text
    nlp = spacy.load('en_core_web_md')
    doc1 = nlp(text1)
    doc2 = nlp(text2)

    # Extract word vectors from the word embeddings
    vecs1 = np.array([token.vector for token in doc1])
    vecs2 = np.array([token.vector for token in doc2])

    # Calculate the average vector for each text
    avg_vec1 = np.mean(vecs1, axis=0)
    avg_vec2 = np.mean(vecs2, axis=0)

    # Calculate cosine similarity between the average vectors
    similarity = cosine_similarity([avg_vec1], [avg_vec2])[0][0]
    return similarity

def query(input_text, path_csv):
    df = pd.read_csv(path_csv)
    word_embeddings_cosine_sim = []

    # Sử dụng multiprocessing để xử lý đồng thời các dòng trong DataFrame
    with Pool() as pool:
        input_data = zip([input_text] * len(df['Caption']), df['Caption'])
        word_embeddings_cosine_sim = pool.starmap(word_embeddings_cosine_similarity, input_data)

    # Sử dụng list comprehension để tạo ra một list các cặp (giá trị, số thứ tự)
    value_index_pairs = [(value, index) for index, value in enumerate(word_embeddings_cosine_sim)]

    # Sắp xếp list các cặp theo giá trị giảm dần
    value_index_pairs.sort(reverse=True)
    # In ra số thứ tự của 10 giá trị lớn nhất
    for value, index in value_index_pairs[:10]:
        print("Giá trị:", value, " - Số thứ tự:", index, " Video :", df.loc[index, 'Video'], " Frame :", df.loc[index, 'frame_idx'])
    # Thời điểm kết thúc
    end_time = time.time()
    # Tính thời gian chạy
    execution_time = end_time - start_time
    print("Thời gian chạy:", execution_time, "giây")

query("there are a lot of people racing bicycles", "C:/Users/ASUS/AI-challenge-2023/summary_map_keyframes.csv")