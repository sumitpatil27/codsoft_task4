import numpy as np
import pandas as pd
import re
import string
import nltk

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer


netflix_df = pd.read_csv("task4/movie_metadata.csv")


required_nf_df = netflix_df[["movie_title","genres"]]


required_nf_df = required_nf_df.dropna().reset_index(drop=True)



stop_words = set(stopwords.words("english"))
stemmer = SnowballStemmer("english")

def clean(text):
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)

    words = [
        stemmer.stem(word)
        for word in text.split()
        if word not in stop_words
    ]

    return " ".join(words)


required_nf_df["movie_title"] = required_nf_df["movie_title"].apply(clean)


tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(required_nf_df["genres"])

similarity = cosine_similarity(tfidf_matrix)

indices = pd.Series(
    required_nf_df.index,
    index=required_nf_df["movie_title"]
).drop_duplicates()


def netflix_recommendation(title, top_n=10):
    index = indices[title]

    
    sim_scores = similarity[index]
    sim_scores = np.asarray(sim_scores).reshape(-1)

    
    similarity_scores = list(enumerate(sim_scores))
    similarity_scores = sorted(
        similarity_scores,
        key=lambda x: float(x[1]),
        reverse=True
    )

 
    similarity_scores = similarity_scores[1:]


    valid_indices = []
    max_index = len(required_nf_df) - 1

    for i, _ in similarity_scores:
        if 0 <= i <= max_index:
            valid_indices.append(i)
        if len(valid_indices) == top_n:
            break

    return required_nf_df["movie_title"].iloc[valid_indices]



print("🎬 Movie Recommendation System")
print("Type 'exit' to quit")

while True:
    user_movie = input("\nEnter movie name: ")

    if user_movie.lower() == "exit":
        print("Goodbye 👋")
        break

    
    user_movie_cleaned = clean(user_movie)

    if user_movie_cleaned not in indices:
        print(" Movie not found. Try full or correct name.")
    else:
        print("\n Recommended Movies:")
        recommendations = netflix_recommendation(user_movie_cleaned)

        for i, movie in enumerate(recommendations, 1):
            print(f"{i}. {movie}")
