# codsoft_task4

# 🎬 Movie Recommendation System (CODSOFT Task)

## 📌 Project Description

This project is a Content-Based Movie Recommendation System developed using Python.

The system recommends movies based on similarity in genres using the TF-IDF Vectorization technique and Cosine Similarity.

When a user enters a movie name, the system suggests similar movies based on genre similarity.

---

## 🚀 Features

- Content-based filtering
- TF-IDF Vectorization
- Cosine similarity for recommendation
- Text preprocessing (cleaning, stemming, stopword removal)
- Handles invalid movie names
- Console-based interaction

---

## 🛠️ Technologies Used

- Python 3
- NumPy
- Pandas
- NLTK
- Scikit-learn

---

## 🧠 Algorithm Used

### 1️⃣ Text Preprocessing
- Convert to lowercase
- Remove punctuation
- Remove URLs and special characters
- Remove stopwords
- Apply stemming

### 2️⃣ TF-IDF (Term Frequency - Inverse Document Frequency)
Converts movie genres into numerical vectors.

### 3️⃣ Cosine Similarity
Measures similarity between movies based on genre vectors.

The system recommends top similar movies excluding the selected movie itself.

---

## 📂 Project Structure

codsoft_task4/
│
├── movie_recommendation.py
├── movie_metadata.csv
└── README.md


---

## ▶️ How to Run the Project

### Step 1: Install Required Libraries

pip install numpy pandas nltk scikit-learn


Download NLTK stopwords (Run once):

import nltk
nltk.download('stopwords')


### Step 2: Run the Program

python movie_recommendation.py


---

## 🎯 How It Works

1. User enters a movie name.
2. System cleans the input.
3. Matches it with dataset.
4. Finds similar movies using cosine similarity.
5. Displays top 10 recommendations.

---

## 💬 Example

🎬 Movie Recommendation System
Type 'exit' to quit

Enter movie name: avatar

Recommended Movies:

guardians of the galaxy

star trek

john carter
...


---

## 📊 Dataset Used

- movie_metadata.csv
- Contains movie titles and genres

---

## 📌 Internship Task

This project is developed as part of the CODSOFT Internship Program.

---


