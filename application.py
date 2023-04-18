from flask import Flask, render_template,request
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import sys
#print(sys.path)

closest_book_index = pickle.load(open('closest_book_index.pkl','rb'))
filtered_books= pickle.load(open('filtered_books.pkl','rb'))
recommended_books= pickle.load(open('recommended_books.pkl','rb'))
similarity_scores_for_filtered_books= pickle.load(open('similarity_scores_for_filtered_books.pkl','rb'))
similarity_scores= pickle.load(open('similarity_scores.pkl','rb'))
books= pickle.load(open('books.pkl','rb'))


application = Flask(__name__)

# Load data
books = pd.read_csv('books_new.csv')
#print(books)

# Fill missing values with empty string
selected_features = ['Index','Title','Author','Genre','SubGenre']
for feature in selected_features:
    books[feature] = books[feature].fillna('')  

# Convert Index to string
books['Index'] = books['Index'].apply(str)
books['Genre'] = books['Genre'].str.strip()
books['SubGenre'] = books['SubGenre'].str.strip()
print(books['Genre'].dtype)
print(books['SubGenre'].dtype)

# Combine features
combined_features = (books['Index'] + ' ' + books['Title'] + ' ' + books['SubGenre']).str.lower()

# Vectorize features
vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combined_features)

# Compute similarity scores
similarity_scores = cosine_similarity(feature_vectors)
#print("Shape of similarity scores:", similarity_scores.shape)
@application.route('/')
def home():
# Render home page with all books
    return render_template('home.html', books=books.to_dict('records'))



@application.route('/recommend')
def recommend_ui():
    return render_template('recommend.html',recommended_books=recommended_books)

@application.route('/recommend_books',methods=['post'])
def recommend():
    user_input = request.form.get('user_input')
    subgenre = request.form.get('subgenre')
#print(user_input)
# Filter books based on genre and subgenre
    filtered_books = books[books['SubGenre'] == user_input]
#print(filtered_books)

# If no books match the genre and subgenre, return an empty list
    if len(filtered_books) == 0:
        return []

# Find the index of the closest matching book
    closest_book_index = filtered_books['Index'].astype(float).idxmin()
#print(closest_book_index)

# Find books with similarity score above a threshold
    threshold = 0.01
    similarity_scores_for_filtered_books = similarity_scores[closest_book_index]
    recommended_books = []
    for index,score in enumerate(similarity_scores_for_filtered_books):
        if score > threshold:
            title = books.iloc[index]['Title']
            author = books.iloc[index]['Author']
            genre = books.iloc[index]['Genre']
            subgenre = books.iloc[index]['SubGenre']
            recommended_books.append((title,author,genre, subgenre))

# Sort recommended books by similarity score and return top n books
    recommended_books = sorted(recommended_books, key=lambda x: x[3], reverse=True)[:5]
#print(recommended_books)
    return recommended_books

if __name__ == '__main__':
    application.run(debug=True)

