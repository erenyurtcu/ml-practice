import csv
import pandas as pd

def clean_row(row):
    combined_row = [] # to hold the cleaned row
    current_part = [] # to hold the current part being processed
    for item in row: # process for each item in the row sequentially
        current_part.append(item)
        if item in ['0', '1'] and len(current_part) > 1:
            combined_row.append(','.join(current_part[:-1]))  # join all elements in current_part except the last one with commas and add to combined_row
            combined_row.append(current_part[-1])
            current_part = []
    return combined_row

cleaned_reviews = []
with open('Restaurant_Reviews.csv') as file:
    reader = csv.reader(file)
    cleaned_reviews = [clean_row(row) for row in reader]

# create DataFrame and remove nan rows
reviews = pd.DataFrame(cleaned_reviews, columns=['Review', 'Liked']).dropna()

# the cleaned data
reviews.to_csv('Cleaned_Restaurant_Reviews.csv', index=False)
reviews = pd.read_csv('Cleaned_Restaurant_Reviews.csv')

print(reviews)

import re
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
nltk.download('stopwords')
ps = PorterStemmer()

corpus = []

for i in range(1000):
    review = re.sub('[^a-zA-z]',' ', reviews['Review'][i]) # [^a-zA-z] means replace any character that is not a letter with a space
    review = review.lower() # make all letters to lowercase
    review = review.split() # make the list which includes the words of the sentence
    review = [ps.stem(word) for word in review if not word in set (stopwords.words('english'))] # if it is not stopword, than take the words' root as a list element
    review = ' '.join(review)
    corpus.append(review)