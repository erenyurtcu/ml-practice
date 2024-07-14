import csv
import pandas as pd
import re

# to clean the row
def clean_row(row):
    combined_row = []
    current_part = []
    for item in row:
        current_part.append(item)
        if item in ['0', '1'] and len(current_part) > 1:
            combined_row.append(','.join(current_part[:-1]))
            combined_row.append(current_part[-1])
            current_part = []
    return combined_row

cleaned_reviews = []
with open('Restaurant_Reviews.csv') as file:
    reader = csv.reader(file)
    cleaned_reviews = [clean_row(row) for row in reader]

# create DataFrame and remove nan rows
reviews = pd.DataFrame(cleaned_reviews, columns=['Review', 'Liked']).dropna()

reviews.to_csv('Cleaned_Restaurant_Reviews.csv', index=False)
reviews = pd.read_csv('Cleaned_Restaurant_Reviews.csv')

print(reviews)

# nltk
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
nltk.download('stopwords')
ps = PorterStemmer()

corpus = []
for i in range(len(reviews)):
    review = re.sub('[^a-zA-Z]', ' ', reviews['Review'][i])  # [^a-zA-Z] means replace any character that is not a letter with a space
    review = review.lower()  # make all letters lowercase
    review = review.split()  # make the list which includes the words of the sentence
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]  # if it is not a stopword, take the word's root
    review = ' '.join(review)
    corpus.append(review)

# feature extraction
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 2000)
X = cv.fit_transform(corpus).toarray()
y = reviews.iloc[:,1].values

# train test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# classification
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train,y_train)

y_pred = gnb.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)