# import packages
import csv
import pickle

# gets sklearn and needed things
import sklearn
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer

# creates test arrays
wikiDataTestX = []
wikiDataTestY = []

with open("datasets/GPT-wiki-intro.csv", "r") as f:
    # creates reader 
    reader = csv.DictReader(f)

    for line in list(reader)[:2]:
        print(line)


