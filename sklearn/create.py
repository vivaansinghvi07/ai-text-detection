# import pacakages
import csv
import pickle
from os import system

# gets sklearn and needed packages
import sklearn 
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer

# gets nltk and needed packages
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# get stopwords from nltk
"""
nltk.download(['stopwords'])
"""

# gets stopwords
stopWords = set(stopwords.words('english'))


system('clear')
print("Getting data...")

# processes string 
def process(str, stopWords):
    # turns the string into stopwords
    words = word_tokenize(str.lower())

    # creates output
    output = ""

    # adds word to output if its a word and if its not a stopword
    for word in words:
        if word.isalpha() and word not in stopWords:
            output += word + " "
    return output

# creates a vectorizer
vectorizer = TfidfVectorizer()

# gets the data from the wiki file
wikiDataWords = []
wikiDataResult = []

# how many lines are read and put into the model
DATACOUNT = 50000

# keeps count of the lines
count = 0

# gets data from file into array
with open("datasets/GPT-wiki-intro.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        # gets human data
        wikiDataWords.append(process(row["wiki_intro"], stopWords))
        wikiDataResult.append('human')
                              
        # gets bot data
        wikiDataWords.append(process(row["generated_intro"], stopWords))
        wikiDataResult.append('ai')
        

        count += 1
        if count == DATACOUNT:
            break

# transforms data
wikiDataWords = vectorizer.fit_transform(wikiDataWords)

# creates model
classifier = MultinomialNB()

system('clear')
print("Training model...")

# trains model
classifier.fit(wikiDataWords, wikiDataResult)

# saves model
with open('models/sklearnmodel.pkl', "wb") as f:
    pickle.dump(classifier, f)

system('clear')
print("Model saved.")