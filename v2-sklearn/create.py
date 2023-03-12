# import functions 
import template

# import pacakages
import csv
import pickle

# gets sklearn and needed packages
import sklearn 
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score

# gets reshaping from numpy
from numpy import reshape


# get stopwords from nltk
"""
nltk.download(['stopwords'])
"""

template.clprint("Getting data...")

# array for train data and test data
trainDataX, trainDataY, testDataX, testDataY = [], [], [], []

# how many lines are read and put into the model
DATACOUNT = 50000

# gets the data from the wiki
wikiDataTrainX, wikiDataTrainY = template.getWikiData(0, DATACOUNT)
wikiDataTestX, wikiDataTestY = template.getWikiData(DATACOUNT, DATACOUNT + 5000)

# puts data into train and test
trainDataX += wikiDataTrainX
testDataX += wikiDataTestX
trainDataY += wikiDataTrainY
testDataY += wikiDataTestY

# transforms data
vectorizer = TfidfVectorizer()
trainDataVecX = vectorizer.fit_transform(trainDataX)
testDataVecX = vectorizer.transform(testDataX)

# creates model
classifier = MultinomialNB()

template.clprint("Training model...")

# trains model
classifier.fit(trainDataVecX, trainDataY)

template.clprint("Saving model...")

# saves model
with open('models/sklearnmodel.pkl', "wb") as f:
    pickle.dump(classifier, f)

# saves vectorizer
with open('models/sklearnvectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

template.clprint("Testing model...")

# runs accuracy test
testPredictedY = classifier.predict(testDataVecX)
template.clprint("The accuracy of the model is " + str(accuracy_score(testDataY, testPredictedY)) + ".")