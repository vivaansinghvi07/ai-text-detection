# import functions 
import template

# import pacakages
import pickle

# gets sklearn and needed packages
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score

# get stopwords from nltk
"""
nltk.download(['stopwords'])
"""

template.clprint("Getting data...")

# how many lines are read and put into the model
WIKIDATACOUNT = 2700    # dataset size: about 150000
AIDATACOUNT = 2700      # dataset size: about 2800
HUMANDATACOUNT = 2700   # dataset size: about 4000 

# gets the data from the wiki
wikiDataX, wikiDataY = template.getWikiData(0, WIKIDATACOUNT)

# gets data from my generated essays 
aiDataX, aiDataY = template.getAIEssayData(0, AIDATACOUNT)

# gets data from human essays
humanDataX, humanDataY = template.getHumanEssays(0, HUMANDATACOUNT)

# puts data into the overall arrays
dataX = wikiDataX + aiDataX + humanDataX
dataY = wikiDataY + aiDataY + humanDataY

# splits data
trainDataX, testDataX, trainDataY, testDataY = train_test_split(dataX, dataY, test_size=0.2, random_state=12) # random state for replicability

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