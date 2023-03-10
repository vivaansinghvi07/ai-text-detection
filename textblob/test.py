# imports packages
import pickle
from textblob import TextBlob
from textblob.classifiers import NaiveBayesClassifier
import csv
from os import system

system('clear')
print("Fetching model...")

# gets the model
with open("../models/textblobmodel", "rb") as file:
    model = pickle.load(file)

# gets the data
wikiData = []

system('clear')
print("Getting data...")

with open("datasets/GPT-wiki-intro.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        # creates a dictionary with only the texts we want
        temp = {"Human": row["wiki_intro"], "Bot": row["generated_intro"]}
        
        # adds the dictionary to the array
        wikiData.append(temp)

# populate test data with wiki data
def processWikiData(dict):
    temp = list()
    temp.append((dict["Human"], 'human'))
    temp.append((dict["Bot"], 'ai'))
    return temp

# dataset for testing
test = []

# arbitrary section of the data
for i in range(20500, 21000):
    test += processWikiData(wikiData[i])

system('clear')
print("Running test...")

# prints the result
print(model.accuracy(test))