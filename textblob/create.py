# starting imports
import csv
from os import system
import pickle

# imports packages
from textblob import TextBlob
from textblob.classifiers import NaiveBayesClassifier
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

# configures dynamic memory
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# creates an array to store the data frrom the GPT-Wiki dataset\
wikiData = []

# controls the amount of data I want from the wiki
wikiDataCount = 2000

count = 0

system('clear')
print("Importing data...")

# train with https://huggingface.co/datasets/aadityaubhat/GPT-wiki-intro
with open("datasets/GPT-wiki-intro.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        # creates a dictionary with only the texts we want
        temp = {"Human": row["wiki_intro"], "Bot": row["generated_intro"]}
        
        # adds the dictionary to the array
        wikiData.append(temp)

        # checks count limit
        count += 1
        if count == wikiDataCount:
            break

# populate train data with wiki data
def processWikiData(dict):
    temp = list()
    temp.append((dict["Human"], 'human'))
    temp.append((dict["Bot"], 'ai'))
    return temp

# creates training dataset
train = []

# adds every set to the training dataset
for dataSet in wikiData:
    train += processWikiData(dataSet)

system('clear')
print("Training model...")

# trains model
cl = NaiveBayesClassifier(train)

# saves model into a file
with open('../models/textblobmodel.pkl', 'wb') as f:
    pickle.dump(cl, f)

print("Model Saved.")