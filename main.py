import csv

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

# train with https://huggingface.co/datasets/aadityaubhat/GPT-wiki-intro
with open("datasets/GPT-wiki-intro.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        # creates a dictionary with only the texts we want
        temp = {"Human": row["wiki_intro"], "Bot1": row["generated_intro"], "Bot2": row["generated_text"]}
        
        # adds the dictionary to the array
        wikiData.append(temp)

# populate train data with wiki data
def processWikiData(dict):
    temp = list()
    temp.append((dict["Human"], 'human'))
    temp.append((dict["Bot1"], 'ai'))
    temp.append((dict["Bot2"], 'ai'))
    return temp

train = []

for i in range(50):
    train += processWikiData(wikiData[i])

cl = NaiveBayesClassifier(train)

print(cl.classify(wikiData[5000]["Human"]))