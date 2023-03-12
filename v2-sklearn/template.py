import csv
from os import system

# imports nltk and everything else needed 
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

""" -- This is here in case it is needed when the repo is cloned or something
nltk.download(['stopwords'])
"""

# sets stopwords
stopWords = set(stopwords.words('english'))

# preprocessing
def process(str):

    # turns the string into stopwords
    words = word_tokenize(str.lower())

    # creates output
    output = ""

    # adds word to output if its a word and if its not a stopword
    for word in words:
        if word.isalpha() and word not in stopWords:
            output += word + " "
    return output

def getWikiData(lowerCount, upperCount):
    # two empty arrays for the words and the results
    words, result = [], []

    # accesses the file
    with open("datasets/GPT-wiki-intro.csv", "r") as f:
        
        # creates reader
        reader = csv.DictReader(f)

        # iterates through reader
        for line in list(reader)[lowerCount:upperCount]:

            # get human data
            words.append(process(line["wiki_intro"]))
            result.append('human')

            # get AI data
            words.append(process(line["generated_intro"]))
            result.append('ai')

    # return generated lists
    return [words, result]

# clear console and print
def clprint(prompt):
    system('clear')
    print(prompt)