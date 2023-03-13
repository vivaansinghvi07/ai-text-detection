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
        if word.isalpha():
            output += word + " "
    return output

# gets data from the wiki-intro dataset
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

            words.append(process(line["generated_intro"]))
            result.append('ai')
            

    # return generated lists
    return [words, result]

# gets data from the generated AI essays
def getAIEssayData(lowerCount, upperCount):
    # two empty arrays for words and results
    words, results = [], []

    # accesses the file
    with open("datasets/ai-essays.txt", "r") as f:

        # goes through every line
        for line in list(f.readlines())[lowerCount:upperCount]:

            # appends the line and says its ai
            words.append(process(line.strip("\n")))
            results.append('ai')


    # return generated arrays
    return [words, results]

# gets data from the feedbackprice essays dataset
def getHumanEssays(lowerCount, upperCount):
    # empty arrays for output
    words, results = [], []

    # generates sets for the huamn written text
    essays = []

    # access the file
    with open("datasets/feedback-essays.csv", "r") as f:

        # makes a reader
        reader = csv.DictReader(f)

        # initial values
        lines = list(reader)
        id = lines[0]["essay_id"]
        essay = lines[0]["discourse_text"]

        # goes through every line
        for line in lines[1:]:

            # if the essay id is the same, add it to the orignial
            # otherwise, create a new essay
            if line["essay_id"] == id:
                essay += " " + line["discourse_text"]
            else:
                id = line["essay_id"]
                essays.append(essay)
                essay = line["discourse_text"]

        # adds data to output
        for essay in essays[lowerCount:upperCount]:
            words.append(process(essay))
            results.append('human')

    return [words, results]


# clear console and print
def clprint(prompt):
    system('clear')
    print(prompt)