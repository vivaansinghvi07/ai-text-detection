import csv
from os import system

# imports nltk and everything else needed
from nltk.tokenize import word_tokenize

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

        # counter
        count = 0

        # iterates through reader
        for line in list(reader)[lowerCount:upperCount]:

            # get human data
            words.append(process(line["wiki_intro"]))
            result.append('human')

            count += 1
            
            if not count % 2:
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

# gets human essays from the LOCNESS corpus
def getHumanEssayData(lowerCount, upperCount):
    # two empty arrays for words and results
    words, results = [], []

    # accesses the file
    with open("datasets/USARG.txt", "r") as f:

        # goes through every line
        for line in list(f.readlines())[lowerCount:upperCount]:

            # appends the line and says its ai
            words.append(process(line.strip("\n")))
            results.append('human')


    # return generated arrays
    return [words, results]

# clear console and print
def clprint(prompt):
    system('clear')
    print(prompt)
