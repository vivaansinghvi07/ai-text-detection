# import template function
import template

# import packages
import csv
import pickle

# gets sklearn and needed things
import sklearn
from sklearn.metrics import accuracy_score

# sets the bounds for data
LOWERBOUND = 60000
UPPERBOUND = 65000

# creates test arrays
wikiDataTestX, wikiDataTestY = template.getWikiData(LOWERBOUND, UPPERBOUND)

# obtains model
with open("models/sklearnmodel.pkl", "rb") as f:
    model = pickle.load(f)

# vectorizes data
wikiDataTestX = template.vectorize(wikiDataTestX)

# evaluates accuracy
wikiDataModelY = model.predict(wikiDataTestX)
print(accuracy_score(wikiDataTestY, wikiDataModelY))