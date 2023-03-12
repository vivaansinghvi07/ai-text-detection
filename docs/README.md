# AI Text Detection
This machine learning model will attempt to differnetiate between human and AI-written text. The model for my textblob implementation is not available due to its size. In hindsight, I probably should have been more optimal programming it, such as using text vectorization as I did in my second one, the sklearn model.

## Setup
- Start a python virtual runtime enviroment
- Create a python terminal
- Run the following commands to install necessary functions:
    ```
    $ pip install textblob
    $ pip install tensorflow
    ```
- Run `create.py` on either the textblob or ___ folder to generate your model
- Then, you can test accuracy using `test.py` or mess with the bot using ____

## Datasets Used
- [GPT Wiki Intro](https://huggingface.co/datasets/aadityaubhat/GPT-wiki-intro)


## Naive Bayes Classifier with TextBlob (version 1)

Files used here will be visible in the `v1-textblob` folder.

### Creation
I used the first 2000 entries in the dataset to train the model. You can edit the number of entries used at line 21. In each entry, I selected the wikipedia intro and the generated intro to teach the model which was which. This is visible in `create.py`. After creating and training the model, the program pickles it into a file (too large to be put on GitHub).

### Accuracy
I then ran a test on the accuracy of the model using TextBlob's accuracy tool, in `test.py`. To do this, I seletected 500 lines in an arbitrary location (not part of the training dataset, namely lines 20500-20999) in the wiki dataset. This location can be changed by editing the for loop's bounds at line 41. The results of the test are shown below. 

```
Running test...
0.824
```

Afterwards, I generated texts from OpenAI's playground, which I entered into the program. Here are a couple observations:
- Using a less advanced model, such as `text-curie-001` rather than `text-davinci-003`, tended to yield more accurate guesses. Advanced text generation such as davinci-003 seems to be capable of replicating human text somewhat well.
- Making the temperature (randomness) of the generation lower also tended to lead to more accurate guesses
- Using my own essays, the program found most to be human.

## Naive Bayes Classifier with SKLearn (version 2)

Due to the massive memory usage of TextBlob, I decided to find another way to do the project. Files used here will be visible in the `v2-sklearn` folder.

### Creation
I used the first 100000 lines of the wiki dataset to train and test this model, again using the `wiki_intro` as the human data and the `generated_intro` as the AI intro. One-fifth of the model was used to test the dataset. This is visible in `create.py`. The bounds for the creation data can be changed within the function.

### Accuracy
I ran an accuracy test using the test data. The results are here:
```
The accuracy of the model is 0.76085.
```
The bounds of the testing data, as the creation data, can be changed within the program.

## Conclusions
While vastly more powerful in memory, my second version using sklearn lags behind my first version using textblob in accuracy.
