# AI Text Detection
This machine learning model will attempt to differentiate between human and AI-written text. The model for my textblob implementation is not available due to its size. In hindsight, I probably should have been more optimal programming it, such as using text vectorization as I did in my second one, the sklearn model.

## Setup
- Start a python virtual runtime enviroment
- Create a python terminal
- Run the following commands to install necessary functions:
    ```
    $ pip install textblob
    $ pip install tensorflow
    $ pip install scikit-learn
    $ pip install nltk
    ```
- Run `create.py` on either the textblob or ___ folder to generate your model
- Then, you can test accuracy using `test.py` or mess with the bot using ____

## Datasets Used
- [GPT Wiki Intro](https://huggingface.co/datasets/aadityaubhat/GPT-wiki-intro)
- [AI-Generated Essays](https://github.com/vivaansinghvi07/ai-essay-dataset) (made by me using OpenAI's API)
- [LOCNESS Corpus](https://www.learnercorpusassociation.org/resources/tools/locness-corpus/)
    - Credit given to the **Centre for English Corpus Linguistics (CECL), Université catholique de Louvain, Belgium**.
    - Since this dataset cannot be released to the public without permission, it will not be visible in this repository.


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

I had originally used the [Feedback Price](https://www.kaggle.com/datasets/yujikomi/feedback-price-datasets-with-essay-text) dataset, but since most or all the essays were on a single topic, I switched to the LOCNESS corpus for more generalization and less bias.

### Creation
Visible in `create.py` and `template.py`, I used data from three datasets:
- Wiki-Intro: I read the first 1200 lines and used every other AI entry and every human entry
- AI Essays: I used the first 600 essays
- Human Essays: I used the first 600 paragraphs of the LOCNESS Corpus's USARG.txt

Then, this data was split into training and testing data using sklearn's `train_test_split()` function. The size of my datasets was limited by the size of the LOCNESS corpus. The model can certainly handle being trained with vastly more data.

### Accuracy
I ran an accuracy test using the test data. The results are here:
```
The accuracy of the model is 0.7733.
```
The bounds of the testing data, as the creation data, can be changed within the program.

Afterwards, when testing with essays written by me and several other bots, I saw somewhat consistent results, except some human essyas were characterized to be AI-written. This is possibly due to the informaility of some of the human essays in the training dataset.

With more data, which this model can definitely handle, the accuracy in real applications may be improved.

## Conclusions
While vastly more efficient in memory, my second version using sklearn lags behind my first version using textblob in accuracy. Regardless, with the advancement of artificial intelligence, it is or soon will be very hard to discern between AI-written and human-written text.
