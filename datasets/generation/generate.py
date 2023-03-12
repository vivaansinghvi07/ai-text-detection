# imports os for the key
import os

# import libraries
import openai as ai
import time

# gets the api key
ai.api_key = os.environ['OPENAIKEY']

# gets an ai response, with the text stripped of new lines so it can fit in a text file
def genResponse():
    response = ai.Completion.create(
        model="text-davinci-003",
        prompt="Generate an essay about anything. Try to make the random topic specific rather than general.", 
        temperature=0.8,
        max_tokens=400,
        top_p=1,
        frequency_penalty=1,
        presence_penalty=1,
        n = 1
    )
    return response.choices[0].text.replace("\n", " ")

# open a file and write responses to it
with open("datasets/ai-essays.txt", "a") as f:

    # write a response to the file
    while True:
        try:
            # prints into file and updates file
            f.write(genResponse() + '\n')
            f.flush()
        except:
            # if rate limited, sleep for 5 minutes
            time.sleep(300)
