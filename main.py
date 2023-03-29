import openai
from api_key import API_KEY
import re
import nltk
from sklearn.tree import DecisionTreeRegressor 
import pandas as pd
import numpy as np
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

openai.api_key = API_KEY

#############################
#### PRE-PROCESSING #########
#############################

# Clean input text by removing punctuation and extra spaces
def clean_text(txt):
    txt = " ".join(re.split("[^a-zA-Z0-9]+", txt))
    txt = txt[1:] if txt[0] == " " else txt
    return txt

# Tokenize input text into sentences
def tokenize_text(txt):
    return nltk.word_tokenize(txt)

# Lemmatize input text
def lemmatize_text(txt):
    lemmatizer = nltk.stem.WordNetLemmatizer()
    return [lemmatizer.lemmatize(w) for w in txt]

def pre_process_text(txt):
    txt = clean_text(txt.lower())
    txt = tokenize_text(txt)
    txt = lemmatize_text(txt)
    txt = txt + ["?"]
    return " ".join(txt)


#############################
#### POST-PROCESSING ########
#############################

def similar_words_perc(str1, str2):
    str1 = set(str1.split())
    str2 = set(str2.split())
    return len(str1 & str2) / len(str1 | str2)

def clean_response(str):
    if (str[0] == " "):
        str = str[1:]
    str = str[0].upper() + str[1:]
    return str

if __name__ == "__main__":

    # Load the data for the decision tree
    data = np.array(pd.read_csv("virtual_assistant_responses.csv", sep=','))
    X = data[:, 0:3].astype(float)
    Y = data[:, 3].astype(float)
    
    regressor = DecisionTreeRegressor(random_state = 0) 
    regressor.fit(X, Y)

    # Start the conversation
    print("Welcome to the virtual assistant! Ask me a question or type 'quit' to exit.")
    user_input = ""
    while user_input != "quit":
        user_input = input("")
        user_input = pre_process_text(user_input)
        prompt_wc = len(user_input.split())
        responses = openai.Completion.create(model="curie:ft-personal-2023-03-29-11-58-44", prompt=user_input, temperature=0.05, max_tokens=50, n=5, stop=["\n"])
        responses_txt = [r["text"] for r in responses["choices"]]

        relevance_scores = []
        for r in responses_txt:
            response_wc = len(r.split())
            sim_words = similar_words_perc(user_input, r)
            relevance_scores.append(regressor.predict([[prompt_wc, response_wc, sim_words]]))

        best_response = responses_txt[np.argmax(relevance_scores)]
        print(clean_response(best_response))