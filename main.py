import openai
from api_key import API_KEY
import re
import nltk
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

openai.api_key = API_KEY

# Two main TODOs:
# 1. Pre-process the input text to make it more suitable for GPT-3 using NLP
# 2. Choose the best answer from the GPT-3 output using ML


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
    return " ".join(txt)

if __name__ == "__main__":

    # Ask for user input until the user types "quit"
    user_input = ""
    while user_input != "quit":
        user_input = input("")
        user_input = pre_process_text(user_input)
        responses = openai.Completion.create(model="text-davinci-003", prompt=user_input, temperature=1, max_tokens=20, n=3)
        print(responses)

    # # Continuously ask for user input
    # while True:
    #     # Get user input
    #     user_input = input("You: ")

    #     # Pre-process user input
    #     user_input = pre_process_text(user_input)
            
    #     # Create prompt for GPT-3
    #     prompt = f"Human: {user_input}"

    #     # Ask GPT-3 for an answer
    #     response = openai.Completion.create()

