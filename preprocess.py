"""
cleans review text before we give it to a model.
"""

import html
import re

from nltk.corpus import stopwords


WORDS_TO_KEEP = [
    # negations
    "no", "nor", "not", "never", "none", "nobody", "nothing", "nowhere",
    "neither", "cannot", "without",
    # contrast words
    "but", "however", "although", "though", "yet", "except", "despite",
    "against",
    # contractions 
    "aren", "couldn", "didn", "doesn", "hadn", "hasn", "haven", "isn",
    "mightn", "mustn", "needn", "shan", "shouldn", "wasn", "weren", "won",
    "wouldn", "don", "ain",
]


def get_stop_words():
    #returns list of words we want to keep besides negations/contrast words/contractions
    nltk_stop_words = stopwords.words("english")

    our_stop_words = []
    for word in nltk_stop_words:
        if word not in WORDS_TO_KEEP:
            our_stop_words.append(word)

    # set faster to search
    return set(our_stop_words)


#removing html/urls
def remove_html_and_urls(text):
    #unescape changes html codes to normal char 
    text = html.unescape(text)

    #replace any html tags
    text = re.sub(r"<[^>]+>", " ", text)

    #replace web links
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"www\.\S+", " ", text)
    return text


#expanding contractions
SPECIAL_CONTRACTIONS = {
    "can't": "can not",
    "won't": "will not",
    "shan't": "shall not",
    "ain't": "is not",
    "let's": "let us",
    "i'm": "i am",
}

#patterns
CONTRACTION_ENDINGS = [
    ("n't", " not"),    
    ("'re", " are"),    
    ("'ve", " have"),   
    ("'ll", " will"),   
    ("'d", " would"),   
    ("'m", " am"),      
    ("'s", " is"),      
]


def expand_contractions(text):
    #curly apostrophes
    text = text.replace("\u2019", "'")
    text = text.replace("\u2018", "'")

    #special cases
    for short, long in SPECIAL_CONTRACTIONS.items():
        text = text.replace(short, long)

    #general
    for ending, replacement in CONTRACTION_ENDINGS:
        text = text.replace(ending, replacement)

    return text


def keep_only_letters(text):
    #replace anything not lowercase
    text = re.sub(r"[^a-z ]", " ", text)
    return text


#removing stop words
def remove_stop_words(text, stop_words):
    words = text.split()

    kept_words = []
    for word in words:
        if word not in stop_words:
            kept_words.append(word)

    return " ".join(kept_words)


STOP_WORDS = get_stop_words()  


def clean_text(text):
    if not isinstance(text, str):
        return "Not a review."

    text = remove_html_and_urls(text)
    text = text.lower()
    text = expand_contractions(text)
    text = keep_only_letters(text)
    text = remove_stop_words(text, STOP_WORDS)
    return text

