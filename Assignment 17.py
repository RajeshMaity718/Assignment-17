import pandas as pd
import re
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

# ==================================================
# DOWNLOAD REQUIRED NLTK PACKAGES
# ==================================================

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# ==================================================
# LOAD DATASET
# ==================================================

# Example Dataset:
# Movie Reviews / Tweets / Product Reviews

df = pd.read_csv("text_dataset.csv")

print("Dataset Shape:")
print(df.shape)

print("\nFirst 5 Rows:")
print(df.head())

# Assume text column name is 'text'

# ==================================================
# PART 1 : NLP PIPELINE & BASIC TEXT CLEANING
# ==================================================

# ==================================================
# Task 1 : Understanding Raw Text Data
# ==================================================

print("\nFirst 5 Text Samples")

for i in range(5):
    print(df['text'][i])

print("\nText Length")

df['text_length'] = df['text'].astype(str).apply(len)

print(df['text_length'].head())

print("""

Common Issues Found:

1. Uppercase and Lowercase mismatch
2. Punctuation marks
3. Numbers
4. Extra spaces
5. URLs
6. Emails
7. Emojis
8. HTML Tags

""")

# ==================================================
# Task 2 : Basic Text Cleaning
# ==================================================

def basic_clean(text):

    text = str(text).lower()

    text = re.sub(r'[^\w\s]', '', text)

    text = re.sub(r'\d+', '', text)

    text = re.sub(r'\s+', ' ', text)

    text = text.strip()

    return text

df['clean_text_basic'] = df['text'].apply(
    basic_clean
)

print("\nOriginal vs Basic Cleaned")

print(
    df[['text', 'clean_text_basic']].head()
)

# ==================================================
# PART 2 : ADVANCED TEXT CLEANING
# ==================================================

# ==================================================
# Task 3 : Removing Noise
# ==================================================

def advanced_clean(text):

    text = str(text)

    # URLs
    text = re.sub(
        r'http\S+|www\S+',
        '',
        text
    )

    # Emails
    text = re.sub(
        r'\S+@\S+',
        '',
        text
    )

    # HTML Tags
    text = re.sub(
        r'<.*?>',
        '',
        text
    )

    # Emojis & Special Characters
    text = re.sub(
        r'[^a-zA-Z\s]',
        '',
        text
    )

    text = text.lower()

    text = re.sub(
        r'\s+',
        ' ',
        text
    )

    return text.strip()

df['clean_text_advanced'] = (
    df['text']
    .apply(advanced_clean)
)

print("\nAdvanced Cleaned Text")

print(
    df['clean_text_advanced']
    .head()
)

# ==================================================
# Task 4 : Stopword Removal
# ==================================================

stop_words = set(
    stopwords.words('english')
)

def remove_stopwords(text):

    words = text.split()

    words = [
        word
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)

df['text_no_stopwords'] = (
    df['clean_text_advanced']
    .apply(remove_stopwords)
)

print("\nAfter Stopword Removal")

print(
    df['text_no_stopwords']
    .head()
)

# ==================================================
# Task 5 : Repeated Characters & Slang
# ==================================================

slang_dict = {
    "u": "you",
    "gr8": "great",
    "b4": "before",
    "luv": "love"
}

def normalize_text(text):

    # repeated characters

    text = re.sub(
        r'(.)\1{2,}',
        r'\1',
        text
    )

    words = text.split()

    words = [
        slang_dict.get(word, word)
        for word in words
    ]

    return " ".join(words)

df['normalized_text'] = (
    df['text_no_stopwords']
    .apply(normalize_text)
)

print("\nNormalized Text")

print(
    df['normalized_text']
    .head()
)

# ==================================================
# PART 3 : BASIC TEXT PREPROCESSING
# ==================================================

# ==================================================
# Task 6 : Tokenization
# ==================================================

print("\nWord Tokenization")

for i in range(min(3, len(df))):

    tokens = word_tokenize(
        df['normalized_text'][i]
    )

    print(tokens)

print("\nSentence Tokenization")

for i in range(min(3, len(df))):

    sentences = sent_tokenize(
        str(df['text'][i])
    )

    print(sentences)

# ==================================================
# Task 7 : Stemming
# ==================================================

stemmer = PorterStemmer()

def stem_text(text):

    words = word_tokenize(text)

    stems = [
        stemmer.stem(word)
        for word in words
    ]

    return " ".join(stems)

df['stemmed_text'] = (
    df['normalized_text']
    .apply(stem_text)
)

print("\nOriginal vs Stemmed")

print(
    df[
        ['normalized_text',
         'stemmed_text']
    ].head()
)

# ==================================================
# Task 8 : Lemmatization
# ==================================================

lemmatizer = WordNetLemmatizer()

def lemmatize_text(text):

    words = word_tokenize(text)

    lemmas = [
        lemmatizer.lemmatize(word)
        for word in words
    ]

    return " ".join(lemmas)

df['lemmatized_text'] = (
    df['normalized_text']
    .apply(lemmatize_text)
)

print("\nStemming vs Lemmatization")

print(
    df[
        ['stemmed_text',
         'lemmatized_text']
    ].head()
)

# ==================================================
# Task 9 : Final NLP Pipeline
# ==================================================

def nlp_preprocess(text):

    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(
        r'http\S+|www\S+',
        '',
        text
    )

    # Remove Emails
    text = re.sub(
        r'\S+@\S+',
        '',
        text
    )

    # Remove Special Characters
    text = re.sub(
        r'[^a-zA-Z\s]',
        '',
        text
    )

    # Remove Extra Spaces
    text = re.sub(
        r'\s+',
        ' ',
        text
    )

    # Tokenization
    tokens = word_tokenize(text)

    # Stopword Removal
    tokens = [
        word
        for word in tokens
        if word not in stop_words
    ]

    # Lemmatization
    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
    ]

    return " ".join(tokens)

df['final_clean_text'] = (
    df['text']
    .apply(nlp_preprocess)
)

print("\nFinal NLP Pipeline Output")

print(
    df[
        ['text',
         'final_clean_text']
    ].head()
)

# ==================================================
# Task 10 : Observations & Insights
# ==================================================

print("""

OBSERVATIONS

1. Basic Cleaning vs Advanced Cleaning

Basic cleaning removes:
- Lowercase issues
- Punctuation
- Numbers

Advanced cleaning additionally removes:
- URLs
- Emails
- HTML Tags
- Emojis
- Special Characters

------------------------------------------------

2. Why Lemmatization is Preferred?

Lemmatization returns meaningful root words.

Example:

better -> good

running -> run

Stemming may produce incomplete words.

------------------------------------------------

3. Importance of NLP Preprocessing

- Reduces noise
- Improves text quality
- Improves model accuracy
- Reduces dimensionality
- Makes text machine readable

------------------------------------------------

""")

print("\nAssignment 17 Completed Successfully")
