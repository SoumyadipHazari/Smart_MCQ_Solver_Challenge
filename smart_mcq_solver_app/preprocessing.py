import re

MAX_LEN = 64

def clean_text(text):

    text = text.replace("\n", " ")
    text = text.replace("\t", " ")

    text = re.sub(r"\s+", " ", text)

    return text.strip()

def create_question_option_pair(question, option):

    pair = (
        f"Question: {question}\n"
        f"Answer: {option}"
    )

    return clean_text(pair)

def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().split()



def encode_text(text, vocab):

    tokens = tokenize(text)

    return[
        vocab.get(token, vocab["<UNK>"])
        for token in tokens
    ]




def pad_sequence(sequence,vocab, max_len=MAX_LEN):

    if len(sequence) < max_len:

        sequence = sequence + [vocab["<PAD>"]] * (max_len - len(sequence))

    else:

        sequence = sequence[:max_len]

    return sequence



