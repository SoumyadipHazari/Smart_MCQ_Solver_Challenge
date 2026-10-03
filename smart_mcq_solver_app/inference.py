import json
import joblib
import torch

from pathlib import Path
from model import BiLSTMClassifier
from preprocessing import (
    create_question_option_pair,
    encode_text,
    pad_sequence,
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"

CONFIG_PATH = MODEL_DIR / "mcq_config.json"
VOCAB_PATH = MODEL_DIR / "mcq_vocab.joblib"
LABEL_MAP_PATH = MODEL_DIR / "label_map.joblib"
MODEL_PATH = MODEL_DIR / "final_bilstm_model.pth"

def load_model():

    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)

    vocab = joblib.load(VOCAB_PATH)

    label_map = joblib.load(LABEL_MAP_PATH)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = BiLSTMClassifier(
        vocab_size=config["vocab_size"],
        embedding_dim=config["embedding_dim"],
        hidden_dim=config["hidden_dim"],
        dropout=config["dropout"]
    )

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=device,
            weights_only=True
        )
    )

    model.to(device)
    model.eval()

    return model, vocab, label_map, config, device

model, vocab, label_map, config, device = load_model()

def prepare_input(question, option, vocab, device):

    pair = create_question_option_pair(
        question,
        option
    )

    encoded = encode_text(
        pair,
        vocab
    )

    padded = pad_sequence(
        encoded,
        vocab
    )

    tensor = torch.tensor(
        padded,
        dtype=torch.long
    ).unsqueeze(0)

    return tensor.to(device)

def predict(
    question,
    option_a,
    option_b,
    option_c,
    option_d,
    option_e
):

    options = {
        "A": option_a,
        "B": option_b,
        "C": option_c,
        "D": option_d,
        "E": option_e,
    }

    raw_scores = []

    with torch.no_grad():

        for label, option in options.items():

            sequence = prepare_input(
                question,
                option,
                vocab,
                device
            )

            output = model(sequence)

            raw_scores.append(output.squeeze())

    logits = torch.stack(raw_scores)

    temperature = 20.0

    probabilities = torch.softmax(
        logits / temperature,
        dim=0
    )

    scores = {
        label: float(probability.item())
        for label, probability in zip(
            options.keys(),
            probabilities
        )
    }
    # Rank options using the original model logits
    ranked = sorted(
        zip(options.keys(), logits.tolist()),
        key=lambda x: x[1],
        reverse=True
    )

    prediction = ranked[0][0]

    top3 = [
        option
        for option, _ in ranked[:3]
    ]

    return prediction, top3, scores
if __name__ == "__main__":

    question = "What is the capital of France?"

    prediction, top3, scores = predict(

        question,

        "London",

        "Paris",

        "Berlin",

        "Madrid",

        "Rome"

    )

    print("Prediction :", prediction)

    print()

    print("Top 3 :", top3)

    print()

    print(scores)