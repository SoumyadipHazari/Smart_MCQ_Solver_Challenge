import streamlit as st
import pandas as pd
from pathlib import Path

from inference import predict


st.set_page_config(
    page_title="Smart MCQ Solver",
    layout="centered"
)

BASE_DIR = Path(__file__).resolve().parent
TRAIN_PATH = BASE_DIR / "train.csv"

@st.cache_data
def load_dataset():
    return pd.read_csv(TRAIN_PATH)


df = load_dataset()


st.title("Smart MCQ Solver")
st.subheader("BiLSTM + Attention")


# --------------------------------------------------
# Random Question
# --------------------------------------------------

if st.button("Generate Random Question"):

    row = df.sample(1).iloc[0]

    st.session_state["question"] = row["prompt"]
    st.session_state["A"] = row["A"]
    st.session_state["B"] = row["B"]
    st.session_state["C"] = row["C"]
    st.session_state["D"] = row["D"]
    st.session_state["E"] = row["E"]
    st.session_state["correct_answer"] = row["answer"]


# --------------------------------------------------
# Get values from session state
# --------------------------------------------------

question = st.text_area(
    "Question",
    value=st.session_state.get("question", ""),
    height=120
)

option_a = st.text_input(
    "Option A",
    value=st.session_state.get("A", "")
)

option_b = st.text_input(
    "Option B",
    value=st.session_state.get("B", "")
)

option_c = st.text_input(
    "Option C",
    value=st.session_state.get("C", "")
)

option_d = st.text_input(
    "Option D",
    value=st.session_state.get("D", "")
)

option_e = st.text_input(
    "Option E",
    value=st.session_state.get("E", "")
)


# --------------------------------------------------
# Predict
# --------------------------------------------------

if st.button("Predict"):

    if not question.strip():
        st.warning("Please enter a question.")

    elif not all([
        option_a.strip(),
        option_b.strip(),
        option_c.strip(),
        option_d.strip(),
        option_e.strip()
    ]):
        st.warning("Please provide all five options.")

    else:

        prediction, top3, scores = predict(
            question,
            option_a,
            option_b,
            option_c,
            option_d,
            option_e
        )

        st.success(
            f"Predicted Answer: {prediction}"
        )

        # ------------------------------------------
        # Top 3
        # ------------------------------------------

        st.subheader("Top 3 Predictions")

        for i, option in enumerate(top3, start=1):
            st.write(f"{i}. {option}")

        # ------------------------------------------
        # Confidence
        # ------------------------------------------

        st.subheader("Relative Confidence")

        for option, score in scores.items():

            score = float(score)

            score = max(0.0, min(1.0, score))

            st.write(f"{option}: {score:.2%}")

            st.progress(score)

        # ------------------------------------------
        # Dataset answer
        # ------------------------------------------

        if "correct_answer" in st.session_state:

            correct = st.session_state["correct_answer"]

            st.info(
                f"Dataset Answer: {correct}"
            )

            if prediction == correct:
                st.success("The model prediction matches the dataset answer.")
            else:
                st.warning(
                    "The model prediction does not match the dataset answer."
                )