import os
import pickle
import torch
import torch.nn.functional as F
import streamlit as st

from model import BiLSTMClassifier

# =====================================================
# Page Configuration
# =====================================================

st.set_page_config(
    page_title="Smart MCQ Solver",
    page_icon="🧠",
    layout="centered"
)

# =====================================================
# Device
# =====================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================================================
# Paths
# =====================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "best_lstm.pt")
TOKENIZER_PATH = os.path.join(BASE_DIR, "tokenizer.pkl")
LABEL_PATH = os.path.join(BASE_DIR, "id2label.pkl")

# =====================================================
# Load Tokenizer
# =====================================================

with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)

# =====================================================
# Load Labels
# =====================================================

with open(LABEL_PATH, "rb") as f:
    id2label = pickle.load(f)

# =====================================================
# Load Model
# =====================================================

model = BiLSTMClassifier(
    vocab_size=tokenizer.vocab_size,
    embedding_dim=256,
    hidden_dim=256,
    num_layers=2,
    num_classes=5
)

state_dict = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=True
)

model.load_state_dict(state_dict)
model.to(device)
model.eval()

# =====================================================
# Prediction Function
# =====================================================

def predict(question, option_a, option_b, option_c, option_d, option_e):

    text = f"""
Question:
{question}

Option A:
{option_a}

Option B:
{option_b}

Option C:
{option_c}

Option D:
{option_d}

Option E:
{option_e}
"""

    encoding = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=256,
        return_tensors="pt"
    )

    input_ids = encoding["input_ids"].to(device)

    with torch.no_grad():

        logits = model(input_ids)

        probs = F.softmax(logits, dim=1)

        values, indices = torch.topk(probs, k=3)

    return values, indices


# =====================================================
# Streamlit UI
# =====================================================

st.title("🧠 Smart MCQ Solver")
st.subheader("Deep Learning & Generative AI Project")

st.write(
    """
Predict the **Top-3 answers** for a Multiple Choice Question using a
**Bidirectional LSTM (BiLSTM)** model.
"""
)

st.divider()

question = st.text_area(
    "Question",
    height=120
)

option_a = st.text_input("Option A")
option_b = st.text_input("Option B")
option_c = st.text_input("Option C")
option_d = st.text_input("Option D")
option_e = st.text_input("Option E")

st.divider()

if st.button("🚀 Predict Top-3 Answers", use_container_width=True):

    if (
        question.strip() == "" or
        option_a.strip() == "" or
        option_b.strip() == "" or
        option_c.strip() == "" or
        option_d.strip() == "" or
        option_e.strip() == ""
    ):

        st.warning("Please fill in all the fields.")

    else:

        values, indices = predict(
            question,
            option_a,
            option_b,
            option_c,
            option_d,
            option_e
        )

        options = {
            "A": option_a,
            "B": option_b,
            "C": option_c,
            "D": option_d,
            "E": option_e
        }

        medals = ["🥇", "🥈", "🥉"]

        st.success("Prediction Complete!")

        st.subheader("Top-3 Predictions")

        for i in range(3):

            label = id2label[indices[0][i].item()]
            confidence = values[0][i].item()

            st.markdown(f"### {medals[i]} Option {label}")

            st.write(f"**Answer:** {options[label]}")

            st.progress(confidence)

            st.write(
                f"**Confidence:** {confidence*100:.2f}%"
            )

            st.divider()

st.markdown("---")
st.caption(
    "Developed using PyTorch • BiLSTM • DistilBERT Tokenizer"
)
