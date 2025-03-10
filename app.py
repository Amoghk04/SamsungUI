import streamlit as st
import torch
from transformers import XLMRobertaTokenizerFast, AutoTokenizer, XLMRobertaModel
from torch.nn.functional import softmax
import json


# Define the JointIntentSlotModel class (must match the training script)
class JointIntentSlotModel(torch.nn.Module):
    def __init__(self, model_name, num_intent_labels, num_slot_labels):
        super(JointIntentSlotModel, self).__init__()
        self.encoder = XLMRobertaModel.from_pretrained(model_name)
        self.intent_classifier = torch.nn.Sequential(
            torch.nn.Linear(self.encoder.config.hidden_size, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(256, num_intent_labels)
        )
        self.slot_classifier = torch.nn.Sequential(
            torch.nn.Linear(self.encoder.config.hidden_size, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(256, num_slot_labels)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        cls_output = sequence_output[:, 0, :]  # Take [CLS] token

        intent_logits = self.intent_classifier(cls_output)
        slot_logits = self.slot_classifier(sequence_output)

        return {"intent_logits": intent_logits, "slot_logits": slot_logits}

# Load necessary artifacts
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("final_model/tokenizer", use_fast=True)

# Load label mappings
with open("final_model/label_mappings.json", "r") as f:
    label_mappings = json.load(f)
slot_label_to_id = label_mappings["slot_label_to_id"]
intent_label_to_id = label_mappings["intent_label_to_id"]
id_to_slot_label = label_mappings["id_to_slot_label"]
id_to_intent_label = label_mappings["id_to_intent_label"]

# Load trained model
checkpoint = torch.load("final_model/best_model.pt", map_location=device)
num_intent_labels = len(intent_label_to_id)
num_slot_labels = len(slot_label_to_id)

model = JointIntentSlotModel("xlm-roberta-base", num_intent_labels, num_slot_labels).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()


# Define inference function
def infer(sentence):
    """
    Perform inference on a single sentence.
    Args:
        sentence (str): Input sentence to predict intent and slots.
    Returns:
        dict: Predicted intent and slot labels.
    """
    tokens = sentence.split()  # Tokenize based on whitespace

    encodings = tokenizer(tokens,
                          truncation=True,
                          padding="max_length",
                          max_length=128,
                          is_split_into_words=True,
                          return_tensors="pt")

    input_ids = encodings["input_ids"].to(device)
    attention_mask = encodings["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        intent_logits = outputs["intent_logits"]
        slot_logits = outputs["slot_logits"]

    # Decode intent
    intent_probs = softmax(intent_logits, dim=-1)
    intent_idx = torch.argmax(intent_probs, dim=-1).item()
    predicted_intent = id_to_intent_label[str(intent_idx)]

    # Decode slot labels
    slot_probs = softmax(slot_logits, dim=-1)
    slot_idxs = torch.argmax(slot_probs, dim=-1).squeeze(0).tolist()

    word_ids = encodings.word_ids()  # Align tokens with words
    predicted_slots = []

    for i, word_id in enumerate(word_ids):
        if word_id is None:  # Skip special tokens or padding
            continue
        if word_id < len(tokens):  # Ensure alignment with original words
            slot_idx = slot_idxs[i]
            predicted_slots.append(id_to_slot_label[str(slot_idx)])

    return {
        "tokens": tokens,
        "predicted_intent": predicted_intent,
        "predicted_slots": predicted_slots
    }


# Streamlit UI
# Streamlit UI for WhatsApp-like design
st.title("Samsung Chatbot")
st.subheader("Intent and Slot Prediction")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if user_input := st.chat_input("Type your message here"):
    # Display user message in chat bubble
    st.chat_message("user").markdown(user_input)
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Perform inference and generate response
    result = infer(user_input)
    
    # Format assistant response
    response_text = f"**Intent:** {result['predicted_intent']}\n\n**Slots:**\n"
    for token, slot in zip(result["tokens"], result["predicted_slots"]):
        if slot != "O":
            response_text += f"- **{token}**: {slot}\n"
    
    if not any(slot != "O" for slot in result["predicted_slots"]):
        response_text += "No slot predictions."
    
    # Display assistant response in chat bubble
    with st.chat_message("assistant"):
        st.markdown(response_text)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response_text})