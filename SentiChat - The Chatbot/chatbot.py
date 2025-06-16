import os
import re
import torch
import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

# Setting the script directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Load the Gemini API key. This is for integration with LLM
with open(os.path.join(script_dir, "your_text_file_for_API_key_here.txt"), "r") as f:
    GOOGLE_API_KEY = f.read().strip()

# Defining paths to the fine tuned Model and Tokenizer
MODEL_PATH = os.path.join(script_dir, "model_weights.pth")
TOKENIZER_PATH = os.path.join(script_dir, "trained_model")

# Loading the tokenizer and the state dictionary
tokenizer = BertTokenizer.from_pretrained(TOKENIZER_PATH)
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
model.eval()

# GPU based if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# The output label map
label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}

# This function will take in the text and predict the sentiment of the customer review
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    inputs = {key: val.to(device) for key, val in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=1).item()
    return label_map[prediction]

# === Helper: Extract aspect categories from text ===
def extract_aspects(text):
    aspects = []
    if re.search(r'food|taste|meal|menu|dish', text, re.IGNORECASE):
        aspects.append("Food")
    if re.search(r'service|waiter|staff|employee', text, re.IGNORECASE):
        aspects.append("Service")
    if re.search(r'price|cost|value|expensive|cheap', text, re.IGNORECASE):
        aspects.append("Price")
    if re.search(r'clean|hygiene|environment|atmosphere', text, re.IGNORECASE):
        aspects.append("Ambience")
    return ", ".join(aspects) if aspects else "General"

# Using Langchain for Google Gemini for interactive response with the customer
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=GOOGLE_API_KEY)
template = PromptTemplate.from_template(
    "You are a chatbot analyzing customer reviews and interacting with them. Predict the user's sentiment and respond empathetically. The current sentiment of the customer is {sentiment}"
    "Ask one follow-up question at a time. You have {questions_left} follow up questions left. If no follow up questions left, close the conversations"
    "Close the conversation with a good suggestion and an empathetic response.\n\n"
    "{history}\n\nUser: {user_input}\nAssistant:"
)
llm_chain = LLMChain(llm=llm, prompt=template)

# Initializing a LLM Chain for the summary as well
summary_prompt = PromptTemplate.from_template(
    "Summarize the customer's feedback based on the following conversation history. Highlight key concerns and suggestions.\n\n{history}\n\nSummary:"
)
summarizer_chain = LLMChain(llm=llm, prompt=summary_prompt)
########################################################################################################################################################################

# Using streamlit for User interface with Python
st.set_page_config(page_title="Your Customer Support Agent", page_icon="🤖")
st.title("🧠  SentiChat: Your AI Support Specialist")

# Initializing some variables to be used during the interaction with customer
if "messages" not in st.session_state:
    st.session_state.messages = []

if "questions_left" not in st.session_state:
    st.session_state.questions_left = 3

if "review_submitted" not in st.session_state:
    st.session_state.review_submitted = False


# Reset Button to reset the chat
if st.button("🔄 Reset Conversation"):
    st.session_state.clear()
    st.rerun()

sentiment_tmp = ""
# Step - 1: Take the input from the customer and analyze it 
if not st.session_state.review_submitted:
    st.subheader("✍️ Submit Your Review")
    user_review = st.text_area("Enter your review to get started:")

    if st.button("Submit Review"):
        if not user_review.strip():
            st.warning("Please enter a review.")
        else:
            sentiment = predict_sentiment(user_review)
            aspects = extract_aspects(user_review)
            user_msg = f"**(Sentiment: _{sentiment}_) | Aspects: {aspects}):** {user_review}"
            st.session_state.messages.append({"role": "user", "content": user_msg})

            # with st.chat_message("user"):
            #     st.markdown(user_msg)

            history = f"User: {user_msg}"
            with st.spinner("🤖 Assistant is replying..."):
                response = llm_chain.run({
                    "sentiment": sentiment,
                    "user_input": user_review,
                    "history": history,
                    "questions_left": st.session_state.questions_left
                })

            st.session_state.messages.append({"role": "assistant", "content": response})
            # with st.chat_message("assistant"):
            #     st.markdown(response)

            st.session_state.questions_left -= 1
            sentiment_tmp = sentiment
            st.session_state.review_submitted = True

# Step 2 - Chat with the customer based on the review submitted and the sentiment of the customer
if st.session_state.review_submitted:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("💬 Continue the conversation:"):
        user_msg = prompt
        st.session_state.messages.append({"role": "user", "content": user_msg})
        with st.chat_message("user"):
            st.markdown(user_msg)

        history = ""
        for m in st.session_state.messages:
            prefix = "User" if m["role"] == "user" else "Assistant"
            content = m["content"]
            history += f"{prefix}: {content}\n"

        with st.spinner("🤖 Assistant is replying..."):
            response = llm_chain.run({
                "sentiment": sentiment_tmp,
                "user_input": prompt,
                "history": history.strip(),
                "questions_left": st.session_state.questions_left
            })

        st.session_state.questions_left = max(0, st.session_state.questions_left - 2)
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)
            
# Section for Summarizing the review
if st.session_state.review_submitted and st.button("📋 Summarize Conversation"):
    history = ""
    for m in st.session_state.messages:
        prefix = "User" if m["role"] == "user" else "Assistant"
        history += f"{prefix}: {m['content']}\n"

    with st.spinner("📄 Generating summary..."):
        try:
            summary = summarizer_chain.run({"history": history.strip()})
        except Exception as e:
            summary = "⚠️ Summary generation failed."
            st.error(f"Gemini error: {e}")

    st.subheader("💡 Summary of Feedback")
    st.markdown(summary)