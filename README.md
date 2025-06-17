# SentiChat - The Chatbot

SentiChat is an AI-powered customer support chatbot that combines a fine-tuned BERT sentiment analysis model with Google Gemini's conversational AI via LangChain. It analyzes customer reviews to detect sentiment and interact empathetically with users through a conversational interface built using Streamlit.

---

## Project Structure

```
.
├── Fine Tuning/
│   └── fine_tuning_notebook.ipynb   # Jupyter notebook for fine-tuning the sentiment model
│
└── SentiChat - The Chatbot/
    ├── trained_model/                # Directory containing the fine-tuned tokenizer
    ├── chatbot.py                   # Main chatbot application script
    ├── model_weights.pth            # Fine-tuned BERT model weights
```

---

## Features

- **Sentiment Analysis:** Uses a fine-tuned BERT model to classify customer reviews into three sentiment categories: Negative, Neutral, Positive.
- **Interactive Chatbot:** Integrates Google Gemini generative AI via LangChain for empathetic, context-aware conversations based on detected sentiment.
- **Streamlit UI:** Provides an easy-to-use web interface for submitting reviews and chatting with the AI support agent.
- **Conversational Flow Control:** Asks follow-up questions to encourage meaningful interactions and proper conversation closure to make the reviewing experience more comforting and interactive for the customers.

---

## Setup and Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/SentiChat.git
cd SentiChat/SentiChat - The Chatbot
```

### 2. Create and activate a Python virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:**  
> Make sure the `requirements.txt` includes:
> - `torch`
> - `transformers`
> - `streamlit`
> - `langchain`
> - `langchain-google-geni`
> - `deep_translator`
> - `langdetect`

### 4. Add your Google Gemini API key

Create a text file in the same directory as `chatbot.py` (e.g. `google_api_key.txt`) and paste your Gemini API key at the following line in the code. Please note that this model is supported for Gemini-2.0-Flash only. Use the key for this model only.

```text
with open(os.path.join(script_dir, "your_text_file_for_API_key_here.txt"), "r") as f:
```

### 5. Run the chatbot app

```bash
streamlit run chatbot.py
```

The web app will open in your default browser at `http://localhost:8501`.

---

## Usage

1. Enter a customer review in the input box and submit.
2. The chatbot will predict the sentiment of your review (Negative, Neutral, or Positive).
3. Interact with the chatbot, which will respond empathetically and ask follow-up questions (up to 3).
4. The conversation ends with a helpful suggestion and empathetic closing statement.

---

## How it works

- The fine-tuned BERT model (`model_weights.pth`) classifies the sentiment of input text.
- The tokenizer in `trained_model/` prepares input text for the model.
- Google Gemini generative AI is used through LangChain to generate chatbot responses conditioned on sentiment and conversation history.
- Streamlit manages the user interface and conversational state.

---

## Fine Tuning

The `Fine Tuning` directory contains the Jupyter notebook used to fine-tune the BERT sentiment classification model. You can modify and retrain the model there if needed.

---

## Notes

- Ensure your `model_weights.pth` and tokenizer directory (`trained_model`) are correctly placed in the chatbot folder.
- Keep your Google API key private. **Do not commit it to the repository.** Use `.gitignore` to exclude files containing secrets.
- The app currently runs on CPU by default but will use GPU if available.
- The Repo consists of a small demo video of a customer interacting with SentiChat. The responses of the chatbot are quite empathetic and based on the customer sentiment


---

## Contact

For questions, suggestions, or contributions, please open an issue or contact [talalahmad76@gmail.com].

---

**Enjoy chatting with your AI-powered support agent! 🤖🧠**
