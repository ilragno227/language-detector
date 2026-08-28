import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

LANG_MAP = {
    "ar": "Arabic",
    "bg": "Bulgarian",
    "de": "German",
    "el": "Modern Greek",
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "hi": "Hindi",
    "it": "Italian",
    "ja": "Japanese",
    "nl": "Dutch",
    "pl": "Polish",
    "pt": "Portuguese",
    "ru": "Russian",
    "sw": "Swahili",
    "th": "Thai",
    "tr": "Turkish",
    "ur": "Urdu",
    "vi": "Vietnamese",
    "zh": "Chinese",
}


# 1. Cache model loading so it only runs once at startup
@st.cache_resource
def load_model():
  model_name = "papluca/xlm-roberta-base-language-detection"
  tokenizer = AutoTokenizer.from_pretrained(model_name)
  model = AutoModelForSequenceClassification.from_pretrained(model_name)
  return tokenizer, model


tokenizer, model = load_model()


# 2. Inference function
def detect_language(text):
  if len(text.strip()) == 0:
    return None

  inputs = tokenizer(
      text, return_tensors="pt", truncation=True, padding=True
  )

  with torch.no_grad():
    outputs = model(**inputs)

  probs = torch.softmax(outputs.logits, dim=1)[0]
  id2lang = model.config.id2label

  val, idx = torch.max(probs, dim=0)

  iso_code = id2lang[idx.item()]
  lang_name = LANG_MAP.get(iso_code, iso_code)

  return lang_name, val.item()


# 3. Streamlit Web UI Layout
st.title("🌐 Language Detection App")
st.write("Enter text below to detect its language:")

# Input text area
user_text = st.text_area(
    "Input Text", placeholder="e.g., Ich liebe dich", height=120
)

# Submit button logic
if st.button("Detect Language", type="primary"):
  if not user_text.strip():
    st.warning("Please enter some text first.")
  else:
    result = detect_language(user_text)
    if result:
      lang_name, confidence = result
      st.success(f"**Detected Language:** {lang_name}")
      st.info(f"**Confidence Score:** {confidence:.2%}")