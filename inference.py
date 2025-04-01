import torch

from transformers import (DistilBertTokenizer, DistilBertForSequenceClassification)

# Load the tokenizer and model from Hugging Face
MODEL_NAME = "ajikadev/circleci-nlp-model"
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()


def predict_sentiment(text):
    inputs = tokenizer(text, padding="max_length", truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    prediction = torch.argmax(logits, dim=-1).item()
    sentiment = "positive" if prediction == 1 else "negative"
    return sentiment

if __name__ == "__main__":
    text = "I absolutely loved this movie! It was fantastic."
    result = predict_sentiment(text)
    print(f"Text: {text}")
    print(f"Sentiment: {result}")