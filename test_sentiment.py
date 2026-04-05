from src.models.sentiment_baseline import predict_sentiment

samples = [
    "I absolutely love this!",
    "This is terrible and I hate it",
    "It is okay, nothing special"
]

for text in samples:
    result = predict_sentiment(text)
    print(text, "->", result)