from src.models.toxicity_baseline import train_model, predict_toxicity

# dummy dataset (temporary)
texts = [
    "I hate you",
    "You are amazing",
    "This is stupid",
    "I love this product",
    "You are an idiot",
    "Great work!"
]

labels = ["Toxic", "Non-Toxic", "Toxic", "Non-Toxic", "Toxic", "Non-Toxic"]

# train model
train_model(texts, labels)

# test predictions
samples = [
    "You are horrible",
    "I like this",
    "This is dumb"
]

for text in samples:
    result = predict_toxicity(text)
    print(text, "->", result)