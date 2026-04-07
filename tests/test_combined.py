from src.layers.combined_insight import generate_insight

cases = [
    ("Positive", "Non-Toxic"),
    ("Negative", "Toxic"),
    ("Positive", "Toxic"),
    ("Neutral", "Non-Toxic")
]

for s, t in cases:
    print(s, "+", t, "->", generate_insight(s, t))