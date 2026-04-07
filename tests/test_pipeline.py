import pandas as pd
from src.data.pipeline import clean_text, preprocess_dataframe

data = {
    "text": [
        "I LOVE this product!!! 😍",
        "This is the worst thing ever...",
        "Visit http://example.com now!",
        None
    ]
}

df = pd.DataFrame(data)

df_clean = preprocess_dataframe(df, "text")

print(df_clean)