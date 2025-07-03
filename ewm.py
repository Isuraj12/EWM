import streamlit as st
import pandas as pd
import numpy as np

# -----------------------------
# Entropy Weight Method Function
# -----------------------------
def entropy_weight_method(df):
    norm_df = (df - df.min()) / (df.max() - df.min() + 1e-10)
    P = norm_df.div(norm_df.sum(axis=0), axis=1)
    k = 1 / np.log(len(df))
    entropy = -k * (P * np.log(P + 1e-10)).sum(axis=0)
    d = 1 - entropy
    weights = d / d.sum()
    return np.round(weights*100 , 2)

# -----------------------------
# Streamlit App
# -----------------------------
st.set_page_config(page_title="Entropy Weight Calculator", layout="centered")

st.title(" Entropy Weight Method (EWM)")
st.markdown("""
This tool calculates variable weights based on their information content using the **Entropy Weight Method**.
""")

# Sidebar user input
st.sidebar.header("🔧 Settings")
n_rows = st.sidebar.slider("Number of Rows", 3, 20, 5)
n_cols = st.sidebar.slider("Number of Columns", 2, 10, 4)
random_seed = st.sidebar.number_input("Random Seed", value=42, step=1)

# Generate dataset
np.random.seed(random_seed)
data = np.random.randint(100, 10000, size=(n_rows, n_cols))
columns = [f"Var_{i+1}" for i in range(n_cols)]
df = pd.DataFrame(data, columns=columns)

st.subheader(" Generated Random Data")
st.dataframe(df)

# Calculate EWM
weights = entropy_weight_method(df)

st.subheader("📈 Entropy Weights")
st.write(weights.round(4))

# Optional bar chart
st.bar_chart(weights)
