import os
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# =========================================================
# SAFE LOADING FUNCTION
# =========================================================
def load_csv_safely(filename, uploaded_file=None):
    """
    Kolejność:
    1. Użytkownik wgrał plik → użyj tego
    2. Plik istnieje w katalogu projektu
    3. Plik istnieje w /mnt/data/
    4. Zwróć None i pokaż błąd
    """
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)

    # repo folder (Streamlit Cloud)
    if os.path.exists(filename):
        return pd.read_csv(filename)

    # Jupyter / ChatGPT environment
    mnt_path = f"/mnt/data/{filename}"
    if os.path.exists(mnt_path):
        return pd.read_csv(mnt_path)

    st.error(f"❌ Nie znaleziono pliku: {filename}. Wgraj go poniżej.")
    return None


# =========================================================
# STREAMLIT UI
# =========================================================
st.title("🍷 Streamlit Wine Analysis Suite — wersja stabilna")

st.write("Aplikacja automatycznie wykrywa pliki lub umożliwia ich przesłanie.")


# =========================================================
# FILE UPLOADERS
# =========================================================
st.sidebar.header("📥 Wgraj pliki danych (opcjonalnie)")

uploaded_food = st.sidebar.file_uploader("wine_food_pairings.csv", type=["csv"])
uploaded_quality = st.sidebar.file_uploader("winequality-red.csv", type=["csv"])


# =========================================================
# LOAD DATA
# =========================================================
wine_food = load_csv_safely("wine_food_pairings.csv", uploaded_food)
wine_quality = load_csv_safely("winequality-red.csv", uploaded_quality)


# Jeśli któregoś pliku nie ma → nie renderuj reszty aplikacji
if wine_food is None or wine_quality is None:
    st.warning("➡️ Wgraj brakujące pliki aby kontynuować.")
    st.stop()


# =========================================================
# TABS
# =========================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "🍽 Food & Wine Pairing",
    "📊 Wine Quality – Analysis",
    "🔧 Data Filters & Export",
    "🤖 ML Models"
])


# =========================================================
# TAB 1 – FOOD PAIRING
# =========================================================
with tab1:
    st.header("Food & Wine Pairing Analysis")

    st.dataframe(wine_food)

    st.subheader("Filtruj dane")
    col1, col2 = st.columns(2)

    sel_wine = col1.multiselect("Kategoria wina", wine_food["wine_category"].unique())
    sel_food = col2.multiselect("Kategoria jedzenia", wine_food["food_category"].unique())

    df = wine_food.copy()
    if sel_wine:
        df = df[df["wine_category"].isin(sel_wine)]
    if sel_food:
        df = df[df["food_category"].isin(sel_food)]

    st.write("🔎 Wyniki filtrowania:")
    st.dataframe(df)

    st.subheader("Średnia ocena parowania wg rodzaju jedzenia")
    fig, ax = plt.subplots()
    df.groupby("food_category")["pairing_quality"].mean().plot(kind="bar", ax=ax)
    st.pyplot(fig)


# =========================================================
# TAB 2 – WINE QUALITY ANALYSIS
# =========================================================
with tab2:
    st.header("Wine Quality Dataset")

    st.dataframe(wine_quality)

    st.subheader("📌 Korelacje")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(wine_quality.corr(), cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    st.subheader("📊 Boxplot zmiennej")
    var = st.selectbox("Wybierz kolumnę:", wine_quality.columns)
    fig, ax = plt.subplots()
    sns.boxplot(x=wine_quality[var], ax=ax)
    st.pyplot(fig)

    st.subheader("Scatterplot")
    x = st.selectbox("X", wine_quality.columns)
    y = st.selectbox("Y", wine_quality.columns, index=1)
    st.plotly_chart(px.scatter(wine_quality, x=x, y=y, color="quality"))

    st.subheader("3D Exploracja")
    x3 = st.selectbox("X (3D)", wine_quality.columns, index=0)
    y3 = st.selectbox("Y (3D)", wine_quality.columns, index=1)
    z3 = st.selectbox("Z (3D)", wine_quality.columns, index=2)

    st.plotly_chart(px.scatter_3d(wine_quality, x=x3, y=y3, z=z3, color="quality"))


# =========================================================
# TAB 3 — FILTERS & EXPORT
# =========================================================
with tab3:
    st.header("Zaawansowane filtrowanie & eksport danych")

    df = wine_quality.copy()
    cols = st.multiselect("Wybierz kolumny do filtrowania:", df.columns)

    for c in cols:
        min_val, max_val = float(df[c].min()), float(df[c].max())
        r = st.slider(f"{c} — zakres", min_val, max_val, (min_val, max_val))
        df = df[df[c].between(r[0], r[1])]

    st.subheader("Wynik filtrowania")
    st.dataframe(df)

    st.download_button("⬇ Pobierz CSV", df.to_csv(index=False), "filtered.csv", "text/csv")


# =========================================================
# TAB 4 — MACHINE LEARNING
# =========================================================
with tab4:
    st.header("Modele ML — przewidywanie jakości wina")

    X = wine_quality.drop("quality", axis=1)
    y = wine_quality["quality"]

    model_choice = st.selectbox("Wybierz model", [
        "RandomForest",
        "Logistic Regression"
    ])

    test_size = st.slider("Test size", 0.1, 0.5, 0.2)

    if st.button("Trenuj model"):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        # Model selection
        if model_choice == "RandomForest":
            model = RandomForestClassifier(n_estimators=300)
        else:
            model = LogisticRegression(max_iter=500)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        st.subheader(f"🎯 Accuracy: {accuracy_score(y_test, y_pred):.3f}")
        st.text(classification_report(y_test, y_pred))

        if model_choice == "RandomForest":
            st.subheader("Feature importance")
            fi = pd.Series(model.feature_importances_, index=X.columns).sort_values()
            fig, ax = plt.subplots()
            fi.plot(kind="barh", ax=ax)
            st.pyplot(fig)
