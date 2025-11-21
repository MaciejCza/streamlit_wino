import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# -----------------------------------------------
# LOAD DATA
# -----------------------------------------------
@st.cache_data
def load_data():
    wine_food = pd.read_csv('wine_food_pairings.csv')
    wine_quality = pd.read_csv('winequality-red.csv')
    return wine_food, wine_quality

wine_food, wine_quality = load_data()

st.title("Wine Data Analysis App 🍷")
st.write("Analiza jakości win oraz parowania wina z jedzeniem.")

# -----------------------------------------------
# TAB MENU
# -----------------------------------------------
tab1, tab2, tab3 = st.tabs(["🍽 Wine & Food Pairing", "🍷 Wine Quality Data", "🤖 Model ML – wine quality"])

# -----------------------------------------------
# TAB 1 – Food Pairing
# -----------------------------------------------
with tab1:
    st.header("Wine & Food Pairing Analysis")
    st.dataframe(wine_food)

    st.subheader("Średnia ocena parowania wg kategorii wina")
    fig, ax = plt.subplots()
    wine_food.groupby("wine_category")["pairing_quality"].mean().plot(kind="bar", ax=ax)
    st.pyplot(fig)

    st.subheader("Filtruj dane")
    wine_type_filter = st.selectbox("Wybierz rodzaj wina", wine_food["wine_type"].unique())
    st.dataframe(wine_food[wine_food["wine_type"] == wine_type_filter])

# -----------------------------------------------
# TAB 2 – Wine Quality
# -----------------------------------------------
with tab2:
    st.header("Wine Quality Dataset")
    st.dataframe(wine_quality)

    st.subheader("Heatmap korelacji")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(wine_quality.corr(), cmap="coolwarm", annot=False, ax=ax)
    st.pyplot(fig)

    variable = st.selectbox("Wybierz zmienną do wykresu", wine_quality.columns)
    fig, ax = plt.subplots()
    sns.histplot(wine_quality[variable], kde=True, ax=ax)
    st.pyplot(fig)

# -----------------------------------------------
# TAB 3 – Model ML
# -----------------------------------------------
with tab3:
    st.header("Model przewidywania jakości wina")

    X = wine_quality.drop("quality", axis=1)
    y = wine_quality["quality"]

    test_size = st.slider("Test size", 0.1, 0.5, 0.2)
    rf_trees = st.slider("Liczba drzew w RandomForest", 50, 500, 200)

    if st.button("Trenuj model"):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        model = RandomForestClassifier(n_estimators=rf_trees)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        st.subheader(f"Accuracy: {acc:.3f}")
        st.text("Classification report:")
        st.text(classification_report(y_test, y_pred))

        st.subheader("Feature importance")
        feat_imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)

        fig, ax = plt.subplots()
        feat_imp.plot(kind="bar", ax=ax)
        st.pyplot(fig)
