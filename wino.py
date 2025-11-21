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
# LOAD DATA
# =========================================================
@st.cache_data
def load_data():
    wine_food = pd.read_csv('/mnt/data/wine_food_pairings.csv')
    wine_quality = pd.read_csv('/mnt/data/winequality-red.csv')
    return wine_food, wine_quality

wine_food, wine_quality = load_data()

st.title("🍷 Streamlit Wine Analysis Suite – Extended Version")

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

    st.subheader("Podgląd danych")
    st.dataframe(wine_food)

    # Multi filter
    st.subheader("Filtruj dane")
    col1, col2 = st.columns(2)
    wine_cat = col1.multiselect("Kategoria wina", wine_food["wine_category"].unique())
    food_cat = col2.multiselect("Kategoria jedzenia", wine_food["food_category"].unique())

    filtered = wine_food.copy()
    if wine_cat:
        filtered = filtered[filtered["wine_category"].isin(wine_cat)]
    if food_cat:
        filtered = filtered[filtered["food_category"].isin(food_cat)]

    st.write("Wyniki filtrowania:")
    st.dataframe(filtered)

    # Plot
    st.subheader("Średnia ocena parowania w zależności od jedzenia")
    fig, ax = plt.subplots()
    filtered.groupby("food_category")["pairing_quality"].mean().plot(kind="bar", ax=ax)
    st.pyplot(fig)

# =========================================================
# TAB 2 – WINE QUALITY ANALYSIS
# =========================================================
with tab2:
    st.header("Wine Quality Analysis")

    st.dataframe(wine_quality)

    # Boxplot
    st.subheader("Boxplot wybranej zmiennej")
    col = st.selectbox("Wybierz kolumnę", wine_quality.columns)
    fig, ax = plt.subplots()
    sns.boxplot(x=wine_quality[col], ax=ax)
    st.pyplot(fig)

    # Scatterplot
    st.subheader("Scatterplot – zależności między zmiennymi")
    x_var = st.selectbox("Oś X", wine_quality.columns, index=0)
    y_var = st.selectbox("Oś Y", wine_quality.columns, index=1)
    fig = px.scatter(wine_quality, x=x_var, y=y_var, color="quality")
    st.plotly_chart(fig)

    # 3D scatter
    st.subheader("3D Exploracja jakości")
    x3 = st.selectbox("X (3D)", wine_quality.columns, index=0)
    y3 = st.selectbox("Y (3D)", wine_quality.columns, index=1)
    z3 = st.selectbox("Z (3D)", wine_quality.columns, index=2)
    fig3d = px.scatter_3d(wine_quality, x=x3, y=y3, z=z3, color="quality")
    st.plotly_chart(fig3d)

# =========================================================
# TAB 3 – FILTERS & EXPORT
# =========================================================
with tab3:
    st.header("Zaawansowane filtrowanie & eksport")

    st.write("Wybierz kolumny do filtrowania:")

    df = wine_quality.copy()

    cols = st.multiselect("Kolumny", df.columns)

    for c in cols:
        min_val, max_val = float(df[c].min()), float(df[c].max())
        values = st.slider(f"{c}: zakres", min_val, max_val, (min_val, max_val))
        df = df[df[c].between(values[0], values[1])]

    st.subheader("Przefiltrowane dane")
    st.dataframe(df)

    # Export
    st.download_button(
        "Pobierz jako CSV",
        df.to_csv(index=False).encode("utf-8"),
        "filtered_data.csv",
        "text/csv"
    )

    st.download_button(
        "Pobierz jako Excel",
        df.to_excel("filtered.xlsx", index=False),
        file_name="filtered_data.xlsx"
    )

# =========================================================
# TAB 4 – ML MODELING
# =========================================================
with tab4:
    st.header("Modele ML do przewidywania jakości wina")

    X = wine_quality.drop("quality", axis=1)
    y = wine_quality["quality"]

    # Model choice
    model_type = st.selectbox("Wybierz model", [
        "RandomForest",
        "Logistic Regression"
    ])

    test_size = st.slider("Test size", 0.1, 0.5, 0.2)

    if st.button("Trenuj model"):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        if model_type == "RandomForest":
            model = RandomForestClassifier(n_estimators=300)
        else:
            model = LogisticRegression(max_iter=500)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        st.subheader(f"Accuracy = {accuracy_score(y_test, y_pred):.3f}")
        st.text("Classification report:")
        st.text(classification_report(y_test, y_pred))

        # Feature importance (RF only)
        if model_type == "RandomForest":
            st.subheader("Feature Importance")
            fi = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
            fig, ax = plt.subplots()
            fi.plot(kind="bar", ax=ax)
            st.pyplot(fig)
