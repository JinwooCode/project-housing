import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(page_title="ניתוח מחירי דיור", page_icon="🏠", layout="wide")

st.markdown("""
<style>
    .block-container { max-width: 1100px; }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px; border-radius: 12px; color: white; text-align: center;
    }
    .metric-card h2 { margin: 0; font-size: 2rem; }
    .metric-card p  { margin: 0; font-size: 0.9rem; opacity: 0.85; }
    .prediction-box {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 30px; border-radius: 16px; color: white; text-align: center;
        font-size: 2.2rem; font-weight: bold; margin: 20px 0;
    }
</style>
""", unsafe_allow_html=True)

st.title("🏠 אנליזה וניבוי מחירי דיור")
st.markdown("**מודל רגרסיה לינארית לחיזוי מחירי דירות**")
st.divider()

# ── Load Data ────────────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv("Project_housing_data.csv")

df = load_data()

# ══════════════════════════════════════════════════════════════
# TAB LAYOUT
# ══════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs([
    "🔧 בניית מודל",
    "📈 הערכת ביצועים",
    "🔮 חיזוי דירה חדשה"
])

# ══════════════════════════════════════════════════════════════
# TAB 1 — בניית מודל
# ══════════════════════════════════════════════════════════════
with tab1:
    st.header("בניית מודל רגרסיה לינארית")

    # ── Encoding ─────────────────────────────────────────────
    st.subheader("שלב 1: קידוד משתנים (One-Hot Encoding)")
    st.markdown("""
    שלושה משתנים דורשים קידוד One-Hot (עם השמטת קטגוריה ראשונה כבסיס — `drop_first=True`):

    - **view** (0–4): 5 רמות → 4 עמודות דמה (`view_1`, `view_2`, `view_3`, `view_4`)
    - **condition** (1–5): 5 רמות → 4 עמודות דמה (`condition_2` עד `condition_5`)
    - **floors** (1–3): 3 ערכים → 2 עמודות דמה (`floors_2`, `floors_3`)

    **waterfront** כבר בינארי (0/1) ולכן לא דורש קידוד נוסף.
    """)

    df_encoded = pd.get_dummies(df, columns=['view', 'condition', 'floors'], drop_first=True, dtype=int)

    enc1, enc2 = st.columns(2)
    enc1.metric("עמודות לפני קידוד", df.shape[1])
    enc2.metric("עמודות אחרי קידוד", df_encoded.shape[1])

    # ── Split ────────────────────────────────────────────────
    st.subheader("שלב 2: פיצול לאימון ומבחן (80/20)")

    X = df_encoded.drop('price', axis=1)
    y = df_encoded['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    sp1, sp2 = st.columns(2)
    sp1.metric("סט אימון", f"{X_train.shape[0]:,} תצפיות")
    sp2.metric("סט מבחן", f"{X_test.shape[0]:,} תצפיות")

    # ── Fit ───────────────────────────────────────────────────
    st.subheader("שלב 3: אימון המודל")

    model = LinearRegression()
    model.fit(X_train, y_train)

    coef_df = pd.DataFrame({
        'משתנה': X.columns,
        'מקדם ($)': model.coef_
    }).sort_values('מקדם ($)', key=abs, ascending=False).reset_index(drop=True)

    st.markdown(f"**Intercept (חותך):** `${model.intercept_:,.0f}`")
    st.markdown("**מקדמי המודל** (ממוינים לפי ערך מוחלט):")
    st.dataframe(
        coef_df.style.format({'מקדם ($)': '{:,.0f}'}),
        use_container_width=True
    )

    st.info("""
    **פרשנות מקדמים נבחרים:**
    - **waterfront (+$561,916):** דירה על קו המים שווה בממוצע ~$562K יותר.
    - **view_4 (+$557,417):** נוף ברמה הגבוהה ביותר מוסיף ערך דומה.
    - **sqft_living (+$192):** כל רגל רבוע נוסף של שטח מגורים מוסיף ~$192.
    - **bedrooms (−$55,837):** מקדם שלילי — בשטח זהה, יותר חדרים = חדרים קטנים = מחיר נמוך יותר.
    """)

# ══════════════════════════════════════════════════════════════
# TAB 3 — הערכת ביצועים
# ══════════════════════════════════════════════════════════════
with tab2:
    st.header("הערכת ביצועי המודל")
    st.markdown("הבדיקה נעשית על **סט המבחן** (600 תצפיות) שהמודל לא ראה באימון.")

    y_pred_test = model.predict(X_test)
    r2 = r2_score(y_test, y_pred_test)

    st.metric("R² על סט המבחן", f"{r2:.4f}")

    st.markdown(f"""
    **R² (Coefficient of Determination) = {r2:.4f}**

    המודל מסביר **{r2*100:.1f}%** מהשונות במחירי הדירות.
    ערך R² נע בין 0 ל-1 — ככל שהוא קרוב יותר ל-1, המודל מסביר טוב יותר את השונות בנתונים.
    """)

# ══════════════════════════════════════════════════════════════
# TAB 4 — חיזוי דירה חדשה
# ══════════════════════════════════════════════════════════════
with tab3:
    st.header("🔮 חיזוי מחיר לדירה חדשה")
    st.markdown("בחרו מאפיינים לדירה והמודל יחזה את מחירה.")

    col_left, col_right = st.columns(2)

    with col_left:
        bedrooms    = st.slider("חדרי שינה", 1, 8, 3)
        bathrooms   = st.slider("חדרי אמבטיה", 1, 6, 2)
        sqft_living = st.number_input("שטח מגורים (sqft)", 500, 8000, 1800, step=50)
        sqft_lot    = st.number_input("שטח מגרש (sqft)", 500, 100000, 7000, step=500)
        yr_built    = st.slider("שנת בנייה", 1900, 2015, 1985)
        waterfront  = st.selectbox("חזית מים?", [0, 1], format_func=lambda x: "כן" if x else "לא")

    with col_right:
        floors_val    = st.selectbox("קומות", [1, 2, 3])
        view_val      = st.selectbox("נוף (0=גרוע, 4=מעולה)", [0, 1, 2, 3, 4])
        condition_val = st.selectbox("מצב הדירה (1=גרוע, 5=מעולה)", [1, 2, 3, 4, 5], index=3)
        sqft_above    = st.number_input("שטח מעל הקרקע (sqft)", 400, 7000, min(sqft_living, 7000), step=50)
        sqft_basement = max(0, sqft_living - sqft_above)
        st.metric("שטח מרתף (חישוב אוטומטי)", f"{sqft_basement:,} sqft")

    # Build the feature vector
    new_apt = {
        'bedrooms': bedrooms, 'bathrooms': bathrooms,
        'sqft_living': sqft_living, 'sqft_lot': sqft_lot,
        'waterfront': waterfront,
        'sqft_above': sqft_above, 'sqft_basement': sqft_basement,
        'yr_built': yr_built,
        'view_1': int(view_val == 1), 'view_2': int(view_val == 2),
        'view_3': int(view_val == 3), 'view_4': int(view_val == 4),
        'condition_2': int(condition_val == 2), 'condition_3': int(condition_val == 3),
        'condition_4': int(condition_val == 4), 'condition_5': int(condition_val == 5),
        'floors_2': int(floors_val == 2), 'floors_3': int(floors_val == 3),
    }
    new_apt_df = pd.DataFrame([new_apt])[X.columns]
    predicted = model.predict(new_apt_df)[0]

    st.markdown(
        f'<div class="prediction-box">💰 מחיר חזוי: ${predicted:,.0f}</div>',
        unsafe_allow_html=True
    )

    # Sanity check
    st.subheader("בדיקת סבירות")
    margin = 200
    similar = df[
        (df['bedrooms'] == bedrooms) &
        (df['bathrooms'] == bathrooms) &
        (df['sqft_living'].between(sqft_living - margin, sqft_living + margin))
    ]

    if len(similar) > 0:
        sc1, sc2, sc3 = st.columns(3)
        sc1.metric("דירות דומות בנתונים", len(similar))
        sc2.metric("מחיר ממוצע דומות", f"${similar['price'].mean():,.0f}")
        sc3.metric("טווח מחירים", f"${similar['price'].min():,.0f} – ${similar['price'].max():,.0f}")

        if similar['price'].min() <= predicted <= similar['price'].max():
            st.success("✅ החיזוי נמצא בטווח מחירי הדירות הדומות — התוצאה נראית סבירה.")
        elif predicted < similar['price'].min():
            st.warning("⚠️ החיזוי נמוך מטווח הדירות הדומות.")
        else:
            st.warning("⚠️ החיזוי גבוה מטווח הדירות הדומות — ייתכן שמאפיין כמו נוף/חזית מים מסביר את ההפרש.")
    else:
        st.info("לא נמצאו דירות דומות מספיק בנתונים להשוואה.")
