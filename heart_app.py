import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(page_title="ניבוי התקפי לב", page_icon="❤️", layout="wide")

st.markdown("""
<style>
    .block-container { max-width: 1100px; }
    .prediction-box {
        padding: 30px; border-radius: 16px; color: white; text-align: center;
        font-size: 2.2rem; font-weight: bold; margin: 20px 0;
    }
    .high-risk { background: linear-gradient(135deg, #cb2d3e 0%, #ef473a 100%); }
    .low-risk  { background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); }
</style>
""", unsafe_allow_html=True)

st.title("❤️ אנליזה וניבוי של התקפי לב")
st.markdown("**מודל רגרסיה לוגיסטית לחיזוי הסיכון להתקף לב**")
st.divider()

# ── Load Data ────────────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv("Project_heart_data.csv")

df = load_data()

# ══════════════════════════════════════════════════════════════
# TAB LAYOUT
# ══════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs([
    "🔧 בניית מודל",
    "📈 הערכת ביצועים",
    "🔮 חיזוי מטופל חדש"
])

# ══════════════════════════════════════════════════════════════
# TAB 1 — בניית מודל
# ══════════════════════════════════════════════════════════════
with tab1:
    st.header("בניית מודל רגרסיה לוגיסטית")

    # ── Encoding ─────────────────────────────────────────────
    st.subheader("שלב 1: קידוד משתנים קטגוריאליים (One-Hot Encoding)")
    st.markdown("""
    שני משתנים קטגוריאליים עם יותר משתי קטגוריות דורשים קידוד One-Hot
    (עם השמטת קטגוריה ראשונה כבסיס — `drop_first=True`):

    - **cp** (סוג כאב בחזה, 0–3): 4 קטגוריות → 3 עמודות דמה (`cp_1`, `cp_2`, `cp_3`)
    - **rest_ecg** (תוצאות א.ק.ג במנוחה, 0–2): 3 קטגוריות → 2 עמודות דמה (`rest_ecg_1`, `rest_ecg_2`)

    הערכים של משתנים אלה הם שמות של קטגוריות ולא סדר או כמות, ולכן אין להשאיר אותם כמספרים.

    **sex**, **exang** ו-**fbs** כבר בינאריים (0/1) ולכן לא דורשים קידוד נוסף.
    **ca** הוא ספירה של כלי דם ולכן נשאר מספרי.
    """)

    df_encoded = pd.get_dummies(df, columns=['cp', 'rest_ecg'], drop_first=True, dtype=int)

    enc1, enc2 = st.columns(2)
    enc1.metric("עמודות לפני קידוד", df.shape[1])
    enc2.metric("עמודות אחרי קידוד", df_encoded.shape[1])

    # ── Split ────────────────────────────────────────────────
    st.subheader("שלב 2: פיצול לאימון ומבחן (80/20)")

    X = df_encoded.drop('target', axis=1)
    y = df_encoded['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    sp1, sp2 = st.columns(2)
    sp1.metric("סט אימון", f"{X_train.shape[0]:,} תצפיות")
    sp2.metric("סט מבחן", f"{X_test.shape[0]:,} תצפיות")

    # ── Fit ───────────────────────────────────────────────────
    st.subheader("שלב 3: אימון המודל")

    model = LogisticRegression(max_iter=2000)
    model.fit(X_train, y_train)

    coef_df = pd.DataFrame({
        'משתנה': X.columns,
        'מקדם': model.coef_[0]
    }).sort_values('מקדם', key=abs, ascending=False).reset_index(drop=True)

    st.markdown(f"**Intercept (חותך):** `{model.intercept_[0]:.3f}`")
    st.markdown("**מקדמי המודל** (ממוינים לפי ערך מוחלט):")
    st.dataframe(
        coef_df.style.format({'מקדם': '{:.3f}'}),
        use_container_width=True
    )

# ══════════════════════════════════════════════════════════════
# TAB 2 — הערכת ביצועים
# ══════════════════════════════════════════════════════════════
with tab2:
    st.header("הערכת ביצועי המודל")
    st.markdown(f"הבדיקה נעשית על **סט המבחן** ({X_test.shape[0]} תצפיות) שהמודל לא ראה באימון.")

    y_pred_test = model.predict(X_test)
    accuracy  = accuracy_score(y_test, y_pred_test)
    precision = precision_score(y_test, y_pred_test)
    recall    = recall_score(y_test, y_pred_test)
    f1        = f1_score(y_test, y_pred_test)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{accuracy:.4f}")
    m2.metric("Precision", f"{precision:.4f}")
    m3.metric("Recall", f"{recall:.4f}")
    m4.metric("F1", f"{f1:.4f}")

    st.markdown(f"""
    - **Accuracy = {accuracy:.4f}:** המודל סיווג נכון **{accuracy*100:.1f}%** מכלל המטופלים בסט המבחן.

    - **Precision = {precision:.4f}:** מתוך המטופלים שהמודל סיווג כבעלי סיכון גבוה, **{precision*100:.1f}%** אכן בסיכון גבוה.

    - **Recall = {recall:.4f}:** מתוך המטופלים שבאמת בסיכון גבוה, המודל זיהה **{recall*100:.1f}%**.

    - **F1 = {f1:.4f}:** ממוצע הרמוני של Precision ו-Recall.
    """)

    st.subheader("איזו מטריקה חשובה יותר בהקשר הרפואי?")
    st.info(f"""
    **המטריקה החשובה ביותר היא Recall.**

    בהקשר הרפואי שתי הטעויות האפשריות אינן שוות במחירן:

    - **False Negative** — המודל קובע "סיכון נמוך" למטופל שבאמת בסיכון גבוה. המטופל נשלח הביתה
      ללא בירור וללא טיפול, והתוצאה עלולה להיות התקף לב ואף מוות.

    - **False Positive** — המודל קובע "סיכון גבוה" למטופל שבאמת בסיכון נמוך. המטופל יעבור
      בדיקות נוספות מיותרות — עלות של זמן, כסף ודאגה, אך לא סכנת חיים.

    Recall מודד בדיוק את שיעור החולים האמיתיים שהמודל הצליח לזהות, כלומר ככל שהוא גבוה יותר
    יש פחות False Negatives. מכיוון שהמטרה היא להציל חיי אדם, עדיף לשלוח מטופל בריא לבדיקה מיותרת
    מאשר לפספס מטופל בסיכון — ולכן Recall חשוב יותר מ-Precision ומ-Accuracy.

    במודל שלנו Recall = {recall:.4f}, כלומר המודל מפספס כ-**{(1-recall)*100:.1f}%** מהמטופלים שבסיכון גבוה.
    """)

# ══════════════════════════════════════════════════════════════
# TAB 3 — חיזוי מטופל חדש
# ══════════════════════════════════════════════════════════════
with tab3:
    st.header("🔮 חיזוי למטופל חדש")
    st.markdown("בחרו מאפיינים למטופל והמודל יחזה את הסיכון שלו להתקף לב.")

    cp_labels = {
        0: "0 — אסימפטומטי",
        1: "1 — תעוקה טיפוסית",
        2: "2 — תעוקה לא טיפוסית",
        3: "3 — כאב שאינו תעוקתי",
    }
    ecg_labels = {
        0: "0 — תקין",
        1: "1 — הפרעה בגל ST-T",
        2: "2 — היפרטרופיה של חדר שמאל",
    }

    col_left, col_right = st.columns(2)

    with col_left:
        age     = st.slider("גיל", 29, 80, 58)
        sex     = st.selectbox("מין", [1, 0], format_func=lambda x: "גבר" if x else "אישה")
        cp_val  = st.selectbox("סוג כאב בחזה (cp)", [0, 1, 2, 3], format_func=lambda x: cp_labels[x])
        exang   = st.selectbox("תעוקה במאמץ (exang)", [0, 1], index=1, format_func=lambda x: "כן" if x else "לא")
        ca      = st.selectbox("מספר כלי דם ראשיים (ca)", [0, 1, 2, 3], index=2)

    with col_right:
        trtbps  = st.slider("לחץ דם במנוחה (mm Hg)", 90, 200, 140)
        chol    = st.slider("כולסטרול (mg/dl)", 120, 570, 260)
        fbs     = st.selectbox("סוכר בצום מעל 120 mg/dl (fbs)", [0, 1], format_func=lambda x: "כן" if x else "לא")
        ecg_val = st.selectbox("א.ק.ג במנוחה (rest_ecg)", [0, 1, 2], format_func=lambda x: ecg_labels[x])
        thalach = st.slider("דופק מרבי שהושג (thalach)", 70, 205, 130)

    # Build the feature vector
    new_patient = {
        'age': age, 'sex': sex, 'exang': exang, 'ca': ca,
        'trtbps': trtbps, 'chol': chol, 'fbs': fbs, 'thalach': thalach,
        'cp_1': int(cp_val == 1), 'cp_2': int(cp_val == 2), 'cp_3': int(cp_val == 3),
        'rest_ecg_1': int(ecg_val == 1), 'rest_ecg_2': int(ecg_val == 2),
    }
    new_patient_df = pd.DataFrame([new_patient])[X.columns]
    predicted = model.predict(new_patient_df)[0]
    probability = model.predict_proba(new_patient_df)[0][1]

    if predicted == 1:
        st.markdown(
            f'<div class="prediction-box high-risk">חיזוי: 1 — סיכוי גבוה יותר להתקף לב ({probability*100:.1f}%)</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div class="prediction-box low-risk">חיזוי: 0 — סיכוי נמוך יותר להתקף לב ({probability*100:.1f}%)</div>',
            unsafe_allow_html=True
        )

    # Clinical interpretation
    st.subheader("המשמעות הקלינית של התוצאה")
    if predicted == 1:
        st.warning(f"""
        המודל מעריך הסתברות של **{probability*100:.1f}%** שהמטופל שייך לקבוצת הסיכון הגבוה (target = 1),
        ולכן מסווג אותו כבעל **סיכוי גבוה יותר להתקף לב**.

        המשמעות הקלינית: המטופל דומה במאפייניו למטופלים שסווגו בעבר בסיכון גבוה. אין זו אבחנה —
        זהו סימן אזהרה שמצדיק הפניה לבירור קרדיולוגי נוסף (כגון בדיקת מאמץ או צנתור) ומעקב צמוד,
        ולא שחרור של המטופל ללא בדיקה.
        """)
    else:
        st.success(f"""
        המודל מעריך הסתברות של **{probability*100:.1f}%** שהמטופל שייך לקבוצת הסיכון הגבוה (target = 1),
        ולכן מסווג אותו כבעל **סיכוי נמוך יותר להתקף לב**.

        המשמעות הקלינית: המטופל דומה במאפייניו למטופלים שסווגו בעבר בסיכון נמוך. אין זו אבחנה
        ואין זו הבטחה שלא יהיה התקף לב — המודל מפספס חלק מהמטופלים שבסיכון (ראו Recall), ולכן
        ההחלטה הסופית נשארת בידי הרופא, בהתאם לתמונה הקלינית המלאה.
        """)
