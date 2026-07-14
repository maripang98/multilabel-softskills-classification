import joblib

from lime.lime_text import LimeTextExplainer

from preprocessing import preprocess_text


# ==================================================
# Load Model dan TF-IDF
# ==================================================

loaded_model = joblib.load(
    "svm_model.pkl"
)

loaded_tfidf = joblib.load(
    "tfidf_vectorizer.pkl"
)


# ==================================================
# Label Soft Skills
# ==================================================

LABEL_NAMES = [
    "Komunikasi",
    "Kerja Sama Tim",
    "Kecerdasan Emosional"
]


# ==================================================
# Prediksi Probabilitas
# ==================================================

def predict_proba(text):
    """
    Menghasilkan probabilitas prediksi
    untuk setiap label soft skills.
    """

    processed_text = preprocess_text(
        text
    )

    vector = loaded_tfidf.transform(
        [processed_text]
    )

    probabilities = loaded_model.predict_proba(
        vector
    )[0]

    return probabilities


# ==================================================
# Prediksi Label
# ==================================================

def predict_softskills(text):
    """
    Menghasilkan label hasil prediksi
    beserta probabilitasnya.
    """

    processed_text = preprocess_text(
        text
    )

    vector = loaded_tfidf.transform(
        [processed_text]
    )

    prediction = loaded_model.predict(
        vector
    )[0]

    probabilities = loaded_model.predict_proba(
        vector
    )[0]

    predicted_labels = []

    for label, value in zip(
        LABEL_NAMES,
        prediction
    ):

        if value == 1:

            predicted_labels.append(
                label
            )

    return {
        "labels": predicted_labels,
        "prediction": prediction,
        "probabilities": probabilities
    }


# ==================================================
# Fungsi Probabilitas untuk LIME
# ==================================================

def predict_proba_lime(texts):
    """
    Digunakan oleh LIME untuk memperoleh
    probabilitas prediksi dari model.
    """

    processed_texts = [

        preprocess_text(text)

        for text in texts

    ]

    vectors = loaded_tfidf.transform(
        processed_texts
    )

    probabilities = loaded_model.predict_proba(
        vectors
    )

    return probabilities


# ==================================================
# Generate LIME Explanation
# ==================================================

def generate_lime(text):
    """
    Menghasilkan penjelasan lokal
    menggunakan LIME.
    """

    processed_text = preprocess_text(
        text
    )

    explainer = LimeTextExplainer(
        class_names=LABEL_NAMES
    )

    explanation = explainer.explain_instance(
        processed_text,
        predict_proba_lime,
        num_features=5,
        labels=[0, 1, 2]
    )

    return explanation