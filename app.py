import re
import streamlit as st

from prediction import (
    predict_softskills,
    generate_lime,
    LABEL_NAMES
)

from preprocessing import (
    preprocess_text
)


# ==================================================
# Konfigurasi Halaman
# ==================================================

st.set_page_config(
    page_title="Klasifikasi Multi-Label Soft Skills",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ==================================================
# Konstanta Aplikasi
# ==================================================

APP_TITLE = "📊 Klasifikasi Multi-Label Soft Skills"

APP_SUBTITLE = (
    "Analisis Jawaban Wawancara Kerja Berbahasa Indonesia"
)

APP_DESCRIPTION = """
Aplikasi ini membantu recruiter atau interviewer
mengidentifikasi aspek soft skills pada jawaban
wawancara kerja berbahasa Indonesia.

Analisis dilakukan menggunakan model Support Vector
Machine (SVM) dengan representasi fitur TF-IDF.

Untuk meningkatkan transparansi hasil prediksi,
aplikasi menerapkan Explainable AI menggunakan
Local Interpretable Model-Agnostic Explanations (LIME).
"""

APP_DISCLAIMER = """
Hasil analisis merupakan rekomendasi berbasis model
machine learning dan digunakan sebagai pendukung
keputusan dalam proses evaluasi kandidat.

Keputusan akhir tetap berada pada recruiter atau
interviewer dengan mempertimbangkan hasil wawancara
secara menyeluruh.
"""

SOFTSKILL_DESCRIPTIONS = {
    "Komunikasi":
        "Kemampuan kandidat dalam menyampaikan informasi secara jelas, "
        "mendengarkan secara aktif, serta berinteraksi secara efektif.",
    "Kerja Sama Tim":
        "Kemampuan kandidat dalam bekerja sama, berkoordinasi, "
        "membantu anggota tim, serta mencapai tujuan bersama.",
    "Kecerdasan Emosional":
        "Kemampuan kandidat dalam mengelola emosi, memahami sudut pandang "
        "orang lain, menghadapi tekanan, serta menjaga hubungan interpersonal."
}

INTERVIEW_EXAMPLES = [
    "Ceritakan pengalaman Anda bekerja dalam tim.",
    "Bagaimana Anda menghadapi konflik di tempat kerja?",
    "Bagaimana cara Anda berkomunikasi dengan rekan kerja?",
    "Bagaimana Anda menghadapi tekanan dalam pekerjaan?"
]

PROBABILITY_THRESHOLDS = {
    "Sangat Tinggi": 0.80,
    "Tinggi": 0.60,
    "Sedang": 0.40
}


# ==================================================
# Helper Functions
# ==================================================

def interpret_probability(probability):
    if probability >= PROBABILITY_THRESHOLDS["Sangat Tinggi"]:
        return "Sangat Tinggi"
    if probability >= PROBABILITY_THRESHOLDS["Tinggi"]:
        return "Tinggi"
    if probability >= PROBABILITY_THRESHOLDS["Sedang"]:
        return "Sedang"
    return "Rendah"


def _find_keyword_spans(original_text, positive_words):
    """
    Cari semua span (posisi karakter) kemunculan kata/frasa
    kunci di dalam teks asli (case-insensitive, word-boundary).
    """
    text_lower = original_text.lower()
    spans = []

    for word in positive_words:
        pattern = re.compile(r'\b' + re.escape(word.lower()) + r'\b')
        for match in pattern.finditer(text_lower):
            spans.append(match.span())

    spans.sort(key=lambda x: x[0])
    return spans


def _get_clause_spans(original_text):
    """
    Pecah teks menjadi klausa/frasa berdasarkan tanda baca
    (. , ; : ! ?). Setiap klausa dikembalikan sebagai span
    (start, end) relatif terhadap teks asli, sehingga highlight
    bisa mencakup satu unit frasa yang bermakna, bukan hanya
    satu kata lepas.
    """
    pattern = re.compile(r'[^.,;:!?]+[.,;:!?]?')
    clauses = []
    for m in pattern.finditer(original_text):
        if m.group().strip():
            clauses.append((m.start(), m.end()))
    return clauses


def highlight_text(original_text, positive_words):
    """
    Highlight kata/frasa positif pada teks asli dengan
    menyertakan konteks klausa di sekitarnya, sehingga hasil
    lebih mudah dibaca dan dipahami oleh recruiter (HR).

    Dua lapis highlight digunakan:
    - Kotak kuning muda: menandai klausa/frasa konteks yang
      memuat kata kunci.
    - Sorotan kuning tua + tebal: menandai kata/frasa kunci
      persis yang memengaruhi prediksi model.
    """
    if not positive_words:
        return original_text

    keyword_spans = _find_keyword_spans(original_text, positive_words)

    if not keyword_spans:
        return original_text

    clause_spans = _get_clause_spans(original_text)

    # Tandai klausa yang memuat minimal satu kata kunci
    context_spans = []
    for c_start, c_end in clause_spans:
        has_keyword = any(
            k_start < c_end and k_end > c_start
            for k_start, k_end in keyword_spans
        )
        if has_keyword:
            context_spans.append((c_start, c_end))

    # Gabungkan klausa konteks yang bersebelahan/tumpang tindih
    merged_context = []
    for start, end in context_spans:
        if merged_context and start <= merged_context[-1][1] + 1:
            merged_context[-1] = (
                merged_context[-1][0],
                max(merged_context[-1][1], end)
            )
        else:
            merged_context.append((start, end))

    def render_context_segment(seg_start, seg_end):
        seg_text = original_text[seg_start:seg_end]

        # Cari kata kunci relatif terhadap segmen ini
        local_spans = []
        for k_start, k_end in keyword_spans:
            rel_start = k_start - seg_start
            rel_end = k_end - seg_start
            if 0 <= rel_start < len(seg_text) and rel_end <= len(seg_text):
                local_spans.append((rel_start, rel_end))

        local_spans.sort()
        merged_local = []
        for s, e in local_spans:
            if merged_local and s <= merged_local[-1][1]:
                merged_local[-1] = (merged_local[-1][0], max(merged_local[-1][1], e))
            else:
                merged_local.append((s, e))

        pieces = []
        prev = 0
        for s, e in merged_local:
            pieces.append(seg_text[prev:s])
            pieces.append(
                f'<mark style="background-color: #FBC02D; '
                f'padding: 1px 3px; border-radius: 3px; '
                f'font-weight: 700; color: #212529;">'
                f'{seg_text[s:e]}</mark>'
            )
            prev = e
        pieces.append(seg_text[prev:])
        inner = "".join(pieces)

        return (
            f'<span style="background-color: #FFF9C4; '
            f'padding: 2px 4px; border-radius: 4px;">'
            f'{inner}</span>'
        )

    result = []
    prev = 0
    for start, end in merged_context:
        result.append(original_text[prev:start])
        result.append(render_context_segment(start, end))
        prev = end
    result.append(original_text[prev:])

    return "".join(result)


def get_label_conclusion(label, pred, probability):
    """Teks kesimpulan per label."""
    level = interpret_probability(probability)
    if pred == 1:
        return (
            f"Model menemukan indikator **{label}** pada jawaban kandidat. "
            f"Tingkat keyakinan model tergolong **{level}** "
            f"({probability:.0%}). "
            f"Prediksi dipengaruhi oleh beberapa kata atau frasa "
            f"yang menunjukkan aktivitas terkait aspek ini."
        )
    return (
        f"Model belum menemukan indikator yang cukup kuat "
        f"untuk aspek **{label}**. "
        f"Tingkat keyakinan model tergolong **{level}** "
        f"({probability:.0%})."
    )


def get_evaluation_note(label, pred, probability, n_keywords):
    """
    Menghasilkan catatan evaluasi pasca-wawancara untuk recruiter:
    seberapa kuat bukti yang ditemukan pada jawaban kandidat,
    beserta implikasinya terhadap proses penilaian.

    Returns
    -------
    tuple(str, str, str)
        (strength_label, note_text, style)
        style: "success" | "warning" | "info"
    """
    level = interpret_probability(probability)

    if pred == 1:
        if level in ("Sangat Tinggi", "Tinggi"):
            strength = "Bukti Kuat"
            style = "success"
            note = (
                f"Jawaban kandidat memuat **{n_keywords} indikator** yang "
                f"cukup jelas untuk aspek **{label}**, dengan tingkat "
                f"keyakinan model **{level.lower()}** ({probability:.0%}). "
                f"Temuan ini dapat dijadikan salah satu bahan penilaian "
                f"yang cukup kuat untuk aspek ini."
            )
        else:
            strength = "Bukti Cukup"
            style = "info"
            note = (
                f"Model menemukan indikator untuk aspek **{label}**, namun "
                f"dengan tingkat keyakinan **{level.lower()}** "
                f"({probability:.0%}). Sebaiknya bukti ini dikonfirmasi "
                f"kembali dengan catatan wawancara secara menyeluruh "
                f"sebelum dijadikan penilaian akhir."
            )
    else:
        strength = "Bukti Tidak Ditemukan"
        style = "warning"
        note = (
            f"Model tidak menemukan indikator yang cukup kuat untuk aspek "
            f"**{label}** pada jawaban yang dianalisis "
            f"(tingkat keyakinan **{level.lower()}**, {probability:.0%}). "
            f"Hal ini tidak selalu berarti kandidat tidak memiliki aspek "
            f"ini — bisa jadi jawaban yang diberikan memang tidak "
            f"menyinggung aspek tersebut secara eksplisit. Pertimbangkan "
            f"aspek ini bersama catatan wawancara lain yang relevan."
        )

    return strength, note, style


# ==================================================
# Header
# ==================================================

st.title(APP_TITLE)
st.subheader(APP_SUBTITLE)

st.write(APP_DESCRIPTION)

st.info(APP_DISCLAIMER)

st.divider()


# ==================================================
# Input Jawaban Wawancara
# ==================================================

st.header("📝 Masukkan Jawaban Wawancara")

st.write(
    "Masukkan satu jawaban wawancara kerja berbahasa Indonesia "
    "yang ingin dianalisis."
)

with st.expander("💡 Contoh Pertanyaan Wawancara"):
    for example in INTERVIEW_EXAMPLES:
        st.write(f"- {example}")

with st.form("prediction_form"):

    user_input = st.text_area(
        "Jawaban Wawancara",
        height=220,
        placeholder="Tuliskan jawaban wawancara kerja di sini..."
    )

    submit = st.form_submit_button(
        "🔍 Analisis Soft Skills",
        use_container_width=True
    )


# ==================================================
# Proses Prediksi
# ==================================================

if submit:

    if not user_input.strip():
        st.warning("Masukkan jawaban wawancara terlebih dahulu.")

    else:
        with st.spinner("Menganalisis jawaban..."):
            result = predict_softskills(user_input)

        st.session_state["result"] = result
        st.session_state["text"] = user_input
        # Reset LIME saat ada input baru
        if "explanation" in st.session_state:
            del st.session_state["explanation"]


# ==================================================
# Menampilkan Hasil Prediksi
# ==================================================

if "result" in st.session_state:

    result = st.session_state["result"]
    prediction = result["prediction"]
    probabilities = result["probabilities"]
    original_text = st.session_state["text"]
    processed_text = preprocess_text(original_text)

    st.success("Analisis berhasil dilakukan.")


    # ==================================================
    # Hasil Analisis
    # ==================================================

    st.header("📌 Hasil Analisis")

    st.write(
        "Sistem mengidentifikasi aspek soft skills "
        "yang terdeteksi berdasarkan jawaban wawancara yang dianalisis."
    )

    with st.expander("ℹ️ Penjelasan Aspek Soft Skills", expanded=False):
        for label in LABEL_NAMES:
            st.markdown(f"#### {label}")
            st.write(SOFTSKILL_DESCRIPTIONS[label])

    col1, col2, col3 = st.columns(3)
    columns = [col1, col2, col3]

    for column, label, pred in zip(columns, LABEL_NAMES, prediction):
        with column:
            if pred == 1:
                st.success(f"✅ {label}")
                st.caption("Terdeteksi")
            else:
                st.error(f"❌ {label}")
                st.caption("Tidak Terdeteksi")

    st.divider()


    # ==================================================
    # Tingkat Keyakinan Model
    # ==================================================

    st.header("📈 Tingkat Keyakinan Model")

    st.write(
        "Nilai berikut menunjukkan tingkat keyakinan model "
        "terhadap masing-masing aspek soft skills."
    )

    for label, pred, probability in zip(LABEL_NAMES, prediction, probabilities):

        st.subheader(label)
        st.progress(float(probability))

        metric_col1, metric_col2, metric_col3 = st.columns(3)

        with metric_col1:
            st.metric("Probabilitas", f"{probability:.2%}")

        with metric_col2:
            st.metric("Kategori", interpret_probability(probability))

        with metric_col3:
            if pred == 1:
                st.success("Terdeteksi")
            else:
                st.error("Tidak Terdeteksi")

        st.write("")

    st.divider()


    # ==================================================
    # Ringkasan Analisis
    # ==================================================

    st.header("📝 Ringkasan Analisis")

    detected_labels = [
        label
        for label, pred in zip(LABEL_NAMES, prediction)
        if pred == 1
    ]

    if detected_labels:
        st.success(
            f"Model mendeteksi indikator pada aspek berikut:\n\n"
            f"**{', '.join(detected_labels)}**"
        )
    else:
        st.info(
            "Model belum menemukan indikator yang cukup kuat "
            "pada ketiga aspek soft skills."
        )

    st.info(
        "Gunakan hasil analisis sebagai informasi pendukung "
        "dalam proses evaluasi kandidat. "
        "Keputusan akhir tetap berada pada recruiter atau interviewer "
        "dengan mempertimbangkan hasil wawancara secara menyeluruh."
    )

    st.divider()


    # ==================================================
    # Bagian 4: Penjelasan Hasil Prediksi (LIME)
    # ==================================================

    st.header("🧠 Penjelasan Hasil Prediksi")

    st.write(
        "Bagian ini menjelaskan alasan model memberikan prediksi "
        "menggunakan metode **LIME (Local Interpretable Model-Agnostic "
        "Explanations)**. LIME mengidentifikasi kata atau frasa pada "
        "jawaban kandidat yang paling memengaruhi keputusan model. "
        "Penjelasan ini bersifat lokal — hanya berlaku untuk jawaban "
        "yang sedang dianalisis."
    )

    # Generate LIME sekali, simpan di session_state
    if "explanation" not in st.session_state:
        with st.spinner("Memproses penjelasan model..."):
            explanation = generate_lime(original_text)
            st.session_state["explanation"] = explanation

    explanation = st.session_state["explanation"]

    # Pilih aspek soft skills
    selected_label = st.selectbox(
        "Pilih Aspek Soft Skills yang Ingin Dijelaskan",
        LABEL_NAMES,
        key="lime_label_selector"
    )

    label_index = LABEL_NAMES.index(selected_label)
    label_pred = prediction[label_index]
    label_prob = probabilities[label_index]

    lime_words = explanation.as_list(label=label_index)

    positive_words = [
        (word, weight)
        for word, weight in lime_words
        if weight > 0
    ]

    negative_words = [
        (word, weight)
        for word, weight in lime_words
        if weight <= 0
    ]

    positive_word_list = [word for word, _ in positive_words]

    st.divider()

    # --------------------------------------------------
    # Kesimpulan per Label
    # --------------------------------------------------

    st.subheader(f"📋 Kesimpulan — {selected_label}")

    conclusion = get_label_conclusion(
        selected_label,
        label_pred,
        label_prob
    )

    if label_pred == 1:
        st.success(conclusion)
    else:
        st.warning(conclusion)

    st.divider()

    # --------------------------------------------------
    # Bukti pada Jawaban (Highlight)
    # --------------------------------------------------

    st.subheader("🔍 Bukti pada Jawaban Kandidat")

    st.write(
        "Kalimat/frasa yang diberi kotak kuning muda menunjukkan "
        "konteks bagian jawaban yang relevan dengan aspek ini. "
        "Kata atau frasa dengan sorotan kuning tua dan tebal adalah "
        "kata kunci spesifik yang paling memengaruhi prediksi model."
    )

    if positive_word_list:

        highlighted_html = highlight_text(
            original_text,
            positive_word_list
        )

        st.markdown(
            f"""
            <div style="
                background-color: #F8F9FA;
                border-left: 4px solid #4CAF50;
                border-radius: 6px;
                padding: 16px 20px;
                font-size: 15px;
                line-height: 1.8;
                color: #212529;
            ">
                {highlighted_html}
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div style="display:flex; gap:20px; margin-top:8px;">
                <div style="display:flex; align-items:center; gap:6px;">
                    <span style="width:14px; height:14px; background-color:#FFF9C4;
                        border-radius:3px; display:inline-block;"></span>
                    <span style="font-size:13px; color:#6c757d;">Konteks kalimat/frasa</span>
                </div>
                <div style="display:flex; align-items:center; gap:6px;">
                    <span style="width:14px; height:14px; background-color:#FBC02D;
                        border-radius:3px; display:inline-block;"></span>
                    <span style="font-size:13px; color:#6c757d;">Kata kunci utama</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            f"""
            <div style="
                background-color: #F8F9FA;
                border-left: 4px solid #9E9E9E;
                border-radius: 6px;
                padding: 16px 20px;
                font-size: 15px;
                line-height: 1.8;
                color: #212529;
            ">
                {original_text}
            </div>
            """,
            unsafe_allow_html=True
        )

        st.caption(
            "Tidak ditemukan kata atau frasa yang mendukung "
            "prediksi model untuk aspek ini."
        )

    st.divider()

    # --------------------------------------------------
    # Faktor Pendukung dan Pengurang
    # --------------------------------------------------

    st.subheader("📊 Faktor yang Memengaruhi Prediksi")

    factor_col1, factor_col2 = st.columns(2)

    with factor_col1:

        st.markdown("**✅ Faktor Pendukung Prediksi**")

        st.caption(
            "Kata atau frasa yang meningkatkan keyakinan model "
            "terhadap aspek ini."
        )

        if positive_words:
            for word, weight in sorted(
                positive_words,
                key=lambda x: x[1],
                reverse=True
            ):
                st.write(f"• **{word}** &nbsp; `+{weight:.3f}`")
        else:
            st.write("—")

    with factor_col2:

        st.markdown("**❌ Faktor Pengurang Prediksi**")

        st.caption(
            "Kata atau frasa yang mengurangi keyakinan model "
            "terhadap aspek ini."
        )

        if negative_words:
            for word, weight in sorted(
                negative_words,
                key=lambda x: x[1]
            ):
                st.write(f"• **{word}** &nbsp; `{weight:.3f}`")
        else:
            st.write("—")

    st.divider()

    # --------------------------------------------------
    # Visualisasi LIME
    # --------------------------------------------------

    st.subheader("📉 Visualisasi Kontribusi Kata")

    st.write(
        "Grafik berikut menampilkan kontribusi masing-masing kata "
        "terhadap prediksi model. "
        "Batang ke kanan (hijau) menunjukkan kontribusi positif. "
        "Batang ke kiri (merah) menunjukkan kontribusi negatif."
    )

    fig = explanation.as_pyplot_figure(label=label_index)
    fig.set_size_inches(8, 4)
    st.pyplot(fig, clear_figure=True)

    st.divider()

    # --------------------------------------------------
    # Detail Kontribusi (Tabel)
    # --------------------------------------------------

    st.subheader("📋 Detail Kontribusi Kata")

    import pandas as pd

    if lime_words:

        table_data = []

        for word, weight in sorted(
            lime_words,
            key=lambda x: abs(x[1]),
            reverse=True
        ):
            table_data.append({
                "Kata atau Frasa": word,
                "Bobot": round(weight, 4),
                "Interpretasi": (
                    "Mendukung Prediksi"
                    if weight > 0
                    else "Mengurangi Prediksi"
                )
            })

        df_lime = pd.DataFrame(table_data)

        st.dataframe(
            df_lime,
            use_container_width=True,
            hide_index=True
        )

    else:
        st.write("Tidak ada data kontribusi kata.")

    st.divider()

    # --------------------------------------------------
    # Catatan Evaluasi untuk Recruiter
    # --------------------------------------------------

    st.subheader("💼 Catatan Evaluasi untuk Recruiter")

    st.caption(
        "Ringkasan kekuatan bukti pada jawaban wawancara yang telah "
        "dianalisis, sebagai bahan pertimbangan penilaian akhir."
    )

    strength_label, evaluation_note, note_style = get_evaluation_note(
        selected_label,
        label_pred,
        label_prob,
        n_keywords=len(positive_word_list)
    )

    badge_col, note_col = st.columns([1, 3])

    with badge_col:
        st.metric("Kekuatan Bukti", strength_label)

    with note_col:
        if note_style == "success":
            st.success(evaluation_note)
        elif note_style == "warning":
            st.warning(evaluation_note)
        else:
            st.info(evaluation_note)

    st.divider()

    # --------------------------------------------------
    # Catatan LIME
    # --------------------------------------------------

    st.subheader("📌 Catatan")

    st.info(
        """
        **Tentang LIME dan Penjelasan Ini**

        - Penjelasan ini dihasilkan menggunakan metode LIME
          (*Local Interpretable Model-Agnostic Explanations*).

        - LIME bersifat **lokal** — penjelasan hanya berlaku
          untuk jawaban yang sedang dianalisis dan dapat
          berbeda untuk jawaban lain.

        - Bobot kata menunjukkan seberapa besar pengaruh kata
          tersebut terhadap prediksi model, bukan tingkat
          kemampuan soft skills kandidat secara keseluruhan.

        - Hasil ini merupakan **alat bantu** dan tidak
          menggantikan penilaian recruiter atau interviewer.
        """
    )

    st.divider()


    # ==================================================
    # Bagian 5: Detail Teknis
    # ==================================================

    with st.expander("⚙️ Detail Teknis", expanded=False):

        st.markdown("#### Informasi Model")

        tech_col1, tech_col2, tech_col3 = st.columns(3)

        with tech_col1:
            st.info("**Model**\n\nSupport Vector Machine (SVM)")

        with tech_col2:
            st.info("**Representasi Fitur**\n\nTF-IDF (1-gram, 2-gram)")

        with tech_col3:
            st.info("**Explainable AI**\n\nLIME")

        st.markdown("#### Teks Setelah Preprocessing")

        st.caption(
            "Tahapan: Cleaning → Case Folding → Tokenisasi"
        )

        st.text_area(
            "Teks Terproses",
            value=processed_text,
            height=100,
            disabled=True,
            label_visibility="collapsed"
        )

        st.markdown("#### Threshold Prediksi")

        st.caption(
            "Label dinyatakan terdeteksi jika probabilitas ≥ 0.50"
        )

    st.divider()


    # ==================================================
    # Footer
    # ==================================================

    st.caption(
        "Klasifikasi Multi-Label Soft Skills pada Jawaban "
        "Wawancara Kerja Berbahasa Indonesia · "
        "Model: SVM · Fitur: TF-IDF · XAI: LIME"
    )