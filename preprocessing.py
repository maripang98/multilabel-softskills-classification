import re


def preprocess_text(text):
    """
    Preprocessing teks untuk proses inferensi.

    Tahapan:
    1. Cleaning
    2. Case Folding
    3. Tokenisasi
    """

    # Validasi input
    if text is None:
        return ""

    text = str(text)

    # ==================================================
    # Cleaning
    # ==================================================

    # Hapus URL
    text = re.sub(
        r"http\S+|www\S+",
        " ",
        text
    )

    # Hapus Email
    text = re.sub(
        r"\S+@\S+",
        " ",
        text
    )

    # Hapus karakter selain huruf, angka, dan spasi
    text = re.sub(
        r"[^a-zA-Z0-9\s]",
        " ",
        text
    )

    # Hapus spasi berlebih
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    # ==================================================
    # Case Folding
    # ==================================================

    text = text.lower()

    # ==================================================
    # Tokenisasi
    # ==================================================

    tokens = text.split()

    # Kembalikan menjadi string
    return " ".join(tokens)