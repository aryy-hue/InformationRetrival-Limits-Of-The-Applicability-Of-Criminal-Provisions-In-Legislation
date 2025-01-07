import os
import fitz  # PyMuPDF
import re
import string
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Path folder dokumen PDF
folder_path = "dokumen"
stopwords_path = "stoplist-id.txt"

# Buat instance stemmer secara global untuk efisiensi
factory = StemmerFactory()
stemmer = factory.create_stemmer()


def load_stopwords(stopwords_path):
    try:
        with open(stopwords_path, "r", encoding="utf-8") as f:
            return set(f.read().split())
    except FileNotFoundError:
        print("File stopwords tidak ditemukan. Pastikan path sudah benar.")
        exit()


stopwords = load_stopwords(stopwords_path)


# Fungsi membaca isi PDF
def read_pdf(file_path):
    try:
        pdf_document = fitz.open(file_path)
        text = ""
        for page in pdf_document:
            text += page.get_text()
        pdf_document.close()
        return text
    except Exception as e:
        print(f"Error membaca file {file_path}: {e}")
        return ""


# Fungsi preprocessing teks
def preprocess_text(text):
    text = text.lower()  # Case folding
    text = re.sub(r"\d+", "", text)  # Hapus angka
    text = text.translate(str.maketrans("", "", string.punctuation))  # Hapus tanda baca
    words = text.split()
    words = [word for word in words if word not in stopwords]  # Stopwords removal
    words = [stemmer.stem(word) for word in words]  # Stemming
    return " ".join(words)


# Fungsi untuk memberikan cuplikan kalimat relevan dengan query
# Fungsi untuk memberikan cuplikan kalimat relevan dengan query
def get_document_snippet(doc_name, text, query, snippet_length=200):
    sentences = re.split(r"(?<=[.!?]) +", text)  # Pisahkan teks menjadi kalimat
    query_words = set(query.lower().split())  # Kata-kata dalam query

    # Hitung skor relevansi berdasarkan jumlah kata dalam query yang muncul di setiap kalimat
    sentence_scores = []
    for sentence in sentences:
        sentence_words = set(sentence.lower().split())
        match_count = len(query_words & sentence_words)  # Hitung jumlah kata cocok
        if match_count > 0:
            sentence_scores.append((sentence, match_count))

    # Urutkan kalimat berdasarkan skor kecocokan
    sentence_scores.sort(key=lambda x: x[1], reverse=True)

    # Pilih kalimat dengan skor kecocokan tertinggi untuk cuplikan
    snippet = " ".join(
        [sentence for sentence, _ in sentence_scores[:5]]
    )  # Ambil hingga 5 kalimat teratas

    # Potong cuplikan jika terlalu panjang
    if len(snippet) > snippet_length:
        snippet = snippet[:snippet_length] + " ..."

    # Highlight kata-kata dalam query
    for word in query.split():
        snippet = re.sub(
            rf"({re.escape(word)})",
            r'<span style="background-color: yellow;">\1</span>',
            snippet,
            flags=re.IGNORECASE,
        )

    return snippet


# Fungsi untuk membaca semua file PDF dalam folder
def read_all_pdfs(folder_path):
    try:
        pdf_files = [file for file in os.listdir(folder_path) if file.endswith(".pdf")]
        if not pdf_files:
            print("Tidak ada file PDF ditemukan di folder.")
            exit()

        pdf_texts = []
        for pdf_file in pdf_files:
            file_path = os.path.join(folder_path, pdf_file)
            text = read_pdf(file_path)
            if text.strip():
                preprocessed_text = preprocess_text(text)
                pdf_texts.append((pdf_file, preprocessed_text, text))
        return pdf_texts
    except Exception as e:
        print(f"Error membaca folder {folder_path}: {e}")
        exit()


# Update fungsi untuk ranking dokumen berdasarkan query
def rank_documents(query, pdf_texts):
    if not query.strip():
        return []

    doc_names = [name for name, _, _ in pdf_texts]
    doc_contents = [content for _, content, _ in pdf_texts]
    raw_doc_texts = [text for _, _, text in pdf_texts]

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(doc_contents)

    query_preprocessed = preprocess_text(query)
    query_vector = vectorizer.transform([query_preprocessed])

    cosine_sim = cosine_similarity(query_vector, tfidf_matrix)
    ranking = cosine_sim[0].argsort()[::-1]
    results = []

    for idx in ranking:
        doc_name = doc_names[idx]
        score = float(cosine_sim[0][idx])
        snippet = get_document_snippet(doc_name, raw_doc_texts[idx], query)
        results.append({"document": doc_name, "similarity": score, "snippet": snippet})

    return results


# Flask app
app = Flask(__name__)
CORS(app)

# Load and preprocess documents at startup
print("Membaca dan memproses dokumen PDF...")
pdf_texts = read_all_pdfs(folder_path)
print(f"{len(pdf_texts)} dokumen berhasil diproses.\n")


@app.route("/search", methods=["POST"])
def search():
    data = request.get_json()
    query = data.get("query", "")
    if not query:
        return jsonify({"error": "Query tidak boleh kosong."}), 400

    results = rank_documents(query, pdf_texts)
    threshold = 0.1  # Batas minimum similarity untuk ditampilkan
    filtered_results = [
        {
            "document": doc["document"],
            "similarity": doc["similarity"],
            "snippet": doc["snippet"],
        }
        for doc in results
        if doc["similarity"] > threshold
    ]
    return jsonify(filtered_results)


@app.route("/get_pdf/<doc_name>", methods=["GET"])
def get_pdf(doc_name):
    file_path = os.path.join(folder_path, doc_name)
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    else:
        return jsonify({"error": "File not found"}), 404


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
