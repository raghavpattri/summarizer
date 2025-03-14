import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx
import re
from PyPDF2 import PdfReader
import docx
import time

# Ensure NLTK data is available
nltk.download('punkt')
nltk.download('stopwords')

# Tokenize and preprocess text
def tokenize_text(text):
    sentences = nltk.sent_tokenize(text)
    sentences = [nltk.word_tokenize(sentence) for sentence in sentences]
    return sentences

# Compute sentence similarity
def sentence_similarity(sent1, sent2, stop_words):
    sent1 = [w.lower() for w in sent1 if w.isalnum() and w.lower() not in stop_words]
    sent2 = [w.lower() for w in sent2 if w.isalnum() and w.lower() not in stop_words]
    
    all_words = list(set(sent1 + sent2))
    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)
    
    for w in sent1:
        vector1[all_words.index(w)] += 1
    for w in sent2:
        vector2[all_words.index(w)] += 1
    
    return 1 - cosine_distance(vector1, vector2)

# Generate similarity matrix
def generate_similarity_matrix(sentences, stop_words):
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 != idx2:
                similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2], stop_words)
    return similarity_matrix

# Summarize text
def summarize_text(text, top_n=5):
    stop_words = stopwords.words('english')
    sentences = tokenize_text(text)
    
    if len(sentences) == 0:
        return "No sentences detected in the input text."

    similarity_matrix = generate_similarity_matrix(sentences, stop_words)
    sentence_similarity_graph = nx.from_numpy_array(similarity_matrix)
    
    try:
        scores = nx.pagerank(sentence_similarity_graph)
    except nx.PowerIterationFailedConvergence:
        return "PageRank algorithm did not converge. Please try again with different input."

    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
    summary = [" ".join(ranked_sentences[i][1]) for i in range(min(top_n, len(ranked_sentences)))]
    return "\n".join(summary)

# Read file contents
def read_file(file, file_type):
    content = ""
    try:
        if file_type == "docx":
            doc = docx.Document(file)
            content = "\n".join([para.text for para in doc.paragraphs])
        elif file_type == "pdf":
            reader = PdfReader(file)
            content = "\n".join(page.extract_text() for page in reader.pages if page.extract_text())
        elif file_type == "txt":
            content = file.read().decode("utf-8")
    except Exception as e:
        content = f"Error reading file: {str(e)}"
    return content

# Streamlit app
def main():
    # App Title
    st.title("🚀 Text Summarizer")
    st.sidebar.title("⚙️ Settings")
    
    # File upload and text input
    uploaded_file = st.sidebar.file_uploader("📂 Upload a File (.pdf, .docx, .txt)", type=["pdf", "docx", "txt"])
    text_input = st.text_area("✍️ Or enter your text here:")

    # Number of sentences input
    num_sentences = st.sidebar.slider("🔢 Number of sentences in summary", min_value=1, max_value=99, value=10)
    
    # Process uploaded file
    if uploaded_file:
        file_type = uploaded_file.name.split('.')[-1].lower()
        text_input = read_file(uploaded_file, file_type)
        st.success(f"✅ Successfully loaded {uploaded_file.name}")
    
    # Summarize button
    if st.button("🚀 Summarize"):
        if not text_input.strip():
            st.error("❌ No text provided. Please upload a file or enter text.")
        else:
            with st.spinner("🚀 Summarizing... Hold tight!"):
                time.sleep(2)  # Simulating processing time
                summary = summarize_text(text_input, num_sentences)
            
            st.snow() # snow animation 🎈
            st.subheader("✨ Summary")
            st.text_area("📜 Summary Output", value=summary, height=200, key="summary_output")

    # About Section
    st.sidebar.subheader("ℹ️ About")
    st.sidebar.info(
        """
        This app summarizes text using the **PageRank Algorithm**. 
        Supported file types: PDF, DOCX, and TXT.
        """
    )

if __name__ == "__main__":
    main()
 