🚀 Text Summarizer
A simple yet powerful Text Summarizer built using Python and NLTK. This project utilizes the PageRank algorithm to generate concise summaries from text data. It supports file uploads (PDF, DOCX, TXT) and allows users to specify the number of sentences for the summary.

🌟 Features
File Support: Accepts PDF, DOCX, and TXT files for text extraction.
Custom Summaries: Adjust the number of sentences in the output summary.
Interactive Interface: User-friendly design with live feedback and animations.
Advanced Algorithm: Uses cosine similarity and PageRank for sentence ranking


🛠️ Installation
1. Clone the repository:
    git clone https://github.com/your-username/text-summarizer.git
  cd text-summarizer

3. Set up a virtual environment (optional but recommended):
 # Create a virtual environment
python -m venv venv

# Activate the virtual environment
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

3. Install dependencies:
pip install -r requirements.txt


4. Download NLTK data:
Ensure necessary NLTK datasets are downloaded
import nltk
nltk.download('punkt')
nltk.download('stopwords')

5. Running as a Streamlit App (optional):
If you want to use the Streamlit interface:
streamlit run app.py

📚 How It Works
The text summarizer follows these steps:

Preprocessing: Tokenizes text into sentences and words.
Similarity Matrix: Computes sentence similarity using cosine distance.
Graph Representation: Builds a sentence similarity graph.
Ranking: Applies the PageRank algorithm to rank sentences by importance.
Summary Generation: Selects the top N sentences as the final summary.

✅ Supported File Types
PDF (.pdf)
Word Documents (.docx)
Plain Text (.txt)
💡 Contributing
Contributions are welcome! If you'd like to improve this project:

Fork the repo.
Create a new branch (git checkout -b feature-branch).
Commit your changes (git commit -m 'Add new feature').
Push to your branch (git push origin feature-branch).
Open a pull request!
📜 License
This project is licensed under the MIT License — feel free to use and modify it as needed.

🙌 Acknowledgments
NLTK for natural language processing tools.
NetworkX for graph-based sentence ranking.
Streamlit for the interactive UI.
