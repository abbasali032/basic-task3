# Import necessary libraries
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Preprocessing function
def preprocess(text):
    # Tokenization
    tokens = word_tokenize(text.lower())
    # Removing stopwords
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# Example documents
document1 = "Python is a programming language."
document2 = "Python is an interpreted, high-level programming language for general-purpose programming."

# Preprocess documents
processed_doc1 = preprocess(document1)
processed_doc2 = preprocess(document2)

# Vectorization
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform([processed_doc1, processed_doc2])

# Compute cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])

# Similarity threshold
threshold = 0.8

# Check for plagiarism
if cosine_sim[0][0] > threshold:
    print("Plagiarism detected!")
else:
    print("No plagiarism detected.")

