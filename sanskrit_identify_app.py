import streamlit as st
import nltk
from nltk.tag import tnt
from nltk.tree import Tree
# from nltk.tokenize import PunktWordTokenizer
#from nltk.tokenize import PunktWordTokenizer
from sklearn.model_selection import train_test_split
import tempfile

# Ensure NLTK data is available
nltk.download('punkt')

# Load POS-tagged file in custom format
def load_custom_pos_file(path):
    tagged_sentences = []
    with open(path, 'r', encoding='utf-8') as f:
        current_sentence = []
        for line in f:
            line = line.strip()
            if line.startswith("<Sentence"):
                current_sentence = []
            elif line.startswith("</Sentence>"):
                if current_sentence:
                    tagged_sentences.append(current_sentence)
            elif line and "_" in line:
                tokens = line.split()
                for token in tokens:
                    if "_" in token:
                        word, pos = token.rsplit("_", 1)
                        current_sentence.append((word, pos))
    return tagged_sentences

# Train the TnT tagger
def train_tnt_model(train_data):
    tnt_pos_tagger = tnt.TnT()
    tnt_pos_tagger.train(train_data)
    return tnt_pos_tagger

# Calculate tagging accuracy
def calculate_accuracy(tagger, test_data):
    total = 0
    correct = 0
    for sentence in test_data:
        words = [word for word, _ in sentence]
        gold_tags = [tag for _, tag in sentence]
        predicted_tags = tagger.tag(words)
        for (_, gold), (_, pred) in zip(sentence, predicted_tags):
            total += 1
            if gold == pred:
                correct += 1
    return correct / total if total > 0 else 0.0

# Extract noun phrase keywords
def extract_keywords(tagged_pos):
    grammar = r"""NP: {<NN.*>}"""
    chunk_parser = nltk.RegexpParser(grammar)
    chunked = chunk_parser.parse(tagged_pos)
    keywords = set()
    for node in chunked:
        if isinstance(node, Tree):
            chunk = " ".join([token for token, _ in node.leaves()])
            keywords.add(chunk)
    return keywords

# === Streamlit App ===
st.title("🕉 Sanskrit POS Tagger & Keyword Extractor")
st.write("Upload a POS-tagged `.pos` file and extract tags or noun phrases from Sanskrit input.")

uploaded_file = st.file_uploader("Upload a POS-tagged Sanskrit file", type=["pos"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pos") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    tagged_sents = load_custom_pos_file(tmp_path)

    if not tagged_sents:
        st.error("The file seems empty or incorrectly formatted.")
    else:
        train_data, test_data = train_test_split(tagged_sents, test_size=0.2, random_state=42)
        model = train_tnt_model(train_data)

        accuracy = calculate_accuracy(model, test_data)
        st.success(f"Model trained successfully!\n**Accuracy:** {accuracy:.2%}")

        user_input = st.text_area("✍️ Enter Sanskrit text to tag:")

        if st.button("🔍 Tag Text"):
            if user_input.strip():
                #tokenizer = PunktWordTokenizer()
                tokens = user_input.strip().split()
                tagged = model.tag(tokens)

                st.markdown("### 🏷 Tagged Output")
                st.write(tagged)

                keywords = extract_keywords(tagged)
                st.markdown("### 🔑 Extracted Keywords (Noun Phrases)")
                st.write(list(keywords) if keywords else "No noun phrases found.")
            else:
                st.warning("Please enter some text to tag.")
else:
    st.info("Upload a `.pos` file in the format: `word_POS` inside `<Sentence>` tags.")
