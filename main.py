import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import joblib
import unicodedata

# Step 1: Load IMDb Data
def load_data_from_directory(directory):
    data = []
    for label_type in ['pos', 'neg']:
        dir_name = os.path.join(directory, label_type)
        for fname in os.listdir(dir_name):
            if fname.endswith(".txt"):
                with open(os.path.join(dir_name, fname), encoding="utf-8") as f:
                    data.append({
                        'text': f.read(),
                        'label': 1 if label_type == 'pos' else 0
                    })
    return pd.DataFrame(data)

# Step 2: Data Preprocessing
def preprocess_data(data):
    # Clean and tokenize text
    data['cleaned_text'] = data['text'].str.lower().str.replace(r'[^a-z\s]', '', regex=True)
    return data

# Utility function to clean text
def clean_text(text):
    # Remove control characters and invalid Unicode
    return ''.join(ch for ch in text if unicodedata.category(ch)[0] != "C")

# Step 3: Dataset Class for BERT
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = clean_text(self.texts[idx])  # Clean text
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=True,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# Step 4: Training and Evaluation with Traditional ML

def train_ml_model(X_train, X_test, y_train, y_test):
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    model = LogisticRegression()
    model.fit(X_train_tfidf, y_train)
    predictions = model.predict(X_test_tfidf)

    print("Accuracy:", accuracy_score(y_test, predictions))
    print("Classification Report:\n", classification_report(y_test, predictions))

    # Save the model and vectorizer
    joblib.dump(model, 'traditional_ml_model.pkl')
    joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

# Step 5: Training and Evaluation with BERT

def train_bert_model(train_data, val_data, model, tokenizer, batch_size, epochs, device):
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    loss_fn = torch.nn.CrossEntropyLoss().to(device)

    model = model.to(device)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        print(f"Epoch {epoch + 1}/{epochs}")
        epoch_iterator = tqdm(train_loader, desc="Training", leave=False)

        for batch in epoch_iterator:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader):.4f}")

    model.eval()
    total_acc = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs.logits, dim=1)
            total_acc += (preds == labels).sum().item()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print("Validation Accuracy:", total_acc / len(val_data))
    print("Classification Report:\n", classification_report(all_labels, all_preds))

    # Save the model
    torch.save(model.state_dict(), 'bert_model.pth')

# Step 6: Load Saved Models and Vectorizer

def load_traditional_ml_model():
    model = joblib.load('traditional_ml_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    return model, vectorizer

def load_bert_model():
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    model.load_state_dict(torch.load('bert_model.pth'))
    return model

# Step 7: Single Prediction

def predict_sentiment(text, model, tokenizer, device):
    model.eval()
    with torch.no_grad():
        inputs = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=128,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        prediction = torch.argmax(probs, dim=-1).item()

        sentiment = "Positive" if prediction == 1 else "Negative"
        return sentiment

# Main Workflow
if __name__ == "__main__":
    # Define directories
    base_dir = "D:/D/right/CS505/project/data/aclImdb_v1/aclImdb"  # Update the path to your dataset
    train_dir = os.path.join(base_dir, "train")
    test_dir = os.path.join(base_dir, "test")

    # Check if models are already trained
    if os.path.exists('traditional_ml_model.pkl') and os.path.exists('bert_model.pth'):
        print("Loading saved models...")
        traditional_model, vectorizer = load_traditional_ml_model()
        bert_model = load_bert_model()
        bert_model = bert_model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    else:
        # Load dataset
        print("Loading data...")
        train_data = load_data_from_directory(train_dir)
        test_data = load_data_from_directory(test_dir)

        # Preprocess data
        train_data = preprocess_data(train_data)
        test_data = preprocess_data(test_data)

        X_train = train_data['cleaned_text']
        y_train = train_data['label']
        X_test = test_data['cleaned_text']
        y_test = test_data['label']

        # Traditional ML Training
        print("Training Traditional ML Model...")
        train_ml_model(X_train, X_test, y_train, y_test)

        # BERT Training
        print("Training BERT Model...")
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        train_dataset = SentimentDataset(X_train.tolist(), y_train.tolist(), tokenizer, max_len=128)
        val_dataset = SentimentDataset(X_test.tolist(), y_test.tolist(), tokenizer, max_len=128)

        bert_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
        train_bert_model(train_dataset, val_dataset, bert_model, tokenizer, batch_size=16, epochs=1, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        # Test single prediction
        sample_text = "Oppenheimer is a stunning and thought-provoking masterpiece. The film dives deep into the psyche of one of the most influential scientists of the 20th century, portraying his struggles, triumphs, and moral dilemmas with incredible depth and emotion. The cinematography and performances are nothing short of brilliant, making it a must-watch for anyone interested in history, science, or humanity."
        sentiment = predict_sentiment(sample_text, bert_model, tokenizer, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        print(f"Sample Text: {sample_text}\nPredicted Sentiment: {sentiment}")