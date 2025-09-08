import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import PyPDF2
import io
import pickle
import matplotlib.pyplot as plt
from typing import List, Dict, Any
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(page_title="ML Architecture Playground", layout="wide")

class TextDataset(Dataset):
    def __init__(self, texts, targets, vectorizer=None, max_len=512):
        self.texts = texts
        self.targets = targets
        self.max_len = max_len
        
        if vectorizer is None:
            self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
            self.features = self.vectorizer.fit_transform(texts).toarray()
        else:
            self.vectorizer = vectorizer
            self.features = vectorizer.transform(texts).toarray()
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.features[idx]), torch.LongTensor([self.targets[idx]])

# Mixture of Experts Model
class Expert(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Expert, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.network(x)

class MixtureOfExperts(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_experts=4):
        super(MixtureOfExperts, self).__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([
            Expert(input_dim, hidden_dim, output_dim) for _ in range(num_experts)
        ])
        self.gating = nn.Sequential(
            nn.Linear(input_dim, num_experts),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        # Get gating weights
        gate_weights = self.gating(x)  # [batch_size, num_experts]
        
        # Get expert outputs
        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(x))
        
        # Stack expert outputs
        expert_outputs = torch.stack(expert_outputs, dim=2)  # [batch_size, output_dim, num_experts]
        
        # Apply gating weights
        gate_weights = gate_weights.unsqueeze(1)  # [batch_size, 1, num_experts]
        output = torch.sum(expert_outputs * gate_weights, dim=2)  # [batch_size, output_dim]
        
        return output

# Simple Transformer Model
class SimpleTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads=4):
        super(SimpleTransformer, self).__init__()
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(self.transformer_layer, num_layers=2)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.input_projection(x)
        x = x.unsqueeze(1)  # Add sequence dimension
        x = self.transformer(x)
        x = x.squeeze(1)  # Remove sequence dimension
        return self.output_layer(x)

# CNN Model
class CNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(CNNModel, self).__init__()
        self.conv1d = nn.Conv1d(1, 64, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(hidden_dim // 64)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, output_dim)
        )
    
    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension
        x = torch.relu(self.conv1d(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, num_layers=2, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = x.unsqueeze(1)  # Add sequence dimension
        lstm_out, (hidden, _) = self.lstm(x)
        return self.fc(hidden[-1])  # Use last hidden state

# Simple MLP
class MLPModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLPModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, output_dim)
        )
    
    def forward(self, x):
        return self.network(x)

def extract_text_from_pdf(pdf_file):
    """Extract text from uploaded PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_file.read()))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return ""

def preprocess_text(text):
    """Basic text preprocessing"""
    # Remove extra whitespace and special characters
    text = re.sub(r'\s+', ' ', text)
    
    # --- FIX IS HERE ---
    # We'll keep periods by adding `.` to the regex
    text = re.sub(r'[^\w\s.]', '', text) 
    # --- END OF FIX ---
    
    return text.lower().strip()

def create_qa_pairs(text, chunk_size=200):
    """Create simple Q&A pairs from text chunks"""
    sentences = text.split('.')
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) < chunk_size:
            current_chunk += sentence + ". "
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    qa_pairs = []
    for i, chunk in enumerate(chunks):
        if len(chunk) > 20:  # Only use meaningful chunks
            qa_pairs.append({
                'question': f"What is mentioned in section {i+1}?",
                'answer': chunk,
                'label': i % 5  # Simple classification into 5 categories
            })
    
    return qa_pairs

def train_model(model, train_loader, val_loader, epochs=10):
    """Train the model"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    train_losses = []
    val_losses = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            target = target.squeeze()
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in val_loader:
                output = model(data)
                target = target.squeeze()
                val_loss += criterion(output, target).item()
                pred = output.argmax(dim=1)
                total += target.size(0)
                correct += pred.eq(target).sum().item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        accuracy = 100. * correct / total
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        progress_bar.progress((epoch + 1) / epochs)
        status_text.text(f'Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Accuracy: {accuracy:.2f}%')
    
    return train_losses, val_losses

def plot_training_curves(train_losses, val_losses):
    """Plot training curves"""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(train_losses, label='Training Loss')
    ax.plot(val_losses, label='Validation Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training and Validation Loss')
    ax.legend()
    ax.grid(True)
    return fig

def main():
    st.title("ðŸš€ ML Architecture Playground")
    st.markdown("Upload PDFs, train different architectures, and test with questions!")
    
    # Sidebar for model selection
    st.sidebar.header("Model Configuration")
    
    architecture = st.sidebar.selectbox(
        "Choose Architecture",
        ["Mixture of Experts (MoE)", "Simple Transformer", "CNN", "LSTM", "MLP"]
    )
    
    hidden_dim = st.sidebar.slider("Hidden Dimension", 32, 512, 128)
    epochs = st.sidebar.slider("Training Epochs", 5, 50, 10)
    batch_size = st.sidebar.slider("Batch Size", 8, 64, 16)
    
    # Initialize session state
    if 'trained_models' not in st.session_state:
        st.session_state.trained_models = {}
    if 'vectorizers' not in st.session_state:
        st.session_state.vectorizers = {}
    if 'qa_data' not in st.session_state:
        st.session_state.qa_data = []
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["ðŸ“ Data Upload", "ðŸ‹ï¸ Model Training", "â“ Q&A Testing"])
    
    with tab1:
        st.header("Upload PDF Data")
        
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type="pdf",
            accept_multiple_files=True
        )
        
        if uploaded_files:
            st.info(f"Uploaded {len(uploaded_files)} PDF file(s)")
            
            if st.button("Process PDFs"):
                with st.spinner("Processing PDFs..."):
                    all_text = ""
                    for uploaded_file in uploaded_files:
                        text = extract_text_from_pdf(uploaded_file)
                        all_text += text + "\n\n"
                    
                    if all_text:
                        processed_text = preprocess_text(all_text)
                        
                        # Check if we have enough text
                        if len(processed_text) < 100:
                            st.warning("PDF content seems too short. Please upload a larger PDF or multiple PDFs.")
                            return
                        
                        qa_pairs = create_qa_pairs(processed_text)
                        st.session_state.qa_data = qa_pairs
                        
                        st.success(f"Processed {len(qa_pairs)} text chunks for training")
                        st.info(f"Text length: {len(processed_text)} characters")
                        
                        # Show sample data
                        if qa_pairs:
                            st.subheader("Sample Training Data")
                            sample_df = pd.DataFrame(qa_pairs[:5])
                            st.dataframe(sample_df)
    
    with tab2:
        st.header("Model Training")
        
        if not st.session_state.qa_data:
            st.warning("Please upload and process PDF data first!")
        else:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.info(f"Ready to train {architecture} with {len(st.session_state.qa_data)} samples")
                
                if st.button("Start Training", type="primary"):
                    with st.spinner("Training model..."):
                        # Prepare data
                        texts = [qa['answer'] for qa in st.session_state.qa_data]
                        labels = [qa['label'] for qa in st.session_state.qa_data]
                        
                        # --- FIX STARTS HERE ---
                        # Add a check to ensure there's enough data for a split.
                        # We need at least a few samples to create a training and validation set.
                        if len(texts) < 5: # Setting a minimum of 5 samples
                            st.error(f"Not enough data to train. You only have {len(texts)} samples, but at least 5 are required. Please upload a larger or more text-rich PDF.")
                            st.stop() # Stop execution if data is insufficient
                        # --- FIX ENDS HERE ---
                        
                        # Create dataset
                        train_texts, val_texts, train_labels, val_labels = train_test_split(
                            texts, labels, test_size=0.2, random_state=42
                        )
                        
                        train_dataset = TextDataset(train_texts, train_labels)
                        val_dataset = TextDataset(val_texts, val_labels, train_dataset.vectorizer)
                        
                        train_dataset = TextDataset(train_texts, train_labels)
                        val_dataset = TextDataset(val_texts, val_labels, train_dataset.vectorizer)
                        
                        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                        val_loader = DataLoader(val_dataset, batch_size=batch_size)
                        
                        input_dim = train_dataset.features.shape[1]
                        output_dim = len(set(labels))
                        
                        # Create model based on selection
                        if architecture == "Mixture of Experts (MoE)":
                            model = MixtureOfExperts(input_dim, hidden_dim, output_dim)
                        elif architecture == "Simple Transformer":
                            model = SimpleTransformer(input_dim, hidden_dim, output_dim)
                        elif architecture == "CNN":
                            model = CNNModel(input_dim, hidden_dim, output_dim)
                        elif architecture == "LSTM":
                            model = LSTMModel(input_dim, hidden_dim, output_dim)
                        else:  # MLP
                            model = MLPModel(input_dim, hidden_dim, output_dim)
                        
                        # Train model
                        train_losses, val_losses = train_model(model, train_loader, val_loader, epochs)
                        
                        # Store trained model
                        st.session_state.trained_models[architecture] = model
                        st.session_state.vectorizers[architecture] = train_dataset.vectorizer
                        
                        st.success("Training completed!")
                        
                        # Plot results
                        fig = plot_training_curves(train_losses, val_losses)
                        st.pyplot(fig)
            
            with col2:
                st.subheader("Training Info")
                st.write(f"**Architecture:** {architecture}")
                st.write(f"**Hidden Dim:** {hidden_dim}")
                st.write(f"**Epochs:** {epochs}")
                st.write(f"**Batch Size:** {batch_size}")
                
                if st.session_state.trained_models:
                    st.subheader("Trained Models")
                    for model_name in st.session_state.trained_models.keys():
                        st.write(f"âœ… {model_name}")
    
    with tab3:
        st.header("Q&A Testing")
        
        if not st.session_state.trained_models:
            st.warning("Please train a model first!")
        else:
            # Model selection for testing
            test_model = st.selectbox(
                "Select model for testing",
                list(st.session_state.trained_models.keys())
            )
            
            question = st.text_input("Enter your question:")
            
            if question and st.button("Get Answer"):
                model = st.session_state.trained_models[test_model]
                vectorizer = st.session_state.vectorizers[test_model]
                
                # Process question
                question_features = vectorizer.transform([question]).toarray()
                question_tensor = torch.FloatTensor(question_features)
                
                # Get prediction
                model.eval()
                with torch.no_grad():
                    output = model(question_tensor)
                    predicted_class = output.argmax(dim=1).item()
                    confidence = torch.softmax(output, dim=1).max().item()
                
                # Find similar content
                relevant_qa = [qa for qa in st.session_state.qa_data if qa['label'] == predicted_class]
                
                if relevant_qa:
                    st.success(f"**Answer** (Confidence: {confidence:.2f}):")
                    st.write(relevant_qa[0]['answer'])
                else:
                    st.info("No relevant information found in the training data.")
            
            # Show some example questions
            if st.session_state.qa_data:
                st.subheader("Example Questions")
                example_questions = [qa['question'] for qa in st.session_state.qa_data[:5]]
                for i, eq in enumerate(example_questions):
                    if st.button(f"Try: {eq}", key=f"example_{i}"):
                        st.text_input("Enter your question:", value=eq, key=f"auto_question_{i}")

if __name__ == "__main__":
    main()
