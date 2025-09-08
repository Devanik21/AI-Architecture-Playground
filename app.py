import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import PyPDF2
import io
import pickle
import matplotlib.pyplot as plt
from typing import List, Dict, Any
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(page_title="ML Architecture Playground", layout="wide")

class RobustTextDataset(Dataset):
    def __init__(self, texts, targets, vectorizer=None, max_features=2000):
        self.texts = texts
        self.targets = targets
        
        if vectorizer is None:
            # Use more sophisticated vectorization
            self.vectorizer = TfidfVectorizer(
                max_features=max_features,
                stop_words='english',
                ngram_range=(1, 3),  # Include bigrams and trigrams
                min_df=2,  # Ignore terms that appear in less than 2 documents
                max_df=0.8,  # Ignore terms that appear in more than 80% of documents
                sublinear_tf=True,  # Apply sublinear tf scaling
                norm='l2'  # L2 normalization
            )
            self.features = self.vectorizer.fit_transform(texts).toarray()
        else:
            self.vectorizer = vectorizer
            self.features = vectorizer.transform(texts).toarray()
        
        # Normalize features for better training stability
        if not hasattr(vectorizer, 'scaler') or vectorizer is None:
            self.scaler = StandardScaler()
            self.features = self.scaler.fit_transform(self.features)
            if vectorizer is not None:
                self.vectorizer.scaler = self.scaler
        else:
            self.features = vectorizer.scaler.transform(self.features)
    
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

class RobustMixtureOfExperts(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_experts=8):
        super(RobustMixtureOfExperts, self).__init__()
        self.num_experts = num_experts
        
        # More sophisticated experts with different architectures
        self.experts = nn.ModuleList()
        for i in range(num_experts):
            if i % 4 == 0:  # Dense expert
                expert = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim * 2),
                    nn.LayerNorm(hidden_dim * 2),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(hidden_dim, output_dim)
                )
            elif i % 4 == 1:  # Sparse expert
                expert = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.4),
                    nn.Linear(hidden_dim, output_dim)
                )
            elif i % 4 == 2:  # Wide expert
                expert = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim * 3),
                    nn.LayerNorm(hidden_dim * 3),
                    nn.GELU(),
                    nn.Dropout(0.25),
                    nn.Linear(hidden_dim * 3, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, output_dim)
                )
            else:  # Deep expert
                expert = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Linear(hidden_dim // 2, output_dim)
                )
            self.experts.append(expert)
        
        # Improved gating network with temperature scaling
        self.gating = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_experts)
        )
        
        # Temperature parameter for confidence calibration
        self.temperature = nn.Parameter(torch.ones(1))
        
        # Expert utilization regularization
        self.expert_usage = torch.zeros(num_experts)
    
    def forward(self, x, training=True):
        batch_size = x.size(0)
        
        # Get gating logits
        gate_logits = self.gating(x)  # [batch_size, num_experts]
        
        # Apply temperature scaling for better calibration
        gate_logits = gate_logits / self.temperature
        
        # Add noise during training for better expert utilization
        if training and self.training:
            noise = torch.randn_like(gate_logits) * 0.1
            gate_logits = gate_logits + noise
        
        # Compute gating weights
        gate_weights = torch.softmax(gate_logits, dim=1)
        
        # Track expert usage for regularization
        if self.training:
            self.expert_usage += gate_weights.sum(dim=0).detach().cpu()
        
        # Get expert outputs
        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(x))
        
        # Stack expert outputs: [batch_size, output_dim, num_experts]
        expert_outputs = torch.stack(expert_outputs, dim=2)
        
        # Apply gating weights: [batch_size, 1, num_experts]
        gate_weights = gate_weights.unsqueeze(1)
        
        # Weighted combination
        output = torch.sum(expert_outputs * gate_weights, dim=2)
        
        # Return both output and gating weights for analysis
        return output, gate_weights.squeeze(1)

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

def robust_preprocess_text(text):
    """Enhanced preprocessing for better feature extraction"""
    # Handle encoding issues
    text = text.encode('ascii', 'ignore').decode('ascii')
    
    # Normalize whitespace but preserve structure
    text = re.sub(r'\n+', '. ', text)  # Convert line breaks to sentence separators
    text = re.sub(r'\s+', ' ', text)
    
    # Keep important punctuation for context
    text = re.sub(r'[^\w\s.,!?;:-]', '', text)
    
    # Handle common abbreviations
    abbreviations = {
        'dr.': 'doctor', 'prof.': 'professor', 'vs.': 'versus',
        'etc.': 'etcetera', 'i.e.': 'that is', 'e.g.': 'for example'
    }
    
    text_lower = text.lower()
    for abbr, full in abbreviations.items():
        text_lower = text_lower.replace(abbr, full)
    
    return text_lower.strip()

def create_robust_qa_pairs(text, chunk_size=300):
    """Create more meaningful Q&A pairs with better categorization"""
    # Better sentence splitting
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) < chunk_size:
            current_chunk += sentence + ". "
        else:
            if len(current_chunk) > 50:  # Only meaningful chunks
                chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "
    
    if len(current_chunk) > 50:
        chunks.append(current_chunk.strip())
    
    # Improved categorization using keyword-based classification
    qa_pairs = []
    keywords_categories = {
        0: ['introduction', 'overview', 'background', 'summary', 'abstract'],
        1: ['method', 'approach', 'technique', 'algorithm', 'procedure'],
        2: ['result', 'finding', 'outcome', 'performance', 'evaluation'],
        3: ['discussion', 'analysis', 'interpretation', 'implication'],
        4: ['conclusion', 'future', 'limitation', 'recommendation']
    }
    
    for i, chunk in enumerate(chunks):
        if len(chunk) > 30:
            # Determine category based on keywords
            chunk_lower = chunk.lower()
            category = i % 5  # default
            
            for cat, keywords in keywords_categories.items():
                if any(keyword in chunk_lower for keyword in keywords):
                    category = cat
                    break
            
            qa_pairs.append({
                'question': f"What information is provided about {keywords_categories[category][0]}?",
                'answer': chunk,
                'label': category
            })
    
    return qa_pairs


def train_robust_model(model, train_loader, val_loader, epochs=30):
    """Enhanced training with proper loss functions and regularization"""
    
    # ADD THIS BLOCK BACK IF IT'S MISSING
    # vv================================================================vv
    # Label smoothing cross entropy for better calibration
    class LabelSmoothingCrossEntropy(nn.Module):
        def __init__(self, smoothing=0.1):
            super().__init__()
            self.smoothing = smoothing
        
        def forward(self, pred, target):
            n_class = pred.size(1)
            one_hot = torch.zeros_like(pred).scatter(1, target.unsqueeze(1), 1)
            one_hot = one_hot * (1 - self.smoothing) + self.smoothing / n_class
            log_prob = F.log_softmax(pred, dim=1)
            return -(one_hot * log_prob).sum(dim=1).mean()
    
    criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
    # ^^================================================================^^

    # Different learning rates for different parts
    # Universal optimizer setup
    if isinstance(model, RobustMixtureOfExperts):
        # Specific optimizer for MoE with different learning rates
        gate_params = list(model.gating.parameters()) + [model.temperature]
        gate_param_ids = {id(p) for p in gate_params}
        expert_params = [p for p in model.parameters() if id(p) not in gate_param_ids]
        
        optimizer = optim.AdamW([
            {'params': expert_params, 'lr': 0.001, 'weight_decay': 0.01},
            {'params': gate_params, 'lr': 0.002, 'weight_decay': 0.005}
        ])
    else:
        # Standard optimizer for all other models
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    
    # ... (the rest of the function continues from here)
    
    # Cosine annealing scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    best_val_acc = 0
    patience = 10
    no_improve = 0
    
    # Streamlit progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    metrics_placeholder = st.empty()
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            
            if isinstance(model, RobustMixtureOfExperts):
                output, gate_weights = model(data, training=True)
            else:
                output = model(data)
            
            target = target.squeeze()
            
            # Main classification loss
            main_loss = criterion(output, target)
            
            # Expert utilization regularization for MoE
            if isinstance(model, RobustMixtureOfExperts):
                gate_entropy = -torch.sum(gate_weights * torch.log(gate_weights + 1e-8), dim=1).mean()
                expert_balance_loss = -gate_entropy * 0.01
                
                expert_usage = gate_weights.mean(dim=0)
                balance_loss = torch.var(expert_usage) * 0.01
                
                total_loss = main_loss + expert_balance_loss + balance_loss
            else:
                total_loss = main_loss
            
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += total_loss.item()
        
        scheduler.step()
        
        # Validation phase
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        all_probs = []
        
        with torch.no_grad():
            for data, target in val_loader:
                if isinstance(model, RobustMixtureOfExperts):
                    output, _ = model(data, training=False)
                else:
                    output = model(data)
                
                target = target.squeeze()
                
                val_loss += criterion(output, target).item()
                
                # Get probabilities for confidence analysis
                probs = torch.softmax(output, dim=1)
                all_probs.extend(probs.cpu().numpy())
                
                pred = output.argmax(dim=1)
                total += target.size(0)
                correct += pred.eq(target).sum().item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        accuracy = 100. * correct / total
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(accuracy)
        
        # Calculate average confidence
        avg_confidence = np.mean([np.max(prob) for prob in all_probs])
        
        # Update Streamlit interface
        progress_bar.progress((epoch + 1) / epochs)
        
        status_text.text(f'Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Accuracy: {accuracy:.2f}%')
        
        # Display detailed metrics
        with metrics_placeholder.container():
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Training Loss", f"{train_loss:.4f}")
            with col2:
                st.metric("Validation Accuracy", f"{accuracy:.2f}%")
            with col3:
                st.metric("Avg Confidence", f"{avg_confidence:.4f}")
        
        # Early stopping
        if accuracy > best_val_acc:
            best_val_acc = accuracy
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                st.info(f'Early stopping at epoch {epoch+1}. Best accuracy: {best_val_acc:.2f}%')
                break
    
    return train_losses, val_losses, val_accuracies

def plot_training_curves(train_losses, val_losses, val_accuracies=None):
    """Plot training curves"""
    if val_accuracies is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Loss curves
        ax1.plot(train_losses, label='Training Loss', color='blue')
        ax1.plot(val_losses, label='Validation Loss', color='red')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy curve
        ax2.plot(val_accuracies, label='Validation Accuracy', color='green')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Validation Accuracy')
        ax2.legend()
        ax2.grid(True)
        
    else:
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
    st.title("🚀 ML Architecture Playground - Robust Version")
    st.markdown("Upload PDFs, train different architectures, and test with questions!")
    
    # Sidebar for model selection
    st.sidebar.header("Model Configuration")
    
    architecture = st.sidebar.selectbox(
        "Choose Architecture",
        ["Mixture of Experts (MoE)", "Simple Transformer", "CNN", "LSTM", "MLP"]
    )
    
    hidden_dim = st.sidebar.slider("Hidden Dimension", 32, 512, 128)
    epochs = st.sidebar.slider("Training Epochs", 5, 50, 30)
    batch_size = st.sidebar.slider("Batch Size", 8, 64, 16)
    
    if architecture == "Mixture of Experts (MoE)":
        num_experts = st.sidebar.slider("Number of Experts", 4, 12, 8)
    
    # Initialize session state
    if 'trained_models' not in st.session_state:
        st.session_state.trained_models = {}
    if 'vectorizers' not in st.session_state:
        st.session_state.vectorizers = {}
    if 'qa_data' not in st.session_state:
        st.session_state.qa_data = []
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["📁 Data Upload", "🏋️ Model Training", "❓ Q&A Testing"])
    
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
                        processed_text = robust_preprocess_text(all_text)
                        
                        # Check if we have enough text
                        if len(processed_text) < 100:
                            st.warning("PDF content seems too short. Please upload a larger PDF or multiple PDFs.")
                            return
                        
                        qa_pairs = create_robust_qa_pairs(processed_text)
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
                        
                        # Check for sufficient data
                        if len(texts) < 5:
                            st.error(f"Not enough data to train. You only have {len(texts)} samples, but at least 5 are required.")
                            st.stop()
                        
                        # Create dataset
                        train_texts, val_texts, train_labels, val_labels = train_test_split(
                            texts, labels, test_size=0.2, random_state=42
                        )
                        
                        train_dataset = RobustTextDataset(train_texts, train_labels)
                        val_dataset = RobustTextDataset(val_texts, val_labels, train_dataset.vectorizer)
                        
                        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                        val_loader = DataLoader(val_dataset, batch_size=batch_size)
                        
                        input_dim = train_dataset.features.shape[1]
                        output_dim = len(set(labels))
                        
                        # Create model based on selection
                        if architecture == "Mixture of Experts (MoE)":
                            model = RobustMixtureOfExperts(input_dim, hidden_dim, output_dim, num_experts)
                        elif architecture == "Simple Transformer":
                            model = SimpleTransformer(input_dim, hidden_dim, output_dim)
                        elif architecture == "CNN":
                            model = CNNModel(input_dim, hidden_dim, output_dim)
                        elif architecture == "LSTM":
                            model = LSTMModel(input_dim, hidden_dim, output_dim)
                        else:  # MLP
                            model = MLPModel(input_dim, hidden_dim, output_dim)
                        
                        # Train model
                        # Train model
                        train_losses, val_losses, val_accuracies = train_robust_model(model, train_loader, val_loader, epochs)
                        
                        # Store trained model
                        st.session_state.trained_models[architecture] = model
                        st.session_state.vectorizers[architecture] = train_dataset.vectorizer
                        
                        st.success("Training completed!")
                        
                        # Plot results
                        fig = plot_training_curves(train_losses, val_losses, val_accuracies)
                        st.pyplot(fig)
            
            with col2:
                st.subheader("Training Info")
                st.write(f"**Architecture:** {architecture}")
                st.write(f"**Hidden Dim:** {hidden_dim}")
                st.write(f"**Epochs:** {epochs}")
                st.write(f"**Batch Size:** {batch_size}")
                if architecture == "Mixture of Experts (MoE)":
                    st.write(f"**Experts:** {num_experts}")
                
                if st.session_state.trained_models:
                    st.subheader("Trained Models")
                    for model_name in st.session_state.trained_models.keys():
                        st.write(f"✅ {model_name}")
    
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
                if hasattr(vectorizer, 'scaler'):
                    question_features = vectorizer.scaler.transform(question_features)
                question_tensor = torch.FloatTensor(question_features)
                
                # Get prediction
                #
# ... inside tab3, inside the "Get Answer" button logic

                # Get prediction
                model.eval()
                with torch.no_grad():
                    # CORRECTED: Unpack the tuple from the MoE model
                    if isinstance(model, RobustMixtureOfExperts):
                        output, gate_weights = model(question_tensor, training=False)
                    else:
                        output = model(question_tensor)
                    
                    # This will now work because 'output' is a tensor
                    predicted_class = output.argmax(dim=1).item()
#
                    confidence = torch.softmax(output, dim=1).max().item()
                
                # Find similar content
                relevant_qa = [qa for qa in st.session_state.qa_data if qa['label'] == predicted_class]
                
                if relevant_qa:
                    st.success(f"**Answer** (Confidence: {confidence:.3f}):")
                    st.write(relevant_qa[0]['answer'])
                    
                    # Show expert usage for MoE
                    if isinstance(model, RobustMixtureOfExperts):
                        st.subheader("Expert Utilization")
                        expert_weights = gate_weights[0].cpu().numpy()
                        expert_df = pd.DataFrame({
                            'Expert': [f'Expert {i+1}' for i in range(len(expert_weights))],
                            'Weight': expert_weights
                        })
                        st.bar_chart(expert_df.set_index('Expert'))
                        
                else:
                    st.info("No relevant information found in the training data.")
            
            # Show some example questions
            if st.session_state.qa_data:
                st.subheader("Example Questions")
                example_questions = [qa['question'] for qa in st.session_state.qa_data[:5]]
                for i, eq in enumerate(example_questions):
                    if st.button(f"Try: {eq}", key=f"example_{i}"):
                        st.text_input("Enter your question:", value=eq, key=f"auto_question_{i}")

def train_model_simple(model, train_loader, val_loader, epochs=10):
    """Simple training function for non-MoE models"""
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

if __name__ == "__main__":
    main()
