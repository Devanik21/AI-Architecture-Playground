5import streamlit as st
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

# Set page config with dark theme
st.set_page_config(
    page_title=" ML Architecture Playground", 
    page_icon="♾️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark mode styling
st.markdown("""
<style>
    /* Import futuristic fonts */
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Exo+2:wght@300;400;500;700&display=swap');
    
    /* Main theme colors */
    :root {
        --primary-bg: #0a0e1a;
        --secondary-bg: #1a1f2e;
        --accent-bg: #252b42;
        --neon-blue: #00d4ff;
        --neon-purple: #7c3aed;
        --neon-green: #00ff88;
        --neon-orange: #ff6b35;
        --neon-pink: #ff1b6b;
        --text-primary: #ffffff;
        --text-secondary: #a0a9c0;
        --border-color: #2d3748;
    }
    
    /* Hide default Streamlit styling */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main app background */
    .stApp {
        background: linear-gradient(135deg, var(--primary-bg) 0%, #0f1419 100%);
        color: var(--text-primary);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, var(--secondary-bg) 0%, var(--accent-bg) 100%);
        border-right: 2px solid var(--neon-blue);
        box-shadow: 0 0 20px rgba(0, 212, 255, 0.3);
    }
    
    /* Title styling */
    .main-title {
        font-family: 'Orbitron', monospace;
        font-size: 3.5rem;
        font-weight: 900;
        text-align: center;
        background: linear-gradient(45deg, var(--neon-blue), var(--neon-purple), var(--neon-green));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 0 30px rgba(0, 212, 255, 0.5);
        margin-bottom: 1rem;
        animation: glow 2s ease-in-out infinite alternate;
    }
    
    @keyframes glow {
        from { text-shadow: 0 0 20px rgba(0, 212, 255, 0.5); }
        to { text-shadow: 0 0 40px rgba(0, 212, 255, 0.8), 0 0 60px rgba(124, 58, 237, 0.6); }
    }
    
    /* Subtitle styling */
    .subtitle {
        font-family: 'Exo 2', sans-serif;
        font-size: 1.2rem;
        text-align: center;
        color: var(--text-secondary);
        margin-bottom: 2rem;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        background: var(--secondary-bg);
        border-radius: 10px;
        padding: 5px;
        border: 1px solid var(--border-color);
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
    }
    
    .stTabs [data-baseweb="tab"] {
        font-family: 'Exo 2', sans-serif;
        font-weight: 500;
        background: transparent;
        color: var(--text-secondary);
        border-radius: 8px;
        margin: 2px;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(45deg, var(--neon-blue), var(--neon-purple));
        color: white !important;
        box-shadow: 0 0 20px rgba(0, 212, 255, 0.4);
    }
    
    /* Button styling */
    .stButton > button {
        font-family: 'Exo 2', sans-serif;
        font-weight: 600;
        background: linear-gradient(45deg, var(--neon-purple), var(--neon-pink));
        border: none;
        border-radius: 25px;
        color: white;
        padding: 0.7rem 2rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(124, 58, 237, 0.3);
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(124, 58, 237, 0.5);
        background: linear-gradient(45deg, var(--neon-pink), var(--neon-orange));
    }
    
    /* Success/Info/Warning boxes */
    .stAlert > div {
        font-family: 'Exo 2', sans-serif;
        border-radius: 15px;
        border-left: 4px solid var(--neon-green);
        background: rgba(0, 255, 136, 0.1);
        backdrop-filter: blur(10px);
    }
    
    .stWarning > div {
        border-left-color: var(--neon-orange);
        background: rgba(255, 107, 53, 0.1);
    }
    
    .stError > div {
        border-left-color: var(--neon-pink);
        background: rgba(255, 27, 107, 0.1);
    }
    
    .stInfo > div {
        border-left-color: var(--neon-blue);
        background: rgba(0, 212, 255, 0.1);
    }
    
    /* Progress bar */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, var(--neon-blue), var(--neon-purple), var(--neon-green));
        height: 8px;
        border-radius: 10px;
    }
    
    /* File uploader */
    .stFileUploader > div {
        background: var(--secondary-bg);
        border: 2px dashed var(--neon-blue);
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .stFileUploader > div:hover {
        border-color: var(--neon-green);
        box-shadow: 0 0 20px rgba(0, 255, 136, 0.3);
    }
    
    /* Selectbox and inputs */
    .stSelectbox > div > div {
        background: var(--secondary-bg);
        border: 1px solid var(--border-color);
        border-radius: 10px;
        color: var(--text-primary);
    }
    
    .stTextInput > div > div > input {
        background: var(--secondary-bg);
        border: 1px solid var(--border-color);
        border-radius: 10px;
        color: var(--text-primary);
        font-family: 'Exo 2', sans-serif;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: var(--neon-blue);
        box-shadow: 0 0 15px rgba(0, 212, 255, 0.3);
    }
    
    /* Slider */
    .stSlider > div > div > div > div {
        background: var(--neon-blue);
    }
    
    /* Dataframe */
    .stDataFrame {
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }
    
    /* Headers */
    h1, h2, h3 {
        font-family: 'Orbitron', monospace;
        color: var(--text-primary);
    }
    
    h2 {
        color: var(--neon-blue);
        border-bottom: 2px solid var(--neon-blue);
        padding-bottom: 10px;
        margin-top: 2rem;
    }
    
    h3 {
        color: var(--neon-purple);
    }
    
    /* Sidebar headers */
    .css-1d391kg h2, .css-1d391kg h3 {
        color: var(--neon-green);
        text-align: center;
    }
    
    /* Architecture cards */
    .architecture-card {
        background: linear-gradient(135deg, var(--secondary-bg) 0%, var(--accent-bg) 100%);
        border: 1px solid var(--border-color);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
    }
    
    .architecture-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.4);
        border-color: var(--neon-blue);
    }
    
    /* Metric containers */
    .metric-container {
        background: var(--secondary-bg);
        border-radius: 10px;
        padding: 1rem;
        border-left: 4px solid var(--neon-green);
        margin: 0.5rem 0;
    }
    
    /* Neural network visual elements */
    .neural-decoration {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        pointer-events: none;
        opacity: 0.1;
        z-index: -1;
    }
    
    /* Status indicators */
    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
    }
    
    .status-success { background-color: var(--neon-green); }
    .status-training { 
        background-color: var(--neon-orange);
        animation: pulse 1s infinite;
    }
    .status-ready { background-color: var(--neon-blue); }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--primary-bg);
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(45deg, var(--neon-blue), var(--neon-purple));
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(45deg, var(--neon-purple), var(--neon-pink));
    }
    
    /* Code blocks */
    .stCodeBlock {
        background: var(--secondary-bg) !important;
        border: 1px solid var(--border-color);
        border-radius: 10px;
    }
    
    /* Tooltips and hover effects */
    .model-tooltip {
        position: relative;
        display: inline-block;
        cursor: help;
        color: var(--neon-blue);
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

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
# Mixture of Experts Model
class Expert(nn.Module):
    """A deeper, more capable expert network."""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Expert, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),  # Additional hidden layer
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, output_dim)
        )
    
    def forward(self, x):
        return self.network(x)

class MixtureOfExperts(nn.Module):
    """
    An advanced Mixture of Experts model with a deeper gating network 
    and Top-K routing for improved specialization and accuracy.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_experts=4, top_k=2):
        super(MixtureOfExperts, self).__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        
        # Create a list of expert models
        self.experts = nn.ModuleList([
            Expert(input_dim, hidden_dim, output_dim) for _ in range(num_experts)
        ])
        
        # Create a deeper gating network for smarter routing
        self.gating = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_experts)
        )
    
    def forward(self, x):
        # 1. Get raw scores (logits) from the gating network
        gate_logits = self.gating(x)  # Shape: [batch_size, num_experts]
        
        # 2. Find the top-k experts and their corresponding scores
        # We use torch.topk to select the best experts for each item in the batch
        top_k_weights, top_k_indices = torch.topk(gate_logits, self.top_k, dim=1) # Shape: [batch_size, top_k]
        
        # 3. Normalize the scores of the selected experts using Softmax
        top_k_weights = nn.functional.softmax(top_k_weights, dim=1)
        
        # 4. Get the outputs from all experts
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=2) # Shape: [batch_size, output_dim, num_experts]

        # 5. Create a sparse tensor for the final gating weights
        # This tensor will have zeros for the experts that were not selected
        final_gate_weights = torch.zeros_like(gate_logits) # Shape: [batch_size, num_experts]
        final_gate_weights.scatter_(1, top_k_indices, top_k_weights) # Fill with top-k weights

        # 6. Combine the expert outputs using the sparse weights
        # We unsqueeze the weights to match dimensions for broadcasting
        final_gate_weights = final_gate_weights.unsqueeze(1) # Shape: [batch_size, 1, num_experts]
        
        # The final output is a weighted sum of the expert outputs
        output = torch.sum(expert_outputs * final_gate_weights, dim=2) # Shape: [batch_size, output_dim]
        
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
        status_text.markdown(f'**Epoch {epoch+1}/{epochs}** | Train Loss: `{train_loss:.4f}` | Val Loss: `{val_loss:.4f}` | Accuracy: `{accuracy:.2f}%`')
    
    return train_losses, val_losses

def plot_training_curves(train_losses, val_losses):
    """Plot training curves with dark theme"""
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Set background
    fig.patch.set_facecolor('#0a0e1a')
    ax.set_facecolor('#1a1f2e')
    
    # Plot with neon colors
    ax.plot(train_losses, label='Training Loss', color='#00d4ff', linewidth=3)
    ax.plot(val_losses, label='Validation Loss', color='#7c3aed', linewidth=3)
    
    ax.set_xlabel('Epoch', color='#ffffff', fontsize=12)
    ax.set_ylabel('Loss', color='#ffffff', fontsize=12)
    ax.set_title('✨ Training Progress', color='#00d4ff', fontsize=16, fontweight='bold')
    
    # Customize legend
    legend = ax.legend(facecolor='#252b42', edgecolor='#00d4ff')
    legend.get_frame().set_alpha(0.9)
    for text in legend.get_texts():
        text.set_color('#ffffff')
    
    # Grid styling
    ax.grid(True, alpha=0.3, color='#2d3748')
    ax.tick_params(colors='#a0a9c0')
    
    # Border styling
    for spine in ax.spines.values():
        spine.set_color('#2d3748')
        spine.set_linewidth(2)
    
    plt.tight_layout()
    return fig

def display_architecture_info(architecture):
    """Display architecture information with styling"""
    arch_info = {
        "Mixture of Experts (MoE)": {
            "emoji": "🧠",
            "description": "Advanced ensemble of specialized neural networks with gating mechanism",
            "color": "#7c3aed"
        },
        "Simple Transformer": {
            "emoji": "⚡",
            "description": "Attention-based model inspired by the Transformer architecture",
            "color": "#00d4ff"
        },
        "CNN": {
            "emoji": "🔍",
            "description": "Convolutional Neural Network for pattern recognition",
            "color": "#00ff88"
        },
        "LSTM": {
            "emoji": "🔄",
            "description": "Long Short-Term Memory for sequential data processing",
            "color": "#ff6b35"
        },
        "MLP": {
            "emoji": "🎯",
            "description": "Multi-Layer Perceptron - Classic feedforward network",
            "color": "#ff1b6b"
        }
    }
    
    info = arch_info.get(architecture, {})
    return f"{info.get('emoji', '🌊')} **{architecture}**\n\n{info.get('description', 'Neural network architecture')}"

def main():
    # Main title with custom styling
    st.markdown('<h1 class="main-title">✨ ML ARCHITECTURE PLAYGROUND</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">🌌 Upload PDFs • Train Neural Networks • Test Intelligence 🌌</p>', unsafe_allow_html=True)
    
    # Add neural network decoration
    st.markdown("""
    <div class="neural-decoration">
        <svg width="100%" height="100%" viewBox="0 0 1200 800">
            <defs>
                <linearGradient id="neuralGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" stop-color="#00d4ff" stop-opacity="0.1"/>
                    <stop offset="50%" stop-color="#7c3aed" stop-opacity="0.1"/>
                    <stop offset="100%" stop-color="#00ff88" stop-opacity="0.1"/>
                </linearGradient>
            </defs>
        </svg>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for model selection with enhanced styling
    with st.sidebar:
        st.markdown("## 🎛️ Neural Control Center")
        
        architecture = st.selectbox(
            "🏗️ Choose Architecture",
            ["Mixture of Experts (MoE)", "Simple Transformer", "CNN", "LSTM", "MLP"],
            help="Select your neural network architecture"
        )
        
        # Display architecture info
        st.markdown(f"""
        <div class="architecture-card">
            {display_architecture_info(architecture)}
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### ⚙️ Hyperparameters")
        
        hidden_dim = st.slider("🔧 Hidden Dimension", 32, 5096, 128, help="Size of hidden layers")
        epochs = st.slider("🔄 Training Epochs", 5, 500, 10, help="Number of training iterations")
        batch_size = st.slider("📦 Batch Size", 4, 256, 4, help="Samples per training batch")
        
        # Status indicators
        st.markdown("### 📊 System Status")
        
    # Initialize session state
    if 'trained_models' not in st.session_state:
        st.session_state.trained_models = {}
    if 'vectorizers' not in st.session_state:
        st.session_state.vectorizers = {}
    if 'qa_data' not in st.session_state:
        st.session_state.qa_data = []
    
    # Main content with enhanced tabs
    tab1, tab2, tab3 = st.tabs(["📁 Data Upload", "🧪 Model Training", "❓ Q&A Testing"])
    
    with tab1:
        st.markdown("## 📂 Upload PDF Data")
        st.markdown("*Transform your documents into neural training data*")
        
        uploaded_files = st.file_uploader(
            "✨ Choose PDF files for neural training",
            type="pdf",
            accept_multiple_files=True,
            help="Upload one or multiple PDF files to create training data"
        )
        
        if uploaded_files:
            st.markdown(f"### 📋 Status: {len(uploaded_files)} PDF file(s) loaded")
            
            # Display file info in columns
            cols = st.columns(min(len(uploaded_files), 3))
            for i, file in enumerate(uploaded_files):
                with cols[i % 3]:
                    st.markdown(f"""
                    <div class="metric-container">
                        <strong>📄 {file.name}</strong><br>
                        <span style="color: #00d4ff;">Size: {file.size:,} bytes</span>
                    </div>
                    """, unsafe_allow_html=True)
            
            if st.button("🔄 Process Neural Data", type="primary"):
                with st.spinner("🧠 Processing neural pathways..."):
                    all_text = ""
                    for uploaded_file in uploaded_files:
                        text = extract_text_from_pdf(uploaded_file)
                        all_text += text + "\n\n"
                    
                    if all_text:
                        processed_text = preprocess_text(all_text)
                        
                        # Check if we have enough text
                        if len(processed_text) < 100:
                            st.warning("⚠️ PDF content seems too short. Please upload a larger PDF or multiple PDFs.")
                            return
                        
                        qa_pairs = create_qa_pairs(processed_text)
                        st.session_state.qa_data = qa_pairs
                        
                        st.success(f"✅ Neural data ready! Processed {len(qa_pairs)} training samples")
                        
                        # Display metrics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("📊 Text Length", f"{len(processed_text):,}", "characters")
                        with col2:
                            st.metric("🧠 Training Samples", len(qa_pairs))
                        with col3:
                            st.metric("📁 Files Processed", len(uploaded_files))
                        
                        # Show sample data
                        if qa_pairs:
                            st.markdown("### 🔍 Sample Training Data")
                            sample_df = pd.DataFrame(qa_pairs[:5])
                            st.dataframe(sample_df, use_container_width=True)
    
    with tab2:
        st.markdown("## 🧪 Neural Network Training Laboratory")
        st.markdown("*Where artificial intelligence comes to life*")
        
        if not st.session_state.qa_data:
            st.markdown("""
            <div style="text-align: center; padding: 3rem; background: linear-gradient(135deg, rgba(255, 107, 53, 0.1), rgba(255, 27, 107, 0.1)); border-radius: 15px; border: 2px dashed #ff6b35;">
                <h3 style="color: #ff6b35;">⚠️ Neural Data Required</h3>
                <p style="color: #a0a9c0;">Please upload and process PDF data first to begin training!</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown(f"""
                <div class="architecture-card">
                    <h4 style="color: #00d4ff;">✨ Ready for Neural Training</h4>
                    <p><strong>Architecture:</strong> <span style="color: #00ff88;">{architecture}</span></p>
                    <p><strong>Training Samples:</strong> <span style="color: #00ff88;">{len(st.session_state.qa_data)}</span></p>
                    <p><strong>Status:</strong> <span class="status-indicator status-ready"></span>Ready to train</p>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button("✨ Initialize Neural Training", type="primary"):
                    with st.spinner("🧠 Training neural pathways..."):
                        # Prepare data
                        texts = [qa['answer'] for qa in st.session_state.qa_data]
                        labels = [qa['label'] for qa in st.session_state.qa_data]
                        
                        # --- FIX STARTS HERE ---
                        # Add a check to ensure there's enough data for a split.
                        # We need at least a few samples to create a training and validation set.
                        if len(texts) < 5: # Setting a minimum of 5 samples
                            st.error(f"❌ Insufficient neural data! You have {len(texts)} samples, but at least 5 are required. Please upload a larger or more text-rich PDF.")
                            st.stop() # Stop execution if data is insufficient
                        # --- FIX ENDS HERE ---
                        
                        # Create dataset
                        train_texts, val_texts, train_labels, val_labels = train_test_split(
                            texts, labels, test_size=0.2, random_state=42
                        )
                        
                        train_dataset = TextDataset(train_texts, train_labels)
                        val_dataset = TextDataset(val_texts, val_labels, train_dataset.vectorizer)
                        
                        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                        val_loader = DataLoader(val_dataset, batch_size=batch_size)
                        
                        input_dim = train_dataset.features.shape[1]
                        output_dim = len(set(labels))
                        
                        # Create model based on selection
                        st.markdown(f"### 🏗️ Constructing {architecture}...")
                        
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
                        
                        # Display model info
                        total_params = sum(p.numel() for p in model.parameters())
                        st.markdown(f"""
                        <div class="metric-container">
                            <h4 style="color: #7c3aed;">🧠 Neural Architecture Stats</h4>
                            <p><strong>Input Dimension:</strong> <span style="color: #00d4ff;">{input_dim}</span></p>
                            <p><strong>Hidden Dimension:</strong> <span style="color: #00d4ff;">{hidden_dim}</span></p>
                            <p><strong>Output Classes:</strong> <span style="color: #00d4ff;">{output_dim}</span></p>
                            <p><strong>Total Parameters:</strong> <span style="color: #00ff88;">{total_params:,}</span></p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Train model
                        st.markdown("### ⚡ Neural Training in Progress")
                        train_losses, val_losses = train_model(model, train_loader, val_loader, epochs)
                        
                        # Store trained model
                        st.session_state.trained_models[architecture] = model
                        st.session_state.vectorizers[architecture] = train_dataset.vectorizer
                        
                        st.success("✅ Neural training completed successfully!")
                        
                        # Plot results
                        st.markdown("### 📈 Training Performance Analysis")
                        fig = plot_training_curves(train_losses, val_losses)
                        st.pyplot(fig)
                        
                        # Final metrics
                        final_train_loss = train_losses[-1]
                        final_val_loss = val_losses[-1]
                        
                        metric_col1, metric_col2, metric_col3 = st.columns(3)
                        with metric_col1:
                            st.metric("🎯 Final Train Loss", f"{final_train_loss:.4f}")
                        with metric_col2:
                            st.metric("🎯 Final Val Loss", f"{final_val_loss:.4f}")
                        with metric_col3:
                            improvement = ((train_losses[0] - final_train_loss) / train_losses[0] * 100)
                            st.metric("📈 Improvement", f"{improvement:.1f}%")
            
            with col2:
                st.markdown("""
                <div class="architecture-card">
                    <h4 style="color: #00ff88;">⚙️ Training Configuration</h4>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"**🏗️ Architecture:** `{architecture}`")
                st.markdown(f"**🔧 Hidden Dim:** `{hidden_dim}`")
                st.markdown(f"**🔄 Epochs:** `{epochs}`")
                st.markdown(f"**📦 Batch Size:** `{batch_size}`")
                
                if st.session_state.trained_models:
                    st.markdown("### 🌊 Trained Models")
                    for i, model_name in enumerate(st.session_state.trained_models.keys()):
                        status_color = ["#00ff88", "#00d4ff", "#7c3aed", "#ff6b35", "#ff1b6b"][i % 5]
                        st.markdown(f'<span class="status-indicator" style="background-color: {status_color};"></span> **{model_name}**', unsafe_allow_html=True)
    
    with tab3:
        st.markdown("## 🌊 Neural Q&A Testing Interface")
        st.markdown("*Query your trained artificial intelligence*")
        
        if not st.session_state.trained_models:
            st.markdown("""
            <div style="text-align: center; padding: 3rem; background: linear-gradient(135deg, rgba(255, 107, 53, 0.1), rgba(255, 27, 107, 0.1)); border-radius: 15px; border: 2px dashed #ff6b35;">
                <h3 style="color: #ff6b35;">🌊 Neural Models Required</h3>
                <p style="color: #a0a9c0;">Please train a neural model first to begin testing!</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Model selection for testing
            st.markdown("### 🎯 Select Neural Model")
            test_model = st.selectbox(
                "🧠 Choose your trained neural network:",
                list(st.session_state.trained_models.keys()),
                help="Select which trained model to use for answering questions"
            )
            
            # Display model info
            st.markdown(f"""
            <div class="architecture-card">
                <h4 style="color: #00d4ff;">✨ Active Neural Model</h4>
                <p>{display_architecture_info(test_model)}</p>
                <p><strong>Status:</strong> <span class="status-indicator status-success"></span>Ready for inference</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Question input with styling
            st.markdown("### 💭 Ask Your Question")
            question = st.text_input(
                "🔍 Enter your question:",
                placeholder="What would you like to know about the uploaded documents?",
                help="Ask any question related to the content of your uploaded PDFs"
            )
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                if question and st.button("✨ Generate Neural Response", type="primary"):
                    model = st.session_state.trained_models[test_model]
                    vectorizer = st.session_state.vectorizers[test_model]
                    
                    with st.spinner("🧠 Neural network processing..."):
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
                            st.markdown(f"""
                            <div class="architecture-card">
                                <h4 style="color: #00ff88;">🎯 Neural Response</h4>
                                <p><strong>Confidence:</strong> <span style="color: #00d4ff;">{confidence:.2%}</span></p>
                                <p><strong>Model:</strong> <span style="color: #7c3aed;">{test_model}</span></p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.markdown("### 💡 Answer")
                            st.markdown(f"""
                            <div style="background: linear-gradient(135deg, rgba(0, 255, 136, 0.1), rgba(0, 212, 255, 0.1)); 
                                        padding: 2rem; border-radius: 15px; border-left: 4px solid #00ff88; margin: 1rem 0;">
                                {relevant_qa[0]['answer']}
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Confidence meter
                            st.markdown("### 📊 Confidence Analysis")
                            confidence_color = "#00ff88" if confidence > 0.7 else "#ff6b35" if confidence > 0.4 else "#ff1b6b"
                            st.markdown(f"""
                            <div style="background: {confidence_color}20; padding: 1rem; border-radius: 10px; border: 1px solid {confidence_color};">
                                <div style="background: {confidence_color}; height: 10px; width: {confidence*100}%; border-radius: 5px; margin-bottom: 10px;"></div>
                                <p style="color: {confidence_color}; margin: 0;"><strong>Neural Confidence: {confidence:.1%}</strong></p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                        else:
                            st.markdown("""
                            <div style="background: linear-gradient(135deg, rgba(255, 107, 53, 0.1), rgba(255, 27, 107, 0.1)); 
                                        padding: 2rem; border-radius: 15px; border-left: 4px solid #ff6b35;">
                                <h4 style="color: #ff6b35;">🔍 No Relevant Information Found</h4>
                                <p>The neural network couldn't find relevant information in the training data for this question.</p>
                            </div>
                            """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div class="architecture-card">
                    <h4 style="color: #7c3aed;">💡 Tips</h4>
                    <p style="font-size: 0.9rem;">• Be specific in your questions</p>
                    <p style="font-size: 0.9rem;">• Use keywords from your documents</p>
                    <p style="font-size: 0.9rem;">• Try different phrasings</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Show example questions
            if st.session_state.qa_data:
                st.markdown("### 🎯 Example Neural Queries")
                st.markdown("*Click any example to test the neural network*")
                
                example_questions = [qa['question'] for qa in st.session_state.qa_data[:8]]
                
                # Display examples in a grid
                cols = st.columns(2)
                for i, eq in enumerate(example_questions):
                    with cols[i % 2]:
                        if st.button(f"🔮 {eq}", key=f"example_{i}", help="Click to use this example question"):
                            # Auto-fill the question
                            st.rerun()
                
                # Additional neural statistics
                st.markdown("### 📊 Neural Dataset Statistics")
                
                stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
                with stat_col1:
                    st.metric("📚 Total Samples", len(st.session_state.qa_data))
                with stat_col2:
                    st.metric("🏷️ Categories", len(set([qa['label'] for qa in st.session_state.qa_data])))
                with stat_col3:
                    avg_length = np.mean([len(qa['answer']) for qa in st.session_state.qa_data])
                    st.metric("📏 Avg Length", f"{avg_length:.0f}")
                with stat_col4:
                    st.metric("🧠 Models Trained", len(st.session_state.trained_models))

if __name__ == "__main__":
    main()
