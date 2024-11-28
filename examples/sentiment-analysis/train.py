import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from hectiq import Lab

# Initialize Lab
lab = Lab(project_name='sentiment-analysis-example')

# Model definition
class SentimentModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 2)  # 2 classes: positive/negative
        
    def forward(self, x):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        # Take the last output
        out = self.fc(lstm_out[:, -1, :])
        return out

# Data processing
def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)

# Main training loop
def train():
    # Setup data
    train_iter = IMDB(split='train')
    tokenizer = get_tokenizer('basic_english')
    vocab = build_vocab_from_iterator(yield_tokens(train_iter), min_freq=20)
    
    # Model setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SentimentModel(
        vocab_size=len(vocab),
        embed_dim=100,
        hidden_dim=256
    ).to(device)
    
    # Training parameters
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    
    # Training loop
    lab.start_run()
    for epoch in range(3):
        model.train()
        for batch_idx, (label, text) in enumerate(train_iter):
            optimizer.zero_grad()
            
            # Process text
            tokens = vocab(tokenizer(text))
            tokens = torch.tensor(tokens).unsqueeze(0).to(device)
            label = torch.tensor(label).to(device)
            
            # Forward pass
            output = model(tokens)
            loss = criterion(output, label)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Log metrics
            if batch_idx % 100 == 0:
                lab.log_metrics({
                    'loss': loss.item(),
                    'epoch': epoch
                })
    
    # Save model
    lab.save_model(model, 'sentiment_model.pth')
    lab.end_run()

if __name__ == '__main__':
    train()