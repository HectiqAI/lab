import torch
import argparse
from train import SentimentModel
from hectiq import Lab

def predict_sentiment(text, model, vocab, tokenizer):
    model.eval()
    with torch.no_grad():
        tokens = vocab(tokenizer(text))
        tokens = torch.tensor(tokens).unsqueeze(0)
        output = model(tokens)
        prediction = torch.argmax(output, dim=1)
        return "Positive" if prediction.item() == 1 else "Negative"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--text', type=str, required=True)
    args = parser.parse_args()
    
    # Load model and vocab
    lab = Lab(project_name='sentiment-analysis-example')
    model = lab.load_model('sentiment_model.pth')
    
    # Make prediction
    sentiment = predict_sentiment(args.text, model)
    print(f"Text: {args.text}")
    print(f"Sentiment: {sentiment}")

if __name__ == '__main__':
    main()