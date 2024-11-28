# Sentiment Analysis Example

This example demonstrates how to train a simple sentiment analysis model using HectiqAI Lab. The model will classify text as either positive or negative sentiment.

## Dataset

We'll use the IMDB Movie Reviews dataset for this example, which contains 50,000 movie reviews labeled as positive or negative.

## Requirements

```bash
pip install hectiq-lab torch torchtext
```

## Model Architecture

We'll implement a simple LSTM-based model for sentiment classification.

## Training

Run the training script:

```bash
python train.py
```

## Inference

Use the trained model for inference:

```bash
python inference.py --text "This movie was fantastic!"
```

## Results

The model typically achieves around 85% accuracy on the test set after training for 3 epochs.