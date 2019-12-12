import torch
from simpletransformers.model import TransformerModel


class MovieSentimentRoBERTa:

    def __init__(self):
        model = TransformerModel('roberta', 'roberta-base', args=({'fp16': False}))
        model.model.load_state_dict(torch.load('outputs/pytorch_model.bin'))
        self.model = model

    def predict(self, X, feature_names, **kwargs):
        print("Predict called")
        X = X.astype('U')
        print(X)
        return self.model.predict(X)[1].argmax(axis=1)
