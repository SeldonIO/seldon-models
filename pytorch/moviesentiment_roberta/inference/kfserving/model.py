import kfserving
from typing import List, Dict
import torch
from simpletransformers.model import TransformerModel
import numpy as np

class RobertaModel(kfserving.KFModel):
    def __init__(self, name: str):
        super().__init__(name)
        self.name = name
        self.ready = False

    def load(self):
        model = TransformerModel('roberta', 'roberta-base', args=({'fp16': False}))
        model.model.load_state_dict(torch.load('outputs/pytorch_model.bin'))
        self.model = model
        self.ready = True

    def predict(self, request: Dict) -> Dict:
        inputs = np.array(request["instances"])

        try:
            return { "predictions":  self.model.predict(inputs)[1].argmax(axis=1).tolist() }
        except Exception as e:
            raise Exception("Failed to predict %s" % e)

if __name__ == "__main__":
    model = RobertaModel("roberta")
    # Set number of workers to 1 as model is quite large
    kfserving.KFServer(workers=1).start([model])