from namedtensor import ntorch
from namedtensor.nn import nn as nnn

class ExactEnsemble(nnn.Module):
    def __init__(self, models):
        super().__init__()
        self.models = nnn.ModuleList(models)
        
    def forward(self, hypothesis, premise):
        log_preds = ntorch.stack([
            model(hypothesis, premise) for model in self.models
        ], 'model')
        
        return log_preds.softmax('classes').mean('model').log()
