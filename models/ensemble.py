from namedtensor import ntorch
from namedtensor.nn import nn as nnn

def logsumexp(named_tensor, dim_name):
    names = list(named_tensor.shape.keys())
    dim_num = names.index(dim_name)
    names.pop(dim_num)
    return NamedTensor(torch.logsumexp(named_tensor.values, dim_num, keepdim=False), 
                names=names)

class ExactEnsemble(nnn.Module):
    def __init__(self, models):
        super().__init__()
        self.models = nnn.ModuleList(models)
        
    def forward(self, hypothesis, premise):
        log_preds = ntorch.stack([
            model(hypothesis, premise) for model in self.models
        ], 'model')
        
        return log_preds.softmax('classes').mean('model').log()
