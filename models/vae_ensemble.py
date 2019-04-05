import torch
from torch.distributions import Categorical

from namedtensor import NamedTensor
from namedtensor import ntorch
from namedtensor.nn import nn as nnn

def logsumexp(named_tensor, dim_name):
    names = list(named_tensor.shape.keys())
    dim_num = names.index(dim_name)
    names.pop(dim_num)
    return NamedTensor(torch.logsumexp(named_tensor.values, dim_num, keepdim=False), 
                names=names)

class VAEEnsemble(nnn.Module):
    def __init__(self, models, q, num_classes=4):
        super().__init__()
        self.models = nnn.ModuleList(models)
        self.q = q
        self.ce_loss = nnn.CrossEntropyLoss(reduction='none').spec('classes')
        self.num_classes = num_classes
    
    def forward(self, hypothesis, premise, y=None):
        if self.training:
            assert(y is not None)
            weights = self.q(hypothesis, premise).softmax('classes')
            m = Categorical(weights.values)
            models = NamedTensor(m.sample(), names=('batch',))
            
            global_log_probs = ntorch.zeros(hypothesis.size('batch'), 
                                            self.num_classes, 
                                            names=('batch', 'classes'),
                                            device=hypothesis.values.device)
            
            for i in range(len(self.models)):
                is_model = models == i
                if is_model.sum().item() == 0:
                    continue
                model_batches = is_model.nonzero(names=('batch', 'extra'))[{'extra': 0}]
                model_hypothesis = hypothesis[{'batch': model_batches}]
                model_premise = premise[{'batch': model_batches}]

                log_probs = self.models[i](model_hypothesis, model_premise)
            
                global_log_probs[{'batch': model_batches}] = log_probs
        
            loss = -m.log_prob(models.values) * \
                self.ce_loss(global_log_probs, y).values
        
            return loss.sum()
        
        else:
            log_preds = ntorch.stack([
                model(hypothesis, premise) for model in self.models
            ], 'model')

            unnorm_preds = logsumexp(log_preds, 'model')
            normalizing_factor = logsumexp(unnorm_preds, 'classes')
            return (unnorm_preds - normalizing_factor)
            #return ntorch.log(log_preds.softmax('classes').mean('model'))

