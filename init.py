import random

import torch
import torchtext
from torchtext.vocab import Vectors, GloVe

from namedtensor import ntorch, NamedTensor
from namedtensor.text import NamedField

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Our input $x$
TEXT = NamedField(names=('seqlen',))

# Our labels $y$
LABEL = NamedField(sequential=False, names=())

train, val, test = torchtext.datasets.SNLI.splits(
    TEXT, LABEL)

TEXT.build_vocab(train)
LABEL.build_vocab(train)

train_iter, val_iter, test_iter = torchtext.data.BucketIterator.splits(
    (train, val, test), batch_size=16, device=device, repeat=False)

batch = next(iter(train_iter))
premise = batch.premise.get("batch", 1)
hypothesis = batch.hypothesis.get("batch", 1)
example = batch.label.get("batch", 1)

# Build the vocabulary with word embeddings
# Out-of-vocabulary (OOV) words are hashed to one of 100 random embeddings each
# initialized to mean 0 and standarad deviation 1 (Sec 5.1)
unk_vectors = [torch.randn(300) for _ in range(100)]
TEXT.vocab.load_vectors(vectors='glove.6B.300d',
                        unk_init=lambda x:random.choice(unk_vectors))

# normalized to have l_2 norm of 1
vectors = TEXT.vocab.vectors
vectors = vectors / vectors.norm(dim=1,keepdim=True)
vectors = NamedTensor(vectors, ('word', 'embedding'))
TEXT.vocab.vectors = vectors


def test_code(model, predictions_fname='predictions.txt'):
    "All models should be able to be run with following command."
    upload = []
    # Update: for kaggle the bucket iterator needs to have batch_size 10
    test_iter = torchtext.data.BucketIterator(test, train=False, batch_size=10)
    for batch in test_iter:
        # Your prediction data here (don't cheat!)
        probs = model(batch.text)
        # here we assume that the name for dimension classes is `classes`
        _, argmax = probs.max('classes')
        upload += argmax.tolist()

    with open(predictions_fname, "w") as f:
        for u in upload:
            f.write(str(u) + "\n")
