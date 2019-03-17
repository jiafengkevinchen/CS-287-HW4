import os
import subprocess
import base64
import torch
import subprocess
import pandas as pd
import numpy as np


from IPython.paths import get_ipython_dir
from urllib.request import urlretrieve
from namedtensor import ntorch, NamedTensor
from torch import Tensor
from tqdm import tqdm_notebook as tqdm


def train_model(
    model, loss_fn=None, optimizer=None, train_iter=None,
    val_iter=None, num_epochs=5, report_frequency=100, writer=None, callback=None,
    inner_callback=None, progress_bar=False, patience=1, run_name="model"):
    """
    TODO
    """

    if hasattr(model, '__train__'):
        model.__train__()
    elif loss_fn is None or optimizer is None:
        raise ValueError
    else:
        num_batches = 0
        train_loss_write = 0
        val_loss_up = 0
        best_model_dict = model.state_dict()

        last_val_loss = np.inf
        try:
            for epoch in range(num_epochs):
                train_loss = 0
                if not progress_bar:
                    tqdm = lambda x : x
                else:
                    from tqdm import tqdm_notebook as tqdm
                for batch in tqdm(train_iter):
                    optimizer.zero_grad()
                    loss = loss_fn(model, batch)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()

                    if inner_callback is not None:
                        inner_callback(**locals())


                    num_batches += 1
                    train_loss_write += loss.item()

                    if num_batches % report_frequency == 0 and writer is not None:
                        if writer is not None:
                            writer.add_scalar(f"{run_name}/training_loss",
                                train_loss_write / report_frequency,
                                num_batches // report_frequency)
                            train_loss_write = 0


                        if val_iter is not None:
                            model.eval()
                            val_loss = 0
                            for batch in val_iter:
                                loss = loss_fn(model, batch)
                                val_loss += loss.item()
                            model.train()
                            if writer is not None:
                                writer.add_scalar(
                                    f'{run_name}/validation_loss', val_loss / len(val_iter),
                                    num_batches // report_frequency)

                # End of epoch validation
                if val_iter is not None:
                    model.eval()
                    val_loss = 0
                    for batch in val_iter:
                        loss = loss_fn(model, batch)
                        val_loss += loss.item()
                    model.train()
                    if val_loss / len(val_iter) > last_val_loss:
                        val_loss_up += 1
                    else:
                        val_loss_up = 0
                        best_model_dict = model.state_dict()
                        last_val_loss = val_loss / len(val_iter)
                    if val_loss_up >= patience:
                        print("Patience exceeded. Early stopping...")
                        return best_model_dict
                if callback is not None:
                    callback(**locals())
        except KeyboardInterrupt:
            print("Don't interrupt me! Fine. Here's your model.")
            return best_model_dict
        return best_model_dict


def tensor_to_text(t, TEXT):
    return ' '.join([TEXT.vocab.itos[i] for i in t])

def kaggle_loss(model, batch, cuda=True):
    _, best_words = ntorch.topk(model(batch.text)[{'seqlen': -1}], 'classes', 20)
    last_target = batch.target[{'seqlen': -1}]
    is_correct = best_words == last_target
    if cuda:
        scores = is_correct.values.float() / (1 + torch.arange(20)).float().cuda()
    else:
        scores = is_correct.values.float() / (1 + torch.arange(20)).float()
    return scores.sum()


def count_parameters(model):
    total = 0
    for p in model.parameters():
        total += p.numel()
    return total


def test_model(model, input_file, filename, TEXT, output_name="classes", use_cuda=False):
    row_num = 0
    V = len(TEXT.vocab)
    with open(filename, "w") as fout:
        print('id,word', file=fout)
        with open(input_file, 'r') as fin:
            for line in tqdm(fin.readlines()):
                if use_cuda:
                    batch_text = NamedTensor(
                    Tensor([TEXT.vocab.stoi[s] for s in line.split(' ')[:-1]]).unsqueeze(1).long(),
                    names=('seqlen', 'batch')
                    ).cuda()
                else:
                    batch_text = NamedTensor(
                        Tensor([TEXT.vocab.stoi[s] for s in line.split(' ')[:-1]]).unsqueeze(1).long(),
                        names=('seqlen', 'batch')
                    )

                _, best_words = ntorch.topk(model(batch_text)[{'seqlen': -1,
                    'classes' : slice(1, V)}],
                                            output_name, 20)
                best_words += 1
                for row in best_words.cpu().numpy():
                    row_num += 1
                    print(f'{row_num},{tensor_to_text(row, TEXT)}', file=fout)


def evaluate_model(val_iter, args, **models):
    results = []
    for i, batch in enumerate(tqdm(val_iter)):
        for name in models:
            model, loss_fn = models[name]
            batch_size = batch.batch_size
            loss = loss_fn(model, batch).item()
            map_ = kaggle_loss(model, batch, **args).item()
            results.append({
                'batch_id' : i,
                'batch_size': batch_size,
                'model_name':name,
                'loss': loss, 'map' : map_ / batch_size})
    return pd.DataFrame(results)

def flatten(x, dim_kill, dim_grow):
    return ntorch.cat(x.unbind(dim_kill), dim=dim_grow)

def unsqueeze(x, dim):
    return ntorch.stack([x], dim)
