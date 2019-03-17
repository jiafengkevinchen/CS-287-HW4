from namedtensor import ntorch
from namedtensor.nn import nn as nnn

class FeedFwd(nnn.Module):
    def __init__(self, d_in, d_out, name_in, name_out, dropout_p=.2, hidden_n=200):
        super().__init__()
        self.w1 = nnn.Linear(d_in, hidden_n).spec(name_in, "hidden1")
        self.w2 = nnn.Linear(hidden_n, hidden_n).spec("hidden1", "hidden2")
        self.w3 = nnn.Linear(hidden_n, d_out).spec("hidden2", name_out)
        self.drop = nnn.Dropout(p=dropout_p)
    
    def forward(self, x):
        x = self.drop(ntorch.relu(self.w1(x)))
        x = self.drop(ntorch.relu(self.w2(x)))
        return self.w3(x)

class VanillaDecompAttn(nnn.Module):
    def __init__(self, TEXT, LABEL, embed_dim=200, comp_dim=100):
        super().__init__()
        
        padding_idx = TEXT.vocab.stoi['<pad>']
        original_embed_dim = TEXT.vocab.vectors.size('embedding')
        num_classes = len(LABEL.vocab)

        # this doesn't get updated
        self.embed = nnn.Embedding(TEXT.vocab.vectors.size('word'), embed_dim, 
                      padding_idx=padding_idx) \
                .from_pretrained(TEXT.vocab.vectors.values)

        # project the unchanged embedding into something smaller
        self.embed_proj = nnn.Linear(original_embed_dim, embed_dim) \
            .spec('embedding', 'embedding')

        # F, G, H respectively in the literature
        #attn_w = nnn.Linear(embed_dim, embed_dim).spec('embedding', 'attnembedding')
        #match_w = nnn.Linear(embed_dim * 2, comp_dim).spec('embedding', 'matchembedding')
        #classifier_w = nnn.Linear(2 * comp_dim, num_classes).spec('matchembedding', 'classes')

        self.attn_w = FeedFwd(embed_dim, embed_dim, 'embedding', 'attnembedding')
        self.match_w = FeedFwd(embed_dim * 2, comp_dim, 'embedding', 'matchembedding')
        self.classifier_w = FeedFwd(2 * comp_dim, num_classes, 
                                  'matchembedding', 'classes', dropout_p=0)
    
    def forward(self, hypothesis, premise):
        embed, embed_proj, attn_w, match_w, classifier_w = (
            self.embed, self.embed_proj, self.attn_w, self.match_w, self.classifier_w)
        
        premise_embed = embed_proj(embed(premise)).rename('seqlen', 'premseqlen')
        hypothesis_embed = embed_proj(embed(hypothesis)).rename('seqlen', 'hypseqlen')

        premise_keys = attn_w(premise_embed)
        hypothesis_keys = attn_w(hypothesis_embed)

        log_alignments = ntorch.dot('attnembedding', premise_keys, hypothesis_keys)

        premise_attns = log_alignments.softmax('hypseqlen').dot('hypseqlen', hypothesis_embed)
        hypothesis_attns = log_alignments.softmax('premseqlen').dot('premseqlen', premise_embed)

        premise_concat = ntorch.cat([premise_embed, premise_attns], 'embedding')
        hypothesis_concat = ntorch.cat([hypothesis_embed, hypothesis_attns], 'embedding')

        result_vec = ntorch.cat([
            match_w(premise_concat).sum('premseqlen'),
            match_w(hypothesis_concat).sum('hypseqlen')],
            'matchembedding')

        return classifier_w(result_vec)
