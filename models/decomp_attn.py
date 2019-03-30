from namedtensor import ntorch
from namedtensor.nn import nn as nnn
from layernorm import LayerNorm


class FeedFwd(nnn.Module):
    def __init__(self, d_in, d_out, name_in, name_out,
                 dropout_p=.2, hidden_n=200):
        super().__init__()
        self.w1 = nnn.Linear(d_in, hidden_n).spec(name_in, "hidden1")
        self.w2 = nnn.Linear(hidden_n, hidden_n).spec("hidden1", "hidden2")
        self.w3 = nnn.Linear(hidden_n, d_out).spec("hidden2", name_out)
        self.drop = nnn.Dropout(p=dropout_p)

    def forward(self, x):
        x = self.drop(ntorch.relu(self.w1(x)))
        x = self.drop(ntorch.relu(self.w2(x)))
        return self.w3(x)


class DecompAttn(nnn.Module):
    def __init__(
            self,
            TEXT,
            LABEL,
            embed_dim=200,
            input_dim=None,
            dropout=0.2):
        super().__init__()

        padding_idx = TEXT.vocab.stoi['<pad>']
        self.padding_idx = padding_idx
        original_embed_dim = TEXT.vocab.vectors.size('embedding')
        num_classes = len(LABEL.vocab)

        self.embed_dim = embed_dim

        # this doesn't get updated
        self.embed = nnn.Embedding(TEXT.vocab.vectors.size('word'), embed_dim,
                                   padding_idx=padding_idx) \
            .from_pretrained(TEXT.vocab.vectors.values)

        # project the unchanged embedding into something smaller
        self.embed_proj = nnn.Linear(original_embed_dim, embed_dim) \
            .spec('embedding', 'embedding')

        if input_dim is None:
            input_dim = embed_dim

        self.attn_w = FeedFwd(input_dim, embed_dim,
                              'embedding', 'attnembedding', dropout_p=dropout)
        self.attn_norm = LayerNorm(embed_dim, 'attnembedding')

        self.match_w = FeedFwd(embed_dim * 2, embed_dim,
                               'embedding', 'matchembedding', dropout_p=dropout)
        self.classifier_w = FeedFwd(embed_dim * 2, num_classes,
                                    'matchembedding', 'classes', dropout_p=0)

    def process_input(self, sentence, seqlen_dim):
        return self.embed_proj(self.embed(sentence))

    def forward(self, hypothesis, premise, debug=False):
        attn_w, match_w, classifier_w = (
            self.attn_w, self.match_w, self.classifier_w)
        premise = premise.rename('seqlen', 'premseqlen')
        hypothesis = hypothesis.rename('seqlen', 'hypseqlen')

        premise_mask = (premise != self.padding_idx).float()
        hypothesis_mask = (hypothesis != self.padding_idx).float()

        log_mask = (1 - premise_mask * hypothesis_mask) * (-1e3)

        # Embedding the premise and the hypothesis
        premise_embed = self.process_input(premise, 'premseqlen')
        hypothesis_embed = self.process_input(hypothesis, 'hypseqlen')

        # Attend
        premise_keys = (attn_w(premise_embed))
        hypothesis_keys = (attn_w(hypothesis_embed))

        # Layernorm
        premise_keys = self.attn_norm(premise_keys)
        hypothesis_keys = self.attn_norm(hypothesis_keys)

        log_alignments = (
            ntorch.dot('attnembedding', premise_keys, hypothesis_keys)
            / (self.embed_dim ** .5) + log_mask)

        premise_attns = log_alignments.softmax(
            'hypseqlen').dot('hypseqlen', hypothesis_embed)
        hypothesis_attns = log_alignments.softmax(
            'premseqlen').dot('premseqlen', premise_embed)
        premise_concat = ntorch.cat(
            [premise_embed, premise_attns], 'embedding')
        hypothesis_concat = ntorch.cat(
            [hypothesis_embed, hypothesis_attns], 'embedding')

        # Compare
        compare_premise = premise_mask * match_w(premise_concat)
        compare_hypothesis = hypothesis_mask * match_w(hypothesis_concat)

        # Aggregate
        result_vec = ntorch.cat([
            compare_premise.sum('premseqlen'),
            compare_hypothesis.sum('hypseqlen')],
            'matchembedding')

        if debug:
            return classifier_w(result_vec), log_alignments
        return classifier_w(result_vec)


class DecompAttnWithIntraAttn(DecompAttn):
    def __init__(
            self,
            TEXT,
            LABEL,
            embed_dim=200,
            max_distance=10,
            **kwargs):
        super().__init__(TEXT, LABEL, embed_dim=embed_dim,
            input_dim=2 * embed_dim, **kwargs)
        self.max_distance = max_distance
        self.distance_embed = nnn.Embedding(num_embeddings=max_distance+1,
                                            embedding_dim=1)

    def process_input(self, sentence, seqlen_dim):
        embedded = super().process_input(sentence)
        other_dim = seqlen_dim + "2"
        other_embedded = embedded.rename(seqlen_dim, other_dim)

        embedded_mask = (embedded != self.padding_idx).float()
        embedded_mask = embedded_mask * \
            embedded_mask.rename(seqlen_dim, other_dim)

        distances = (
            (ntorch.arange(embedded.size(seqlen_dim), names=seqlen_dim) -
             ntorch.arange(embedded.size(seqlen_dim), names=other_dim))
            .abs().clamp(max=self.max_distance))
        d_mat = self.distance_embed(distances)[{'embedding': 0}]

        log_alignments = (
            embedded.dot("embedding", other_embedded)
            + d_mat + (1-embedded_mask) * (-1e3))

        embedded_attns = log_alignments.softmax(
            other_dim).dot(other_dim, other_embedded)
        return ntorch.cat([embedded, embedded_attns], "embedding")
