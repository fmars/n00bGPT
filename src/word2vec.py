#!/usr/bin/python3

import datasets
import transformers
import re
import torch
from tqdm import tqdm
from time import time

window_size = 2
negative_sample_size = 4
batch_size = 512
emb_dim = 128
epochs = 50

d = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

from collections import defaultdict
import random
class Dataset:
  def __init__(self, data, window_size, negative_sample_size, batch_size):
    self.lines = [line.lower() for line in data.split('\n') if line]
    self.lines = [re.sub(r'[^a-z ]','',line) for line in self.lines]
    self.lines = [line.split(' ') for line in self.lines]

    self.t2i = {}
    self.i2t = {}
    self.count = defaultdict(lambda: 0)
    for line in self.lines:
      for token in line:
        self.count[token] += 1
        if token not in self.t2i:
          self.t2i[token] = len(self.t2i)
    self.i2t = {i:t for t,i in self.t2i.items()}
    self.vocab_size = len(self.t2i)
    self.window_size = window_size
    self.negative_sample_size = negative_sample_size
    self.batch_size = batch_size

  def encode(self, tokens):
    if not isinstance(tokens, list):
      return self.t2i[tokens]
    else:
      return [self.t2i[token] for token in tokens]

  def decode(self, ids):
    if not isinstance(ids, list):
      return self.i2t[ids]
    else:
      return [self.i2t[id] for id in ids]

  def stats(self):
    print(f'Total lines: {len(self.lines)}')
    print(f'Total tokens: {sum([len(line) for line in self.lines])}')
    print(f'Avg line length: {sum([len(line) for line in self.lines]) / len(self.lines)}')
    print(f'Vocab size: {self.vocab_size}')

  def generate_n_random_not_x(self, n, x):
    result = []
    while len(result) < n:
        random_num = random.randint(0,self.vocab_size-1) # randint() returns range(a,b), both sides included
        if random_num != x and random_num not in result:
            result.append(random_num)
    return result

  def gen_training_example(self, line_no):
    word = []
    context = []
    label = []
    line = self.lines[line_no]
    for i, token in enumerate(line):
      positives = line[max(0,i-self.window_size): i] + line[i+1:min(len(line), i+1+self.window_size)]
      positives = self.encode(positives)
      for positive in positives:
        negatives = self.generate_n_random_not_x(self.negative_sample_size, positive)
        word.append(self.encode(token))
        context.append([positive] + negatives)
        label.append([1] + [0] * self.negative_sample_size)
    return word, context, label

  def generator(self):
    buffer_size = 0
    word = []
    context = []
    label = []
    prefetch = 100
    line_ptr = 0
    while True:
      if buffer_size >= self.batch_size:
        o_word = word[:self.batch_size]
        word = word[self.batch_size:]
        o_context = context[:self.batch_size]
        context = context[self.batch_size:]
        o_label = label[:self.batch_size]
        label = label[self.batch_size:]
        buffer_size -= self.batch_size
        t_w, t_c, t_l = torch.tensor(o_word), torch.tensor(o_context), torch.tensor(o_label)
        t_w = t_w.to(d)
        t_c = t_c.to(d)
        t_l = t_l.to(d)
        yield t_w, t_c, t_l
        continue
      if line_ptr >= len(self.lines):
        break
      buffer_size_before = buffer_size
      for i in range(min(prefetch, len(self.lines) - line_ptr)):
        assert line_ptr < len(self.lines)
        w,c,l = self.gen_training_example(line_ptr)
        word.extend(w)
        context.extend(c)
        label.extend(l)
        buffer_size = len(word)
        line_ptr += 1
      # print(f'Consuing to {line_ptr} and gen {buffer_size - buffer_size_before} examples')

data = datasets.load_dataset('tiny_shakespeare')['train']
data = next(iter(data))['text']
ds = Dataset(data, window_size, negative_sample_size, batch_size)

ds.stats()
print(f'Sentence: {ds.lines[1]}')
word,context,label=ds.gen_training_example(1)
print('Target -> context')
for w,c in zip(word[:5], context[:5]):
  print(f'{ds.decode(w)} -> {ds.decode(c)}')

class Word2VecSkipGram(torch.nn.Module):
  def __init__(self, vocab_size, emb_dim):
    super().__init__()
    self.tgt_emb = torch.nn.Embedding(
        num_embeddings=vocab_size,
        embedding_dim=emb_dim
    )
    self.ctx_emb = torch.nn.Embedding(
        num_embeddings=vocab_size,
        embedding_dim=emb_dim
    )
  def forward(self, target, context):
    # target [batch_size]
    # context [batch_size, 1+negative_sample_size]
    tgt_emb = self.tgt_emb(target)
    cxt_emb = self.ctx_emb(context)
    dots = torch.einsum('be,bce->bc', tgt_emb, cxt_emb)
    return dots
    
    
import tensorflow as tf
writer = tf.summary.create_file_writer('./logs/word2vec')

model = Word2VecSkipGram(ds.vocab_size, emb_dim)
model = model.to(d)
model.train()
opt = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9,0.999),eps=1e-7)
loss_fn = torch.nn.CrossEntropyLoss()

for i in range(epochs):
  dl = ds.generator()
  t1 = time()
  print(f'Epoch {i}/{epochs}')
  with tqdm(total=len(ds.lines)) as pbar:
    line_no = 0 
    n_example = 0
    for target, context, label in dl:
      logits = model(target, context)
      loss = loss_fn(logits, label.double())
      loss.backward()
      opt.step()
      opt.zero_grad()

      new_line_no = dl.gi_frame.f_locals['line_ptr']
      pbar.update(new_line_no - line_no)
      line_no = new_line_no
      n_example += target.shape[0]
      t2 = time()
      qps = n_example/(t2-t1)
      pbar.set_postfix_str(f'loss: {loss.item()}, qps: {qps}')
  with writer.as_default():
    tf.summary.scalar('loss', loss.item(), step=i)
    tf.summary.scalar('qps', qps, step=i)

