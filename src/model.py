import torch
import math
import transformers
from dataclasses import dataclass

def softmax(x):
  # Note, without normalization it suffers from instability
  # in attention computation, causing either NaN or wrong result
  x = x - torch.max(x,axis=-1,keepdim=True)[0]
  exp = torch.exp(x)
  sum = torch.sum(exp, dim=-1, keepdim=True)
  return exp / sum

def scaled_dot_product_attention(
    q: torch.Tensor, # [N,...,L,E]
    k: torch.Tensor, # [N,...,L,E]
    v: torch.Tensor, # [N,...,L,E]
) -> torch.Tensor:
    t = torch.matmul(q, k.transpose(-1,-2))
    dk = torch.tensor(k.shape[-1], dtype=torch.float32)
    t = t / torch.sqrt(dk)
    t=softmax(t)
    return torch.matmul(t, v)

class MultiheadAttentionNoVectorization(torch.nn.Module):
  def __init__(self, emb_dim, n_head):
    super().__init__()
    self.emb_dim = emb_dim
    self.n_head = n_head
    self.head_dim = emb_dim // n_head
    assert self.head_dim * n_head == emb_dim
    self.qw = torch.randn(emb_dim, n_head, self.head_dim) # [emb_dim, n_head, head_dim]
    self.kw = torch.randn(emb_dim, n_head, self.head_dim) # [emb_dim, n_head, head_dim]
    self.vw = torch.randn(emb_dim, n_head, self.head_dim) # [emb_dim, n_head, head_dim]
    self.w = torch.randn(emb_dim, emb_dim)

  def forward(self, q,k,v):
    # q,k,v: [n_batch, seq_len, emb_dim]
    heads = []
    for i in range(self.n_head):
      qi = torch.matmul(q, self.qw[:,i,:]) # [n_batch, seq_len, head_dim]
      ki = torch.matmul(k, self.kw[:,i,:]) # [n_batch, seq_len, head_dim]
      vi = torch.matmul(v, self.vw[:,i,:]) # [n_batch, seq_len, head_dim]
      head = torch.nn.functional.scaled_dot_product_attention(qi, ki, vi) # [n_batch, seq_len, head_dim]
      heads.append(head)
    concated = torch.concat(heads, dim=-1) # [n_batch, seq_len, emb_dim]
    out = torch.matmul(concated, self.w) # [n_batch, seq_len, emb_dim]
    return out

class MultiheadAttention(torch.nn.Module):
  def __init__(self, emb_dim, n_head):
    super().__init__()
    self.emb_dim = emb_dim
    self.n_head = n_head
    self.head_dim = emb_dim // n_head
    assert self.head_dim * self.n_head == self.emb_dim
    self.bias = True

    self.attn = torch.nn.Linear(self.emb_dim, 3 * self.emb_dim, bias = self.bias)
    self.proj = torch.nn.Linear(self.emb_dim, self.emb_dim, bias=self.bias)

  def forward(self, x):
    # x: [n_batch, seq_len, emb_dim], assuming q=k=v=x in MultiheadAttentionSimple implementation
    n_batch, seq_len, emb_dim = x.shape
    # Vectorized/concated form of QW_q_i, KW_k_i, and VW_v_i
    attn = self.attn(x) # [n_batch, seq_len, 3*emb_dim]
    q,k,v = attn.split(self.emb_dim, dim=-1)
    # Reshape to per-head form
    q = q.view(n_batch, seq_len, self.n_head, self.head_dim).transpose(1,2)
    k = k.view(n_batch, seq_len, self.n_head, self.head_dim).transpose(1,2)
    v = v.view(n_batch, seq_len, self.n_head, self.head_dim).transpose(1,2)
    # Compute dot product attention, which input is [N,..., seq_len, emb_dim]
    y = torch.nn.functional.scaled_dot_product_attention(q,k,v, is_causal=True) # [n_batch, n_head, seq_len, head_dim]
    y = y.transpose(1,2).contiguous().view(n_batch, seq_len, emb_dim) # [n_batch, seq_len, emb_dim]
    y = self.proj(y)
    return y

class FeedForward(torch.nn.Module):
  def __init__(self, emb_dim, bias, dropout):
    super().__init__()
    self.context_proj_1 = torch.nn.Linear(emb_dim, 4 *emb_dim, bias=bias)
    # Note, torch.nn.GELU() and hf.gelu_new behaves differently, though
    # it doesn't seem so from wiki doc
    self.gelu = transformers.activations.ACT2FN['gelu_new']
    self.context_proj_2 = torch.nn.Linear(4*emb_dim, emb_dim, bias=bias)
    self.dropout = torch.nn.Dropout(dropout)

  def forward(self, x):
    x = self.context_proj_1(x)
    x = self.gelu(x)
    x = self.context_proj_2(x)
    x = self.dropout(x)
    return x

class Layer(torch.nn.Module):
  def __init__(self, emb_dim, n_head, bias, dropout):
    super().__init__()
    self.ln_1 = torch.nn.LayerNorm(emb_dim)
    self.attn = MultiheadAttention(emb_dim, n_head)
    self.ln_2 = torch.nn.LayerNorm(emb_dim)
    self.feed_fwd = FeedForward(emb_dim, bias, dropout)

  def forward(self, x):
    residual = x
    x = self.ln_1(x)
    x = self.attn(x)
    x = x + residual
    residual = x
    x = self.ln_2(x)
    x = self.feed_fwd(x)
    x = x + residual
    return x

from dataclasses import dataclass
@dataclass
class ModelConfig:
  vocab_size: int = 50304
  block_size: int = 1024
  n_layer: int = 7
  n_head: int = 4
  emb_dim: int = 64
  bias: bool = True
  dropout: float = 0.0
  use_torch_mhattention: bool = True

  def __init__(self, gpt_cfg):
    self.vocab_size = gpt_cfg.vocab_size
    self.block_size = gpt_cfg.n_positions
    self.n_layer = gpt_cfg.n_layer
    self.n_head = gpt_cfg.n_head
    self.emb_dim = gpt_cfg.n_embd
    self.bias = True
    self.droput = 0.0

class N00bGPTLMHeadModel(torch.nn.Module):
  def __init__(self, cfg: ModelConfig):
    super().__init__()
    self.cfg = cfg
    self.emb_dim = self.cfg.emb_dim

    self.token_emb = torch.nn.Embedding(self.cfg.vocab_size, self.emb_dim)
    self.pos_emb = torch.nn.Embedding(self.cfg.block_size, self.emb_dim)
    self.dropout = torch.nn.Dropout(self.cfg.dropout)
    self.layers = torch.nn.ModuleList([Layer(cfg.emb_dim, cfg.n_head, cfg.bias, cfg.dropout) for _ in range(self.cfg.n_layer)])
    self.ln = torch.nn.LayerNorm(self.emb_dim)

    self.lang_model_head = torch.nn.Linear(self.emb_dim, self.cfg.vocab_size, bias=False) # Do we need to explicitly disable bias here?

  def base_forward(self, x):
    n_batch, seq_len = x.shape
    assert seq_len <= self.cfg.block_size
    pos = torch.arange(0, seq_len)
    if torch.cuda.is_available():
      pos = pos.to(torch.device("cuda"))

    token_emb = self.token_emb(x) # [n_batch, seq_len, emb_dim]
    pos_emb = self.pos_emb(pos) # [seq_len, emb_dim]
    x = self.dropout(token_emb+pos_emb)
    for layer in self.layers:
      x = layer(x)
    x = self.ln(x)
    return x

  def forward(self, x, targets=None):
    x = self.base_forward(x)

    if targets: # Training
      logits = self.lm_head(x)
      loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
    else: # Inference
      logits = self.lang_model_head(x[:,[-1],:]) # Only need to compute lm_head for the last token
      loss = None
    return logits, loss

  def sample(self, ids, len):
    for _ in range(len):
      logits = self.forward(ids)[0][:,-1,:]
      _, id = torch.topk(logits, 1)
      ids = torch.concat([ids,id],dim=1)
    return ids

  @classmethod
  def from_hf_pretrained_weight(clas):
    cfg_hf = transformers.GPT2Config()
    m_hf = transformers.GPT2LMHeadModel(cfg_hf).from_pretrained('gpt2')
    cfg = ModelConfig(cfg_hf)
    m = N00bGPTLMHeadModel(cfg)

    def weight_copy(x,y,src,dst,trans):
      print('start weights copy')
      for a,b in zip(src, dst):
        print(f'{a} -> {b}')
        need_transpose = any([a.endswith(s) for s in trans])
        if need_transpose:
          y.state_dict()[b].copy_(x.state_dict()[a].T)
        else:
          y.state_dict()[b].copy_(x.state_dict()[a])

    src = [str(i) for i in m_hf.state_dict().keys()]
    dst = [str(i) for i in m.state_dict().keys()]
    trans=['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight','mlp.c_proj.weight']
    weight_copy(m_hf,m,src,dst,trans)
    return m

def example():
    tokenizer = transformers.AutoTokenizer.from_pretrained('gpt2')
    input = 'i am a software engineer and i like to'
    input_ids=tokenizer(input,return_tensors='pt')['input_ids']

    model = N00bGPTLMHeadModel.from_hf_pretrained_weight()

    output_ids=model.sample(input_ids,20)[0]
    output = tokenizer.decode(output_ids)
    print(f'input: {input}')
    print(f'output: {output}')

if __name__ == "__main__":
    example()
