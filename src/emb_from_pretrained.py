import transformers
import torch

tokenizer = transformers.AutoTokenizer.from_pretrained('bert-base-uncased')
model = transformers.AutoModel.from_pretrained('bert-base-uncased')
emb = model.state_dict()['embeddings.word_embeddings.weight']
print(f'Load embedding: {emb.shape}')

def get_emb(w):
  id = tokenizer.encode(w)[1]
  return emb[id]
  
king = get_emb('king')
man = get_emb('man')
woman = get_emb('woman')
x = king - man + woman

similarity = torch.matmul(emb, x)
probs, ids = torch.topk(similarity, 10)

for prob, id in zip(probs, ids):
   print(f'{tokenizer.decode(id)} -> {prob}')