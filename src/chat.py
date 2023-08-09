import torch
import transformers
from model import N00bGPTLMHeadModel

def chat():
    tokenizer = transformers.GPT2Tokenizer.from_pretrained('gpt2')
    m_nb = N00bGPTLMHeadModel.from_hf_pretrained_weight()
    m_hf = transformers.GPT2LMHeadModel.from_pretrained('gpt2')

    while True:
        print('-' * 30)
        which_model = input("Use N00bGPT or Huggingface GPT2 (nb/hf)?").lower()
        input_str = input("Enter a starting text: ")
        gen_len = int(input("Enter sequence length: "))
        input_ids = tokenizer(input_str,return_tensors='pt')['input_ids']
        
        if not which_model or which_model == 'nb':
            model = m_nb
            output_ids = model.sample(input_ids, gen_len)[0]
        else:
            model = m_hf
            ids = input_ids
            for _ in range(gen_len):
                logits = model(ids)[0][:,-1,:]
                _, id = torch.topk(logits, 1)
                ids = torch.concat([ids,id],dim=1)
            output_ids = ids[0]

        output_str = tokenizer.decode(output_ids)
        print(f'Output: {output_str}')

if __name__ == "__main__":
    chat()


