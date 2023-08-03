# n00bGPT (WIP)

This is a learning note of Transformer and LLM. I like learning by doing. I have some experience in Infra and little in ML, though barely little on LLM. Below is my learning jounery. And hope it helps yours.

1. **Kaggle competition**: fresh touch on LLM with Huggingface Transformer fine tune
2. **Attention is all you need**: understand how does transformer work
3. **NanoGPT**: a codelab to build your own Transformer model from scratch
4. **Transformer variant**: understand the landscope of transformers
5. **Training Infra**: study scalability by FSDP, GSPMD, JAX, etc

## 0.Prior Knowledge

You probably don't need to read all of them. But those are what I know before this jounery.
- [Neural Network and Deep Learning by Andrew Ng](https://www.coursera.org/learn/neural-networks-deep-learning)
- [PyTorch tutorial quickstart](https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html)
- [Kaggle Pandas and ML courses](https://www.kaggle.com/learn)
- [Programming Massively Parallel Processors: A Hands-on Approach](https://www.amazon.com/Programming-Massively-Parallel-Processors-Hands/dp/0124159923)


## 1. Kaggle competition

Goal: use it before understand it. Train and inference a transformer model by Huggingface

Study for study is boring. Study for competition is fun! [LLM Science Exam](https://www.kaggle.com/competitions/kaggle-llm-science-exam) is what we need. Little prior knowledge required. You can build and train you very first transformer model now! Get a fresh touch, and also obviously a low rank on leaderboard. Create questions in mind, study knowledges below, and make our rank higher!

[llm-notebook](https://github.com/fmars/n00bGPT/blob/main/llm-science-exam-s1.ipynb) is our first trial. The score wasn't high but that's ok. We got fresh touch on transformer, and played with it!

- [HuggingFace Datasets](https://huggingface.co/docs/datasets/index) 
  - The library to access and process the data, in a very easy way!
  - `Dataset::load_dataset()`, with split, config
  - `Dataset::index()` and `slicing()`, `IterableDataset::iter()` and `next()`
  - `Dataset::map()` to preprocess
- [HuggingFace Transformers](https://huggingface.co/docs/transformers/index)
  - The library to download, fine-tune and inference the pretrained models, in a even easier way!
  - [AutoTokenizer](https://github.com/huggingface/transformers/blob/v4.31.0/src/transformers/tokenization_utils_base.py#L1494)
    - `AutoTokenizer::from_pretrained() -> TokenizerBase`. A concrete derived class will be returned, which is compatible with corresponding model type
    - `Tokenizer::tokenize(text, text_pair, padding, truncation) -> {input_ids, token_type_ids, attention_masks, label}`. Inspect input and output values to understand more
  - [AutoModel](https://github.com/huggingface/transformers/blob/4033ea7167c4a826f895830bac04c2561680572c/src/transformers/models/auto/modeling_auto.py#L1170)
    - Inspect `AutoModelForMultipleChoice(AutoModel)::_model_mapping` to understand model factory
    - `AutoModelForMultipleChoice::from_pretrained()` to download and initialize model with concrete type (e.g. DebertaV2ForMultipleChoice)
    - DebertaV2ForMultipleChoice::forward(input) -> output. It's very interesting to see all field names in input and output are literal strings (from a infra engineer perspective)
    - `Input: dict<str, tensor>`: "Input_ids": [num_batch, num_choices, sequence_length], same for "token_type_ids" and "attention_mask"
    - `Collator` is used to convert data layout from dataset input to model input, essentially, change the first dimension from num batch to feature names.
  - [PreTrainedModel](https://github.com/huggingface/transformers/blob/v4.31.0/src/transformers/models/deberta_v2/modeling_deberta_v2.py#L917)
    - Each concrete model (e.g. DebertaV2, bert, llama) derives from base class `PreTrainedModel`, with task based variants (e.g. DebertaV2ForMaskedLM, DebertaV2ForQuestionAnswering, DebertaV2ForMultipleChoice)
    - A single .py file includes all implementation of a model, following [Repeat Yourself](https://discuss.huggingface.co/t/repeat-yourself-transformers-design-philosophy/16483) philosopy. Another very interesting thing (from a infra engineer perspective).
  - [HuggingFace Trainer](https://github.com/huggingface/transformers/blob/v4.31.0/src/transformers/trainer.py#L2968)
      - TODO, take a closer look to understand HF API design philosopy
  

### 1. Attention is all you need

Goal: understand major concepts of transformer by paper reading
#### Reading list
- [Atention is all you need](https://arxiv.org/abs/1706.03762)
  - The original proposal of transformer. Get some rough ideas. If you still don't understand how it actually works, that's okay. I didn't either. Following blogs provide intuitive explainations.
- [Attention mechnism explained by Jay Alammar](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/)
  - Not neccessarily need to understand RNN (I still don't), though it helps build up the intuition of attention
- [The Illustrated Transformer by Jay Alammar](http://jalammar.github.io/illustrated-transformer/)
  - Step-by-step visualized explanation of attention and transformer. Quite clear and detailed.
  - You should know how does transformer work at high level, after reading this. It's okay if some details still feel confusing. It'll all be resolved once we write our own transformer model
- [Embedding and word2vec](https://arxiv.org/abs/1301.3781)
  - If the concept of embedding sounds unfamiliar to you, this paper is for you.
  - Again, it's totally fine if you don't fully understand the details. We'll build our own word2vec as well

#### Fun Questions to Test Ourselves
- Does transformer incluse its own word embedding and trained together with attention layers? Or it takes embedding as input trained from somewhere else?
  - Paper doesn't mention this (likely a too detailed engineering question), but seems the answer is yes for most of transformer implementations I have seen.
- The model architecture seems to work for language translation. But how does that work for tasks, e.g. multiple choice, question answering, text summarization, etc?
  - Only using the model arch in paper itself won't. The task head comes into play. Attention/transformer can be thought of as a fundation that stores the universal general knowledges. A task specific head arch is added on top of it, to produce task specific output.
  - Take a read on Huggingface transformer implementation. The idea will be clear. We'll also get hands on experience on this when wring our own model, with a Language Model Head for text generation.
- How large is a transformer model? How much is dense and how much is sparse?
  - It seems attention layer is quite small, just some MLP layers. How can modern LLM models use tens of even hundreds of billions of parameters.
  - Parameter here refers to learnable parameters, i.e. the weights defined in init() method in torch.nn.module. Major parameters include
      - Attention QKV metrix: 3 * emb_dim * emb_dim
      - Attention projection layer: emb_dim * emb_dim
      - Feed forward: 4 * emb_dim * emb_dim + emb_dim * emb_dim
      - Word embedding: emb_dim * vocab_size
      - Position embedding: seq_len * emb_dim
  - Note that, it seems LLM folks usually call emb_dim as model_dim.
  - We also need to multiple the first 3 by the number of layers. Confirmed by a friend, who cannot reveal the actual number, this calculation is right and will get to hundreds of billions under production setup.
- How can PyTorch autograd compute the grad for a particular row in a embedding table?
  - Transformer (and all other use cases as well) doesn't use entire embedding table but some of rows during each training batch. So the gradient and weights update should only happen to those rows
  - TODO look into how does Pytorch implement such partial-tensor. I thought it's computed at tensor level.
- In text generation (e.g. ChatGPT) how does transformer know when to stop generating new words? 
  -   Transformer generates a new word at a time. Some API takes num_word_to_generate as input for inference. If it's not provided (which is the case in ChatGPT), how does it know when to stop.
  -   I suspect there is a special token, e.g. session end, etc. Generation stops when such token is generated.
- What is an autoregressive language model?
  - Autoregressive model in statistics, refers to the model that makes prediction for a time series data based on previous data point in the time series
  - Auto refers to self: model uses its own output from previous prediction as input for generating the next word
  - Regressive refers to looking backward: model use previous data point to predict next data point
- What is Beam search in LLM/sampling?
  - When generate the next word, instead of always use the highest probability one, it pick top k words and generate k sequences. 
  - In next step generates k sub-sequence for each sequence thus in totally k*k sequences
  - It then runs pruning and only keep top-k quality sequence
  - And repeat above steps
- What loss function does it use?
  - Different tasks use different loss functions
  - Cross entropy for language translation. Think of as a classification problem. Next work can be thought of as the label. And the output of the model is the probability of each word in the vocabulary. 



