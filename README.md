# n00bGPT (WIP)

This is my 3-weeks learning jounery of Transformer and LLM. It's challenging. Because I initially knew nothing about it. However, it's exhilarating and rewarding. I like learning by doing. You guess right. We named it n00bGPT since we built our own transformer. I hope this note help yours!

1. **Kaggle competition**: fresh touch on LLM with Huggingface Transformer fine tune
2. **Attention is all you need**: understand how does transformer work
3. **NanoGPT**: a codelab to build your own Transformer model from scratch
4. **Transformer variant**: understand the landscope of transformers
5. **Training Infra**: study scalability by FSDP, GSPMD, JAX, etc

## 0.Prior Knowledge

You probably don't need to read all of them, but those are the things I knew before this learning journey.
- [Neural Network and Deep Learning by Andrew Ng](https://www.coursera.org/learn/neural-networks-deep-learning)
- [PyTorch tutorial quickstart](https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html)
- [Kaggle Pandas and ML courses](https://www.kaggle.com/learn)
- [Programming Massively Parallel Processors: A Hands-on Approach](https://www.amazon.com/Programming-Massively-Parallel-Processors-Hands/dp/0124159923)


## 1. Kaggle competition

Goal: Applying before comprehending. Train and infer with a Huggingface Transformer model.

Learning for learning's sake is tedious. Competing in [LLM Science Exam](https://www.kaggle.com/competitions/kaggle-llm-science-exam) is exhilarating. Minimal prior knowledge needed. Build and train your first transformer model now! Gain a fresh experience, and obviously a low rank on leaderboard. Then, formulate questions, explore knowledges, and elavate our rank!

[llm-notebook](https://github.com/fmars/n00bGPT/blob/main/llm-science-exam-s1.ipynb) is our first trial. The score wasn't high but that's ok.

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

Goal: Grasping the key concepts of Transformers through paper reading.

#### Reading list
- [Atention is all you need](https://arxiv.org/abs/1706.03762)
  - Start by reading the original paper. Gather initial insights, even if you don't fully comprehend its inner working. That's fine. I didn't either. Below blogs offer intuitive explanations.
- [Attention mechnism explained by Jay Alammar](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/)
  - Not neccessarily need to understand RNN (I still don't), though it helps build up the intuition of attention
- [The Illustrated Transformer by Jay Alammar](http://jalammar.github.io/illustrated-transformer/)
  - Step-by-step visualized explanation of attention and transformer. Quite clear and detailed.
  - After reading this, you should have a high-level understanding of how the Transformer works. Don't worry if some details still seem confusing; they will become clearer as we proceed to write our own transformer model.
- [Embedding and word2vec](https://arxiv.org/abs/1301.3781)
  - If the concept of embedding sounds unfamiliar to you, this paper is for you.

#### Fun Questions
- Does transformer incluse its own word embedding and trained together with attention layers? Or it takes embedding as input trained from somewhere else?
  - The paper doesn't explicitly mention this detailed engineering question, but most Transformer implementations I've seen include their own word embeddings and train them along with attention layers.

- The model architecture seems to work for language translation. But how does that work for tasks, e.g. multiple choice, question answering, text summarization, etc?
  - The Transformer architecture alone isn't sufficient for these tasks; a task-specific head is added on top of the universal knowledge stored in the attention/transformer layers. This task head is responsible for producing task-specific outputs.
  - For instance, check out the Huggingface Transformer implementation to understand this concept better. When we build our own model, we will also gain hands-on experience with the Language Model Head for text generation.
  - How large is LLM? It seems attention layer is small. How can modern LLM models use tens of even hundreds of billions of parameters?
  - Parameter here refers to learnable parameters, i.e. the weights defined in init() method in torch.nn.module. Major parameters include
      - Attention QKV metrix: 3 * emb_dim * emb_dim
      - Attention projection layer: emb_dim * emb_dim
      - Feed forward: 4 * emb_dim * emb_dim + emb_dim * emb_dim
      - Word embedding: emb_dim * vocab_size
      - Position embedding: seq_len * emb_dim
  - Note that, it seems LLM folks usually call emb_dim as model_dim.
  - We also need to multiple the first 3 by the number of layers. Confirmed by a friend, who cannot reveal the actual number, this calculation is right and will get to hundreds of billions under production setup.
- How can PyTorch autograd compute gradients for specific rows in an embedding table?
  - Transformers (and other use cases) don't utilize the entire embedding table but only specific rows during each training batch. So the gradient and weights update should only happen to those rows
  - TODO look into how does Pytorch implement such partial-tensor. I thought it's computed at tensor level.
- In text generation (e.g. ChatGPT) how does transformer know when to stop generating new words?
  - The Transformer generates words one at a time. Some APIs take "num_words_to_generate" as input for inference.
  - My guess is, in the case of ChatGPT, where this input is not provided, a special token, like an "end of sentence" token, signals the model to stop generating new words.
- What is an autoregressive language model?
  - In statistics, an autoregressive model predicts a time series data point based on previous data points in the same series.
  - In language models, "auto" refers to self, meaning the model uses its own output from previous predictions as input to generate the next word.
  - "Regressive" indicates looking backward, as the model uses previous data points to predict the next data point (word).

- What is Beam search in LLM/sampling?
  - In Beam search, instead of always selecting the word with the highest probability, the model picks the top k words and generates k sequences in parallel.
  - In the next step, k sub-sequences are generated for each sequence, resulting in k * k sequences in total.
  - The model then applies pruning and keeps only the top-k quality sequences.
  - This process is repeated until the desired length or condition is met.

- What loss function does it use?
  - Different tasks in Transformers use different loss functions.
  - For language translation, it uses cross-entropy, treating it as a classification problem where the next word is the label, and the model output is the probability distribution over the vocabulary.



