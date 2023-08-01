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
#### Reading list
- 
- Self attention and Transformer: explanation of Attention is all you Need
http://jalammar.github.io/illustrated-transformer/ 
Attention mechanism in RNN
https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/ 
Word embedding
https://machinelearningmastery.com/what-are-word-embeddings/ 

asdf 
asdf asdf sadf 

asdf asdf 

Questions
How is one language translated to another, i.e. how is the output of encoder processed in decoder?
How is transformer, seq-to-seq model be used in other tasks, e.g. multiple choice, question answering, text summarization, etc
Look into how is hugging face version implemented?
How large is a transformer model? How much is dense and how much is sparse?
How is words embedding calculated?
What loss function does it use? How does benchmark work? 
How can PyTorch autograd compute the grad for a particular element or row in a metrics for embedding table?
