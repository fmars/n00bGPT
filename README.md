# n00bGPT

Build GPT model from scratch. Learn perf optimiation and practice in the competition. How exhilarating and rewarding. It's named N00bGPT because I barely know nothing about LLM and wanted to how much I can get in 3 weeks. This is my jounery.

1. **Kaggle competition**: fresh touch on LLM with Huggingface Transformer fine tune
2. **Attention is all you need**: understand how does transformer work
3. **N00bGPT**: write your own Transformer model from scratch
4. **Transformer variants**: understand the landscope of transformers
5. **Performance**: study scalability w/ FSDP, GSPMD, JAX, etc

```mermaid
%%{ "width": 200, "height": 100 }%%

flowchart TB;
    A([Kaggle Competition])
    B([Attention is all you need])
    C([NoobGPT])
    D([Transformer Variants])
    E([Performance])
    A -. first touch .-> B
    B -. learn concepts .-> C
    C -. hands dirty .->  D
    D -. know landscpe .-> E
    E -. iterate on .-> A
```
## 0. Prior Knowledge

You probably don't need to read all of them, but those are the things I knew before this learning journey.
- [Neural Network and Deep Learning by Andrew Ng](https://www.coursera.org/learn/neural-networks-deep-learning)
- [PyTorch tutorial quickstart](https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html)
- [Kaggle Pandas and ML courses](https://www.kaggle.com/learn)
- [Programming Massively Parallel Processors: A Hands-on Approach](https://www.amazon.com/Programming-Massively-Parallel-Processors-Hands/dp/0124159923)



## 1. Kaggle competition

Goal: Applying before comprehending. Train and infer with a Huggingface Transformer model.

Learning for learning's sake is tedious. Competing in [LLM Science Exam](https://www.kaggle.com/competitions/kaggle-llm-science-exam) is exhilarating. Minimal prior knowledge needed. Build and train your first transformer model now! Gain a fresh experience, and obviously a low rank on leaderboard. Then, formulate questions, explore knowledges, and elavate our rank!

[llm-notebook](https://github.com/fmars/n00bGPT/blob/main/colab/llm-science-exam-s1.ipynb) is our first trial. The score wasn't high but that's ok.

Checkout class diagram of [huggingface major abstractions](https://github.com/fmars/n00bGPT/blob/main/huggingface_class_diagram.md).
- [HuggingFace Datasets](https://huggingface.co/docs/datasets/index) 
  - The library to access and process the data, in a very easy way!
  - `Dataset::load_dataset()`, with split, config
  - `Dataset::index()` and `slicing()`, `IterableDataset::iter()` and `next()`
  - `Dataset::map()` to preprocess
- [HuggingFace Transformers](https://huggingface.co/docs/transformers/index)
  - The library to download, fine-tune and inference the pretrained models, in a even easier way!
  - [AutoTokenizer](https://github.com/huggingface/transformers/blob/v4.31.0/src/transformers/tokenization_utils_base.py#L1494)
    - `AutoTokenizer::from_pretrained() -> TokenizerBase`. Return a concrete tokenizer class.
    - `Tokenizer::tokenize(text, text_pair, padding, truncation) -> {input_ids, token_type_ids, attention_masks, label}`
  - [AutoModel](https://github.com/huggingface/transformers/blob/4033ea7167c4a826f895830bac04c2561680572c/src/transformers/models/auto/modeling_auto.py#L1170)
    - Inspect `AutoModelForMultipleChoice(AutoModel)::_model_mapping` to understand model factory
    - `AutoModelForMultipleChoice::from_pretrained()` download model in concrete class type
    - `DebertaV2ForMultipleChoice::forward()` inspect input and output data formats 
    - `Collator`, essentially, changes the first dimension from num batch to feature names
  - [PreTrainedModel](https://github.com/huggingface/transformers/blob/v4.31.0/src/transformers/models/deberta_v2/modeling_deberta_v2.py#L917)
    - Each concrete model (e.g. DebertaV2, bert, llama) derives from base class `PreTrainedModel`, with task based variants (e.g. DebertaV2ForMaskedLM, DebertaV2ForQuestionAnswering, DebertaV2ForMultipleChoice)
    - A single .py file includes all implementation of a model, following [Repeat Yourself](https://discuss.huggingface.co/t/repeat-yourself-transformers-design-philosophy/16483) philosopy
  - [HuggingFace Trainer](https://github.com/huggingface/transformers/blob/v4.31.0/src/transformers/trainer.py#L2968)
      - TODO, take a closer look to understand HF API design philosopy

## 2. Attention is all you need

Goal: Grasping the key concepts of Transformers through paper reading.

<p align="center">
    <img title="ABC" src="https://github.com/fmars/n00bGPT/blob/main/images/aiayn.png" title="asdfasdf asdf" width="300" height="390">
    <em> From “Attention is all you need” paper by Vaswani, et al., 2017 </em>
</p>

        
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

<details><summary>Does transformer incluse own embedding and train together with attention layers?</summary>
    <ul>
<li> The paper doesn't explicitly mention this detailed engineering question, but most Transformer implementations I've seen include their own word embeddings and train them along with attention layers.</li>
    </ul>
</details>


<details><summary> How does model work for tasks besides language translation e.g. multiple choice, question answering, text summarization, etc?</summary>
    <ul>
<li>The Transformer architecture alone isn't sufficient for these tasks; a task-specific head is added on top of the universal knowledge stored in the attention/transformer layers. This task head is responsible for producing task-specific outputs.</li>
<li>For instance, check out the Huggingface Transformer implementation to understand this concept better. When we build our own model, we will also gain hands-on experience with the Language Model Head for text generation.</li>
    </ul>
</details>

<details><summary>How can LLM be that large, e.g. 300B parameters?</summary>
        <ul>
<li>Parameter here refers to learnable parameters, i.e. the weights defined in init() method in torch.nn.module. Major parameters include</li>
     <li> Attention QKV metrix: 3 * emb_dim * emb_dim</li>
     <li> Attention projection layer: emb_dim * emb_dim</li>
    <li>  Feed forward: 4 * emb_dim * emb_dim + emb_dim * emb_dim</li>
    <li>  Word embedding: emb_dim * vocab_size</li>
     <li> Position embedding: seq_len * emb_dim</li>
<li>Note that, it seems LLM folks usually call emb_dim as model_dim.</li>
<li>We also need to multiple the first 3 by the number of layers. Confirmed by a friend, who cannot reveal the actual number, this calculation is right and will get to hundreds of billions under production setup.</li>
            </ul></details>

<details><summary><b>TODO How does PyTorch autograd compute gradients for individual rows in an embedding table?</b></summary><ul>
<li>Transformers (and other use cases) don't utilize the entire embedding table but only specific rows during each training batch. So the gradient and weights update should only happen to those rows</li>
<li><p style="color:red;">TODO look into how does Pytorch implement such partial-tensor. I thought it's computed at tensor level.</p></li>
</ul></details>

<details><summary><b>TODO In text generation (e.g. ChatGPT) how does transformer know when to stop generating new words?</b></summary><ul>
<li>The Transformer generates words one at a time. Some APIs take "num_words_to_generate" as input for inference.</li>
<li>My guess is, in the case of ChatGPT, where this input is not provided, a special token, like an "end of sentence" token, signals the model to stop generating new words.</li>
</ul></details>

<details><summary>What is an autoregressive language model?</summary><ul>
<li>In statistics, an autoregressive model predicts a time series data point based on previous data points in the same series.</li>
<li>In language models, "auto" refers to self, meaning the model uses its own output from previous predictions as input to generate the next word.</li>
<li>"Regressive" indicates looking backward, as the model uses previous data points to predict the next data point (word).</li>
</ul></details>

<details><summary>What is Beam search in LLM/sampling?</summary><ul>
<li>In Beam search, instead of always selecting the word with the highest probability, the model picks the top k words and generates k sequences in parallel.</li>
<li>In the next step, k sub-sequences are generated for each sequence, resulting in k * k sequences in total.</li>
<li>The model then applies pruning and keeps only the top-k quality sequences.</li>
<li>This process is repeated until the desired length or condition is met.</li>
</ul></details>

<details><summary>What loss function does it use?</summary><ul>
<li>Different tasks in Transformers use different loss functions.</li>
<li>For language translation, it uses cross-entropy, treating it as a classification problem where the next word is the label, and the model output is the probability distribution over the vocabulary.</li>
</ul></details>


## 3. N00bGPT
Now, it's time to get hands dirty. Let's first build a word2vec to train our word embedding, and then build our simple gpt model.

<p align="center">
    <img title="Hands-on Transformer" src="https://github.com/fmars/n00bGPT/blob/main/images/build_transformer.jpg" title="" width="300" height="390">
</p>

### Dev environment - Colab
If don't have a dev environment set up yet, I strongly recommand to start with [Colab](https://colab.research.google.com/), easy to run, easy to manage dependency, easy to inspect result, easy to switch between hardwards, etc. In short, it's so easy (of course for learning use cases)! Not sure what happened to others but I still remember the afeternoon when I was trying to install cuda driver for my windows desktop, and eventually gave up after hours trying but still got torch.cuda.is_available==False. :(

 
### Dev environment - GCP VM

I figured GCP (Google Cloud Platform) is an alternative. It feels more of a serious, hmm, at least you get the control of (kind of) a real machine, which is technically speaking a VM though. Don't forget to install dependencies. 
```
pip3 install torch
pip3 install transformers
pip3 install datasets
pip3 install tensorflow
```
(If `pip3 install torch` failed with an error like ` 619.9 MB 2.6 MB/s eta 0:00:01Killed`, congrats, we're all frugal people and only applied a small vm. This is due to out of ram. Run instead `pip3 install torch --no-cache-dir`).

Find all of the colab/notebooks under [n00bGPT/colab/](https://github.com/fmars/n00bGPT/tree/main/colab), and plain python code under [n00bGPT/src](https://github.com/fmars/n00bGPT/tree/main/src)


### Word2vec
Ever wonder how is word embedding generated? Word2vec is the algorithm for it. Read the original paper [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/pdf/1301.3781.pdf) to understand how do CBOW (continuous bag-of-words) and Skip-Gram work. This is a [pytorch version word2vec](https://github.com/fmars/n00bGPT/blob/main/colab/word2vec_skip_gram.ipynb) to re-write [tensorflow version](https://www.tensorflow.org/text/tutorials/word2vec). The implementation is quite straightforward. It contains major pieces of a training program.

Let's look into the code [word2vec](https://github.com/fmars/n00bGPT/blob/main/colab/word2vec_skip_gram.ipynb):
- [Dataset](https://github.com/fmars/n00bGPT/blob/main/src/word2vec.py#L20): poor man's version of tokenizer with encode and decode, iteratable interface, and also a collator
- [Skip-Gram model](https://github.com/fmars/n00bGPT/blob/main/src/word2vec.py#L127): use dot-product to measure similarity(distance) and use CrossEntropy to measure loss
- [Trainer](https://github.com/fmars/n00bGPT/blob/main/src/word2vec.py#L155): poor man's trainer, with minibatch multi-epoch training, a cool progress bar, and also tensorboard logging
- [Optimizer](https://github.com/fmars/n00bGPT/blob/main/src/word2vec.py#L153): understand SGD, momentum, adagrad, and adam. Searching around I found [this blog](https://www.analyticsvidhya.com/blog/2021/10/a-comprehensive-guide-on-deep-learning-optimizers/) gives quite a good explanation of intuition. Though I figurted ChatGPT gives me the best explanation of how exactly does each of them work.

 <p align="left">
    <img title="word2vec tensorboard" src="https://github.com/fmars/n00bGPT/blob/main/images/word2vec%20_tb.png" title="" width="450" height="150">
</p>

### King-Man+Woman=Queen?
Our poorman's word2vec seems working well. The loss curve converges in a descent way. However, if you try the classic test, king-man+woman actually works very poorly (which is kind of expected otherwise why they needs hundreds or thunsands gpus to cotrain :)

Now, a fun quiz: write your embedding to load from pre-trained LLM model, and predict what does king-man+woman result. Give it a try before looking into [this](https://github.com/fmars/n00bGPT/blob/main/src/emb_from_pretrained.py).


## N00bGPT

<p align="center">
    <img title="weights mapping" src="https://github.com/fmars/n00bGPT/blob/main/images/chat.png" title="" width="1000" height="200">
</p>

[mode.py](https://github.com/fmars/n00bGPT/blob/main/src/model.py) and [chat.py](https://github.com/fmars/n00bGPT/blob/main/src/chat.py) is our own version of Transformer implementation from scratch, including 
- softmax()
- scaled_dot_product_attention()
- MultiheadAttention
- GPTModel
- GPTLMHeadModel
 


We load weights from Huggingface pretrained GPT-2, and run text generation. I found it’s not hard to write the initial version, however the generated text doesn’t make sense at all. The challenging part is to make our model correct. First time debugging the numerical issue. I found  it really fun, and of course time consuming :)

**Step 1: How does Huggingface GPT work**

Let’s first look into what happens step by step when we run 
```
from transformers import pipeline
generator = pipeline('text-generation', model='gpt2')
input = ‘i am a software engineer and i like to'
generator("input, max_length=30, num_return_sequences=5)
```

1. A [TextGenerationPipeline](https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/text_generation.py#L24C7-L24C29
) is created Which is a derived class of [Pipeline](https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/base.py#L1066)
2. [call](https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/base.py#L1129) method of Pipeline essentially calls derived preprocess(), forward(), postprocess()
3. [TextGenerationPipeline::_forward()](https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/text_generation.py#L265 )  calls model.generate(), where model is [GPT2LMHeadModel](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py#L955)
4. PretrainedModel derives from GenerationMixin, which defines [generate() method](https://github.com/huggingface/transformers/blob/080a97119c0dabfd0fb5c3e26a872ad2958e4f77/src/transformers/generation/utils.py#L1249)
5. generate() method first generates [genearation_config](https://github.com/huggingface/transformers/blob/080a97119c0dabfd0fb5c3e26a872ad2958e4f77/src/transformers/generation/utils.py#L1633 ), which has a field called generation_mode, which is sample in our case 
6. [sample() method](https://github.com/huggingface/transformers/blob/080a97119c0dabfd0fb5c3e26a872ad2958e4f77/src/transformers/generation/utils.py#L2740) set up some configs, then runs into a while true loop, to generate token one at a time, and stops when stop criterion meets
7. sample() method calls actual model (GPT2LMHeadMode), which has LM head as a MLP, and invokes [forward() method](https://github.com/huggingface/transformers/blob/080a97119c0dabfd0fb5c3e26a872ad2958e4f77/src/transformers/generation/utils.py#L2755)
8. GPT2LMHHeadModel’s forward contains a [LM head](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py#L1098 ) on top of basic transformer


**Step 2: Fun questions**

- How does stop criterion work?
    - Generates a [stop criteria list](https://github.com/huggingface/transformers/blob/080a97119c0dabfd0fb5c3e26a872ad2958e4f77/src/transformers/generation/utils.py#L1014), which includes a default list, stop based on max length, and max time 
    - It seems huggingface currently only uses the most [basic criterions](https://github.com/huggingface/transformers/blob/main/src/transformers/generation/stopping_criteria.py#L36 ) (hmm thought some smart solution is used)
- How does sample actually work
    - Sample picks the next token based on multinomial distribution over all possible tokens, which provides some randomness, whereas higher probability token has higher probability to be picked 
    - It generates one token at a time, through step above, and repeats until stop criterion satisfied
    - Other generation mode exist, e.g. beam search, greedy search, etc
- What’s the input and output format of base model
    - Output is [BaseModelOutputWithPastAndCrossAttentions](https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_outputs.py#L245 )
- What’s the input and output format of LMHead model
    - Output is [CausalLMOutputWithCrossAttentions ](https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_outputs.py#L623 )

**Step 3: Inspect and verify each module/method**

1. GPT2Attention <> MultiheadAttention
2. GPT2MLP <> FeedForward
3. GPT2Block <> Layer
4. GPT2Model <> GPTLMHeadModel

Also inspect the correctness of weights mapping.
 <p align="left">
    <img title="weights mapping" src="https://github.com/fmars/n00bGPT/blob/main/images/weights_mapping.png" title="" width="1500" height="500">
</p>

[Parity debugging](https://github.com/fmars/n00bGPT/blob/main/colab/model_parity_debugging.ipynb) are unit tests that we compare Huggingface and our version layer by layer. Below are bugs found in the initial implementation 
- MultiheadAttention
    - Huggingface uses dropout in both attention and residual network
    - (Actually this should have no effect since dropout is disabled automatically in eval mode)
    - Forgot to implement the attention mask as we’re computing causal attention. Technically it shouldn’t affect final output (i.e. generated text), it affects logits of intermediate hidden state
        - In torch.nn.functional.scaled_dot_product(), it’s is_causal=True parameter
    - In the attention layer, both 'in_proj_weight', 'out_proj.weight' require transpose from pretrained weights
- MLP <> FeedForward
    - torch gelu runs different than GPT2 gelu_new, though their documentation states the same 
- Block <> Layer
    - When adding residual to the attention output, residual should equal to original input (i.e. hidden state) rather than linear norm output 


After fixing all of those bugs, our n00bGPT works and finally generates some reasonable text!


## 4. Transformer Variants
- [Transformer model family](https://huggingface.co/docs/transformers/model_summary) gives a good summary
- [LLAMA2](https://ai.meta.com/research/publications/llama-2-open-foundation-and-fine-tuned-chat-models/), Meta's 7B,14B,70B model with 4k sequence length
- [DeBARTa](https://huggingface.co/docs/transformers/v4.31.0/en/model_doc/deberta#overview) Microsoft optimized transformer arch
- [BERT](https://huggingface.co/docs/transformers/v4.31.0/en/model_doc/bert#overview ), L: 12, H:1024, A:16. Parameters: 340M
- [PALM2](https://arxiv.org/abs/2305.10403) Google's latest pretrained model
- [BigBird](https://huggingface.co/docs/transformers/model_doc/big_bird), a sparse attention mechanism that reduces quadratic dependency on the sequence length to linear.
- [Fast transformer](https://arxiv.org/abs/1911.02150), optimized attention algorithm to reduce memory pressure on inference


## 5. Performance

#### Single card GPU
Follow [single gpu perf opt](https://huggingface.co/docs/transformers/perf_train_gpu_one) to try out basics techniques. We’ll use our transformer model to measure training QPS, and see [notebook](https://github.com/fmars/n00bGPT/blob/main/colab/single_card_perf_opt.ipynb) for details. Baseline setting is
```
batch_size=36
seq_len=128
emb_dim=768
vocab_size=50257
n_layer=12
optimizer=adam
```
| Desc        | Change       | QPS@T4     |  QPS@V100 |
| :---        |    :----:   |    :----:   |      ---: |
| Baseline      |        |  30 | 108 |
|Torch Dynamo + Inductor | `opt_model = torch.compile(model, backend="inductor")` | 30 | 117 |
| | batch_size=48 | | 117 |
| | batch_size=16 | | 107 |
| Quantization: DMP fp16 | `with torch.cuda.amp.autocast()` | 115 | 344 |
| Tensor Core with TF32 | `torch.backends.cuda.matmul.allow_tf32 = True` | | 340 | 
| HG accelerator | `m,opt = accelerator.prepare(m,opt)` | | 115 |


#### GPU Memory
For our transformer, number of parameter is 
- Embedding table: vocab_size * emb_dim * 2
- Attention: n_layer * (emb_dim ^ 2) * 9 # mh attention + feed forward
- LM head: emb_dim * vocab_size

Model weights = num_parameters * 4 
Optimizer state = num_parameters * 8 (adam for momentum)
Gradient = num_parameters * 4

[Notebook](https://github.com/fmars/n00bGPT/blob/main/colab/mem_usage.ipynb). Follow simple example program, inspect reserved and allocated memory during training. 
```python
class Model(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.a = torch.nn.Parameter(torch.randn(1024*1024*1024//4).to(dev))
    self.b = torch.nn.Parameter(torch.randn(1024*1024*1024//4).to(dev))

  def forward(self):
    c = self.a + self.b
    d = self.a - self.b
    e = self.a * self.b
    f = c + d
    return f

m = Model()
opt = torch.optim.Adam(m.parameters())
for i in range(5)
  preds = m()
  loss = torch.mean(preds)
  loss.backward()
  opt.step()
  opt.zero_grad()

before creating model    : GPU memory (Torch) total: 14.7, reserved: 0.0, allocated: 0.0, free: 14.7
after creating model     : GPU memory (Torch) total: 14.7, reserved: 2.0, allocated: 2.0, free: 12.7
after creating opt       : GPU memory (Torch) total: 14.7, reserved: 2.0, allocated: 2.0, free: 12.7
iteration 0
inside foward 1          : GPU memory (Torch) total: 14.7, reserved: 3.0, allocated: 3.0, free: 11.7
inside foward 2          : GPU memory (Torch) total: 14.7, reserved: 4.0, allocated: 4.0, free: 10.7
inside foward 3          : GPU memory (Torch) total: 14.7, reserved: 5.0, allocated: 5.0, free: 9.7
inside foward4           : GPU memory (Torch) total: 14.7, reserved: 6.0, allocated: 6.0, free: 8.7
after forward            : GPU memory (Torch) total: 14.7, reserved: 6.0, allocated: 3.0, free: 11.7
after loss               : GPU memory (Torch) total: 14.7, reserved: 6.0, allocated: 3.0, free: 11.7
after loss.backward      : GPU memory (Torch) total: 14.7, reserved: 6.0, allocated: 5.0, free: 9.7
after opt.step           : GPU memory (Torch) total: 14.7, reserved: 13.0, allocated: 9.0, free: 5.7
after opt.zero_grad      : GPU memory (Torch) total: 14.7, reserved: 13.0, allocated: 7.0, free: 7.7
iteration 1
inside foward 1          : GPU memory (Torch) total: 14.7, reserved: 13.0, allocated: 8.0, free: 6.7
inside foward 2          : GPU memory (Torch) total: 14.7, reserved: 13.0, allocated: 9.0, free: 5.7
inside foward 3          : GPU memory (Torch) total: 14.7, reserved: 13.0, allocated: 10.0, free: 4.7
inside foward4           : GPU memory (Torch) total: 14.7, reserved: 13.0, allocated: 11.0, free: 3.7
after forward            : GPU memory (Torch) total: 14.7, reserved: 13.0, allocated: 8.0, free: 6.7
after loss               : GPU memory (Torch) total: 14.7, reserved: 13.0, allocated: 8.0, free: 6.7
after loss.backward      : GPU memory (Torch) total: 14.7, reserved: 13.0, allocated: 10.0, free: 4.7
after opt.step           : GPU memory (Torch) total: 14.7, reserved: 14.0, allocated: 10.0, free: 4.7
after opt.zero_grad      : GPU memory (Torch) total: 14.7, reserved: 14.0, allocated: 8.0, free: 6.7
```

#### WIP
Pt2.0, dynamo + inductor
JIT + JAX + XLA
Single card perf tuning 
https://huggingface.co/docs/transformers/perf_train_gpu_one 
Distributed training
Launcher
Orchestrator 
Parallelism & Sharding: FSDP & PJIT 
FSDP 
https://engineering.fb.com/2021/07/15/open-source/fsdp/ 
GSPMD & gshard


PJIT
https://irhum.github.io/blog/pjit/ 
https://jax.readthedocs.io/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html 
Efficient scaling transformer inference 
https://arxiv.org/pdf/2211.05102.pdf 
JAX (model training framework on top of python/numpy, comparable to tensorflow and pytorch)
https://github.com/google/jax 
https://jax.readthedocs.io/en/latest/index.html 
Paxml (Pax) is a ML framework on top of JAX, designed specific for large LLM, comparable to MVAI
https://github.com/google/paxml 
Praxis: a layer library on top of Pax
https://github.com/google/praxis 


Questions
What does JIT mean in details?
MOE and gating network
What is JAX and how is it different from Tensorflow?
What does AutoGrad mean and how is it different from Pytorch’s?
How does JAX achieve auto parallelism?





