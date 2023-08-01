# n00bGPT (WIP)
This is a learning note of Transformer and LLM. I like learning by doing. I have some experience in Infra and little in ML, though barely little on LLM. Below is my learning jounery. And hope it helps yours.

1. Kaggle competition: fresh touch on LLM with Huggingface Transformer fine tune
2. Attention is all you need: understand how does transformer work
3. NanoGPT: a codelab to build your own Transformer model from scratch
4. Transformer variant: understand the landscope of transformers
5. Training Infra: study scalability by FSDP, GSPMD, JAX, etc

### 0.Prior Knowledge
You probably don't need to read all of them. But those are what I know before this jounery.
- [Neural Network and Deep Learning by Andrew Ng](https://www.coursera.org/learn/neural-networks-deep-learning)
- [PyTorch tutorial quickstart](https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html)
- [Kaggle Pandas and ML courses](https://www.kaggle.com/learn)
- [Programming Massively Parallel Processors: A Hands-on Approach](https://www.amazon.com/Programming-Massively-Parallel-Processors-Hands/dp/0124159923)

### 1. Kaggle competition
Study for study is boring. Study for competition is fun! [LLM Science Exam](https://www.kaggle.com/competitions/kaggle-llm-science-exam) is what we need. Little prior knowledge required. You can build and train you very first transformer model now! Get a fresh touch, and also obviously a low rank on leaderboard. Create questions in mind, study knowledges below, and make our rank higher!



### 1. Attention is all you need
#### Reading list
- 
- Self attention and Transformer: explanation of Attention is all you Need
http://jalammar.github.io/illustrated-transformer/ 
Attention mechanism in RNN
https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/ 
Word embedding
https://machinelearningmastery.com/what-are-word-embeddings/ 


Questions
How is one language translated to another, i.e. how is the output of encoder processed in decoder?
How is transformer, seq-to-seq model be used in other tasks, e.g. multiple choice, question answering, text summarization, etc
Look into how is hugging face version implemented?
How large is a transformer model? How much is dense and how much is sparse?
How is words embedding calculated?
What loss function does it use? How does benchmark work? 
How can PyTorch autograd compute the grad for a particular element or row in a metrics for embedding table?
