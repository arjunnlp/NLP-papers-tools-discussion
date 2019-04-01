## NLP-papers-tools-discussion
Anything useful goes here

### Possible research directions and Fundamental discussions

[The 4 Biggest Open Problems in NLP Today](http://ruder.io/4-biggest-open-problems-in-nlp/)

[What innate priors should we build into the architecture of deep learning systems? A debate between Prof. Y. LeCun and Prof. C. Manning](https://www.abigailsee.com/2018/02/21/deep-learning-structure-and-innate-priors.html)

### SOTA

For practically anything involving LM:

[TRANSFORMER-XL: ATTENTIVE LANGUAGE MODELS
BEYOND A FIXED-LENGTH CONTEXT](https://arxiv.org/pdf/1901.02860.pdf)

### New Libraries we care about

[Flair - LM/Embedding/General NLP lib. Fastest growing NLP project on github](https://github.com/zalandoresearch/flair)

[FAIRseq - Facebook AI Research toolkit for seqence modeling; features multi-GPU (distributed) training on one machine or across multiple machines. PyTorch](https://github.com/pytorch/fairseq)

[AllenNLP - An open-source NLP research library, built on PyTorch by Allen AI Research Institute](https://github.com/allenai/allennlp)

[FastAI - ULMFit, Transformer, TransformerXL implementations and more](https://docs.fast.ai/text.html)

### My Bag of Tricks for Deep Learning performance - add yours too:

First, make sure you got all the NVIDIA stuff:

[NVIDIA Apex: A PyTorch Extension: Tools for easy mixed precision and distributed training in Pytorch](https://github.com/nvidia/apex)

[NVIDIA cuDNN: provides highly tuned implementations for standard routines such as forward and backward convolution, pooling, normalization, and activation layers.](https://developer.nvidia.com/cudnn)

[NVIDIA NCCL: The NVIDIA Collective Communications Library (NCCL) implements multi-GPU and multi-node collective communication primitives that are performance optimized for NVIDIA GPUs](https://developer.nvidia.com/nccl)

[NVIDIA DALI: A library containing both highly optimized building blocks and an execution engine for data pre-processing in deep learning applications](https://github.com/NVIDIA/DALI)

Consider using these:

[NVIDIA optimized and tuned containers for various frameworks](https://developer.nvidia.com/deep-learning-frameworks)

Next:

[Mini-batch data parallelism, sort of default in PyTorch](https://towardsdatascience.com/speed-up-your-algorithms-part-1-pytorch-56d8a4ae7051)

[Numba: compiled and highly optimized C/C++/Fortran code will be used instead of slow numpy (even cython is slower)](https://towardsdatascience.com/speed-up-your-algorithms-part-2-numba-293e554c5cc1)


Best of all you still code in python, just need a decorator on top of time-consuming function. MAKE SURE IT IS TIME CONSUMING - just spamming @njit eveywhere will do the opposite of what you want, initializing numba costs resources!

```
from numba import jit, int32

# a 4-letter magick word that will make any function that
# takes 20-30 seconds finish in 5 or so!

@njit
def function0(a, b):
    # your loop 
    return result
    
# we declare return value and types, turn off jit compiler 
# and go directly for binary (making it harder to debug 
# but SO much faster. Finally all vector ops will be 
# distributed between cores if your CPU
    
@jit(int32(int32, int32), nopython=true, parallel=true)
def function(a, b):
    # your loop or numerically intensive computations
    return result

# in this function we are saying "you are no longer restricted
# to types we specify, just run it all in parallel, on one
# or more CPUs, using threads or processes or whatever!
# numba is smart enough to figure out the best way to do so

@vectorize
def function2(c):
    # your loop or numerically intensive computations
    return result
```

[DASK - parallelizing numpy, pandasm python, scikit-learn, literally everything...](https://towardsdatascience.com/speeding-up-your-algorithms-part-4-dask-7c6ed79994ef)

[Modin: An alternative to DASK but only for Pandas - much simpler and lighter if I/O is what you need. Will process 10 GB DataFrame in seconds.](https://github.com/modin-project/modin)
```
# replace the following line
#import pandas as pd
# with
import modin.pandas as pd
```

You are done, pandas is 10-30 times faster on some tasks! but sometimes will crash :)

[ipyexperiments - will save you 20-30% video and 10-15% system memory](https://github.com/stas00/ipyexperiments)

[ipyexperiof usage examples in some kaggle contest code I wrote](https://github.com/arjunnlp/NLP-papers-tools-discussion/blob/master/preprocess-dainis.ipynb)

Make sure to either use IPyTorchExperiments all the time, or IPyCPUExperiments if don't care to use GPU. If you are using a GPU, you must be sure to use the IPyTorchExperiments and that the text after the cell tells you it is indeed using GPU backend.

### Visualization Tools

General:

[TensorboardX for PyTorch](https://github.com/arjunnlp/tensorboardX)

[Visdom - similar to tensorboard](https://github.com/arjunnlp/visdom)

Sequential:

[LSTMVis: Visualizng LSTM](https://github.com/HendrikStrobelt/LSTMVis)

[Seq2Seq Vis: Visualization for Sequential Neural Networks with Attention](https://github.com/HendrikStrobelt/Seq2Seq-Vis)

Attention:

[BERTVizTool for visualizing attention in BERT and OpenAI GPT-2](https://github.com/jessevig/bertviz)

[tensor2tensor: visualizing Transformer paper](https://github.com/tensorflow/tensor2tensor/tree/master/tensor2tensor/visualization)

How to do visualization or highly visual articles:

[Guide to visualization of NLP representations and neural nets by C.Olah @ Google Brain](http://colah.github.io/)

[Data Visualization](https://towardsdatascience.com/data-visualization/home)

[The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html)

[Deconstructing BERT: Distilling 6 Patterns from 100 Million Parameters, Part 1](https://towardsdatascience.com/deconstructing-bert-distilling-6-patterns-from-100-million-parameters-b49113672f77)

[Deconstructing BERT: Distilling 6 Patterns from 100 Million Parameters, Part 2](https://towardsdatascience.com/deconstructing-bert-part-2-visualizing-the-inner-workings-of-attention-60a16d86b5c1)

[The illustrated BERT, ELMo, ULMFit, and Transformer](https://jalammar.github.io/illustrated-bert/)

[Visualizing Representations: Deep Learning for Human Beings](http://colah.github.io/posts/2015-01-Visualizing-Representations/)

[Jay Allamar: Visualizing machine learning one concept at a time](https://jalammar.github.io/)

[Attention? Attention!](https://lilianweng.github.io/lil-log/2018/06/24/attention-attention.html)

### Data Preprocessing tools

[PyTorch imbalanced-dataset-toolkit](https://github.com/ufoym/imbalanced-dataset-sampler)

### Paper and Technical Writing HOWTO

On writing research papers:

["How to Write an Introduction" by Dr. Om Gnawali](http://www2.cs.uh.edu/~gnawali/courses/cosc6321-s17/hw7.html)

Some of the best examples of technical writing (papers & blogs go hand in hand!):

[How to trick a neural network into thinking a panda is a vulture](https://codewords.recurse.com/issues/five/why-do-neural-networks-think-a-panda-is-a-vulture)

[Picking an optimizer for Style Transfer](https://blog.slavv.com/picking-an-optimizer-for-style-transfer-86e7b8cba84b)

[How do we 'train' Neural Networks?](https://towardsdatascience.com/how-do-we-train-neural-networks-edd985562b73)

### Must-read papers and technical articles

[Disciplined Training of Neural Networks](https://arxiv.org/abs/1803.09820)

On Language Models:

[NLP's ImageNet moment has arrived](http://ruder.io/nlp-imagenet/)

[BERT: Pre-training of Deep Bidirectional Transformers for
Language Understanding](https://arxiv.org/pdf/1810.04805.pdf)

[Attention is All You Need](https://arxiv.org/pdf/1706.03762.pdf)

[Improving Language Understanding by Generative Pre-Training](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)

[Deep contextualized word representations](https://arxiv.org/pdf/1802.05365.pdf)

[Universal Language Model Fine-tuning for Text Classification](https://arxiv.org/pdf/1801.06146.pdf)

[TRANSFORMER-XL: ATTENTIVE LANGUAGE MODELS
BEYOND A FIXED-LENGTH CONTEXT](https://arxiv.org/pdf/1901.02860.pdf)

[Comparing complex NLP models for complex languages on a set of real tasks](https://towardsdatascience.com/complexity-generalization-computational-cost-in-nlp-modeling-of-morphologically-rich-languages-7fa2c0b45909)

[You don't need RNNs: When Recurrent Models Don't Need to be Recurrent](https://bair.berkeley.edu/blog/2018/08/06/recurrent/)

Other:

[Deep Graph Methods Survey](https://arxiv.org/pdf/1901.00596.pdf)

[Baselines and Bigrams: Simple, Good Sentiment and Topic Classification](https://www.aclweb.org/anthology/P12-2018)

### Blogs You Should Follow

[Stanford AI Salon: bi-weekly discussion on important topic in AI/NLP/ML, closed from the public due to space restrictions, but notes and videos are now posted in a blog. Previous guests include LeCun, Hinton, Ng., and others](http://ai.stanford.edu/blog/)

[Anrej Karpathy](https://medium.com/@karpathy)

[Vitaliy Bushaev](https://towardsdatascience.com/@bushaev)

[Sylvain Gugger](https://sgugger.github.io/)

[Sebastian Ruder](http://ruder.io/)

[Ilya Sutskever](https://blog.openai.com/tag/ilya-sutskever/) - Sutskever published his Transformer in this blog, not even on arxiv

[Jeremy Howard](https://twitter.com/jeremyphoward) 

[Jay Allamar](https://jalammar.github.io/)
