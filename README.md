## NLP-papers-tools-discussion
Anything useful goes here

### Possible research directions and Fundamental discussions

[Structural probing: visualizing and understanding deep self-supervised language models, by Manning @ Stanford NLP - very important read](https://nlp.stanford.edu//~johnhew//structural-probe.html)

[On interpretability of BERT - similar in spirit to the work above](https://pair-code.github.io/interpretability/bert-tree/)

[The 4 Biggest Open Problems in NLP Today](http://ruder.io/4-biggest-open-problems-in-nlp/)

[What innate priors should we build into the architecture of deep learning systems? A debate between Prof. Y. LeCun and Prof. C. Manning](https://www.abigailsee.com/2018/02/21/deep-learning-structure-and-innate-priors.html)

### SOTA

For practically anything involving LM:

[XLNet: This King(s) are dead, long live the King! Sorry BERT, you were cool but now you are obolete :)](https://github.com/zihangdai/xlnet)

[GPT-2: Too dangerous to release to the public. Well here it is, with the weights and all.](https://github.com/huggingface/pytorch-pretrained-BERT#14-gpt2model)

[TRANSFORMER-XL: Still the only one you can use for realistically large documents. In the long term, IMHO this paper is a much more important contribution than BERT.](https://arxiv.org/pdf/1901.02860.pdf)

[BERT: The great multi-tasker, trained to do a number of things really well. Great theoreical contributions -- the pinnacle of Attentions.](https://github.com/google-research/bert)

[OpenAI Transformer, aka GPT-1](https://github.com/huggingface/pytorch-openai-transformer-lm)

### New Libraries we care about

[PAIR: People + AI Research by Google Brain](https://ai.google/research/teams/brain/pair)

[StellarGraph: a Python library for machine learning on graph-structured or network-structured data](https://github.com/stellargraph/stellargraph)

[The prupose of this repository is to store tools on text classification with deep learning](https://github.com/brightmart/text_classification)

[GluonNLP](https://gluon-nlp.mxnet.io/index.html)

[Representation learning on large graphs using stochastic graph convolutions.](https://github.com/bkj/pytorch-graphsage)

[Wow this is good! ULMFit for graphs! This person has a ton of other stuff, more productive thansome institutes](https://github.com/bkj/ulm-basenet)

[The Big-&-Extending-Repository-of-Transformers: Pretrained PyTorch models for Google's BERT, OpenAI GPT & GPT-2, Google/CMU Transformer-XL.](https://github.com/huggingface/pytorch-pretrained-BERT)

[Flair - LM/Embedding/General NLP lib. Fastest growing NLP project on github](https://github.com/zalandoresearch/flair)

[FAIRseq - Facebook AI Research toolkit for seqence modeling; features multi-GPU (distributed) training on one machine or across multiple machines. PyTorch](https://github.com/pytorch/fairseq)

[AllenNLP - An open-source NLP research library, built on PyTorch by Allen AI Research Institute](https://github.com/allenai/allennlp)

[FastAI - ULMFit, Transformer, TransformerXL implementations and more](https://docs.fast.ai/text.html)

### Visualization Tools

People + AI Research (PAIR) by the Google Brain team:

[What If...you could inspect a machine learning model, with minimal coding required?](https://pair-code.github.io/what-if-tool/)

[FACETS - KNOW YOUR DATA](https://pair-code.github.io/facets/)

#### General:

[TensorboardX for PyTorch](https://github.com/arjunnlp/tensorboardX)

[Visdom - similar to tensorboard](https://github.com/arjunnlp/visdom)

#### Sequential:

[LSTMVis: Visualizng LSTM](https://github.com/HendrikStrobelt/LSTMVis)

[Seq2Seq Vis: Visualization for Sequential Neural Networks with Attention](https://github.com/HendrikStrobelt/Seq2Seq-Vis)

#### Attention:

[BERTVizTool for visualizing attention in BERT and OpenAI GPT-2](https://github.com/jessevig/bertviz)

[tensor2tensor: visualizing Transformer paper](https://github.com/tensorflow/tensor2tensor/tree/master/tensor2tensor/visualization)

#### How to do visualization or highly visual articles:

[A.I. Experiments: Visualizing High-Dimensional Space](https://www.youtube.com/watch?v=wvsE8jm1GzE)

[Guide to visualization of NLP representations and neural nets by C.Olah @ Google Brain](http://colah.github.io/)

[Data Visualization](https://towardsdatascience.com/data-visualization/home)

[The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html)

[Deconstructing BERT: Distilling 6 Patterns from 100 Million Parameters, Part 1](https://towardsdatascience.com/deconstructing-bert-distilling-6-patterns-from-100-million-parameters-b49113672f77)

[Deconstructing BERT: Distilling 6 Patterns from 100 Million Parameters, Part 2](https://towardsdatascience.com/deconstructing-bert-part-2-visualizing-the-inner-workings-of-attention-60a16d86b5c1)

[The illustrated BERT, ELMo, ULMFit, and Transformer](https://jalammar.github.io/illustrated-bert/)

[Visualizing Representations: Deep Learning for Human Beings](http://colah.github.io/posts/2015-01-Visualizing-Representations/)

[Jay Allamar: Visualizing machine learning one concept at a time](https://jalammar.github.io/)

[Attention? Attention!](https://lilianweng.github.io/lil-log/2018/06/24/attention-attention.html)

### My Bag of Tricks for Deep Learning performance - add yours too:

#### First, make sure you got all the NVIDIA stuff:

[NVIDIA Apex: A PyTorch Extension: Tools for easy mixed precision and distributed training in Pytorch](https://github.com/nvidia/apex)

[NVIDIA cuDNN: provides highly tuned implementations for standard routines such as forward and backward convolution, pooling, normalization, and activation layers.](https://developer.nvidia.com/cudnn)

[NVIDIA NCCL: The NVIDIA Collective Communications Library (NCCL) implements multi-GPU and multi-node collective communication primitives that are performance optimized for NVIDIA GPUs](https://developer.nvidia.com/nccl)

[NVIDIA DALI: A library containing both highly optimized building blocks and an execution engine for data pre-processing in deep learning applications](https://github.com/NVIDIA/DALI)

Consider using these:

[NVIDIA optimized and tuned containers for various frameworks](https://developer.nvidia.com/deep-learning-frameworks)

#### Next we do parallelism:

[pandas.DataFrame.swifter.apply](https://medium.com/@jmcarpenter2/swiftapply-automatically-efficient-pandas-apply-operations-50e1058909f9)

Swifter will automatically apply the fastest method available (or so it says, more on it later). You want to make sure you have stuff like Dask intalled. It chooses between vectorization, Dask, and traditional pandas.apply

```
$ pip install -U pandas
$ pip install swifter

import pandas as pd
import swifter

mydf['outCol'] = df['inCol'].swifter.apply(anyfunction)
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

[Mini-batch data parallelism, sort of default in PyTorch](https://towardsdatascience.com/speed-up-your-algorithms-part-1-pytorch-56d8a4ae7051)

#### Compile your code the easy way

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

#### Eliminate memory leaks

[ipyexperiments - will save you 20-30% video and 10-15% system memory](https://github.com/stas00/ipyexperiments)

[ipyexperiments usage examples in some kaggle contest code I wrote](https://github.com/arjunnlp/NLP-papers-tools-discussion/blob/master/preprocess-dainis.ipynb)

Make sure to either use IPyTorchExperiments all the time, or IPyCPUExperiments if don't care to use GPU. If you are using a GPU, you must be sure to use the IPyTorchExperiments and that the text after the cell tells you it is indeed using GPU backend.

#### Speed up your loops

In general, using numpy operations is preferred, e.g. `np.sum()` beats iterating. 

Avoid if-else by using np.where is a big one. Here is an example of going from 1 trillion operations to 1 operation. Assuming
each operation takes a nanosecond, that's 17 minutes vs 1 nanosecond.

```
# X is some numpy array, and you have a 1000 of those in a dataframe or in a list
# If your column is 1000 in length, this is 1000 operations * size of numpy array (say 1000) = 1000000 operations
def fun(x):
    if x > 0:
        x =+ 1
    else:
        x = 0
    return x
    
# ~1000000000000 operations
for X in data:
    for x in X:
        output.append(fun(x))

# ~1000000 operations
df['data'].apply([x for x in X])

# ~1000 operations you are doing  no looping but pandas is single-threaded...
df['data'].apply(X)

# This is very fast, using vector math extensions. 1 Op on Xeon or i9 with MKL Installed.
def fun2(x):
    x[np.where(x > 0)] += 1
    x[np.where(x <= 0)] = 0
    return x

df['data'].swifter.apply(fun2)
```

Assume `data` contains some items that we abstract as `...` . In general, follow this rule of thumb:

1. Slowest: `for i in range(len(data)):`

2. OK: `for d in data:`

3. Faster: `[d for d in data]`

4. Fastest `(d for d in data)`

### Help with Class Balance and Distribution Issues

[Learning to Reweight Examples for Robust Deep Learning](https://arxiv.org/pdf/1803.09050.pdf)
[implementation of the "Learning to Reweight..." paper](https://github.com/danieltan07/learning-to-reweight-examples)

[PyTorch imbalanced-dataset-toolkit](https://github.com/ufoym/imbalanced-dataset-sampler)

[A Python Package to Tackle the Curse of Imbalanced Datasets in Machine Learning](https://github.com/scikit-learn-contrib/imbalanced-learn)

[CVPR, Kaggle Winner: "Large Scale Fine-Grained Categorization and Domain-Specific Transfer Learning" with Imbalanced Class Labels](https://vision.cornell.edu/se3/wp-content/uploads/2018/03/FGVC_CVPR_2018.pdf)

[8 Tactics to Combat Imbalanced Classes in Your Machine Learning Dataset](https://machinelearningmastery.com/tactics-to-combat-imbalanced-classes-in-your-machine-learning-dataset/)

[Probability calibration](https://www.kaggle.com/dowakin/probability-calibration-0-005-to-lb/notebook)

[Training on validation set when train and test are different distributions](https://www.kaggle.com/c/imaterialist-challenge-fashion-2018/discussion/57944)

[Deep Learning Unbalanced Data](https://towardsdatascience.com/deep-learning-unbalanced-training-data-solve-it-like-this-6c528e9efea6)

### Other useful tools

[mlextend 0 a library with useful extensions to a variaty of ML/NLP tools](http://rasbt.github.io/mlxtend/)

[DataFrameSummary: An extension to pandas dataframes describe function](https://github.com/mouradmourafiq/pandas-summary)

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

[Jeremy Howard](https://twitter.com/jeremyphoward) 

[Jay Allamar](https://jalammar.github.io/)
