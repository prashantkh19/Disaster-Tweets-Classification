# Disaster-Tweets-Classification
    Predict which Tweets are about real disasters and which ones are not.

Twitter has become an important communication channel in times of emergency.
The ubiquitousness of smartphones enables people to announce an emergency they’re observing in real-time. Because of this, more agencies are interested in programatically monitoring Twitter (i.e. disaster relief organizations and news agencies).

But, it’s not always clear whether a person’s words are actually announcing a disaster.
The task is to build a machine learning model that predicts which Tweets are about real disasters and which one’s aren’t.

# Dataset

The dataset has been acquired from the Kaggel Competition -[Real or Not? NLP with Disaster Tweets](https://www.kaggle.com/c/nlp-getting-started/overview).

Each sample in the train and test set has the following information:

- The `text` of a tweet
- A `keyword` from that tweet (although this may be blank!)
- The `location` the tweet was sent from (may also be blank)

### Acknowledgments

This dataset was created by the company figure-eight and originally shared on their ‘Data For Everyone’ website here.

Tweet source: https://twitter.com/AnyOtherAnnaK/status/629195955506708480

# Initial Research and References Used

The problem is a typical text classification problem. I used the approach mentioned in the paper 
[Universal Language Model Fine-tuning for Text Classification](https://arxiv.org/abs/1801.06146) by Jeremy Howard and Sebastian Ruder.

ULMFiT used a state-of-the-art language model at the time, the [AWD-LSTM](https://arxiv.org/abs/1708.02182). The AWD-LSTM is a regular LSTM with tuned dropout hyper-parameters. 

**Validation error rates for supervised and semi-supervised ULMFiT vs. 
training from scratch with different numbers of training examples on IMDb:**

<p align="center">
<img src="https://user-images.githubusercontent.com/27685757/71545890-be32ac00-29b6-11ea-8b0e-2dc5d2d22b92.PNG" />
</p>

### High level ULMFiT approach (IMDb example)

<p align="center">
<img src="https://user-images.githubusercontent.com/27685757/71545980-d820be80-29b7-11ea-9a6a-6664b992bf23.PNG" />
</p>

The approach involves fine-tuning a pre-trained language model in un-supervised manner and then using the encoder to build the classifier.

<I>The images are taken from the [fastai NLP page](http://nlp.fast.ai/).</I>

# Code:

The code is written using [fastai](https://docs.fast.ai/) modules based on [pytorch](https://pytorch.org/docs/stable/index.html).

Kaggle Kernel is used to code the whole project. The notebook can be viewed [here](https://www.kaggle.com/prashantkh19/disaster-tweets).

# Results:

- Got a public score of `0.80764` on the competition's test set.

### Loss Plot:

<p align="center">
<img src="https://user-images.githubusercontent.com/27685757/71546065-c8ee4080-29b8-11ea-95f2-14a4b19b3005.PNG" />
</p>

> **The notebook can be easily viewed [here](https://www.kaggle.com/prashantkh19/disaster-tweets).**
