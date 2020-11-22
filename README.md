# music-maker-2000

Create new and original songs / song snippets based on any artist in the world

This repo uses recurrent neural networks to generate new songs or song snippets based on lyrics from any current artist.

The application is deployed as a [stremlit app](https://www.streamlit.io/), and allows you to pick from a few pre-trained models, or train your own model from scratch, allowing you to select a number of hyper parameters and text cleaning options. 

There were a number of great online resources I used when creating this work, and would have been hard to do without:

 - [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) By Andrej Karpathy

 - [Text Generation with Pytorch](https://machinetalk.org/2019/02/08/text-generation-with-pytorch/) By Trung Tran 

 - [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) By Colah

 - [Pytorch Examples](https://github.com/jcjohnson/pytorch-examples) By J.C. Johnson

## Setup 


0 - Install [pipenv](https://pipenv.pypa.io/en/latest/) if you have not already. Ensure you have python 3.8 installed, or a version manager like [pyenv](https://realpython.com/intro-to-pyenv/) 

1 - Clone the repository

2 - Navigate to `music-maker-2000` directory

3 - Install the dependencies

> pipenv install

4 - Launch the app

> pipenv run app

(this should auto-direct you to `http://localhost:8501/` in your browser, but if it does not, navigate there)

5 - Enjoy!