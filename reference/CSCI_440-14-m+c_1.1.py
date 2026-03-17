#filename: CSCI_440-14-m+c_1.0.py

#I. Math


#II. Code
#the following code is to start a new file: train_1.0.py file

#dataset: https://huggingface.co/datasets/opus_books/viewer/en-it/train?row=60
#this is only library we'll use besides pytorch and huggingface tokenizer library to transform this text into a vocabulary
#subset: en-it
#Dataset Anatomy
# each data item has id (string), translation (a pair of sentences in { "src": "source blah", "tgt": "tgt blah" }
#we'll train our transformer to translate from src language to tgt language
#make code to dl this dataset, create tokenizer
#tokenizer: comes before Input Embedding
#tokenizer goal: split the sentence into single words (many strategies: BPE tokenizer, word level tokenizer, sub-word level, word part)
#Byte Pair Encoding (BPE) is a subword tokenization algorithm used in LLMs (GPT, RoBERTa) to balance vocabulary size and efficiency by merging frequent character pairs. It mitigates the out-of-vocabulary (OOV) problem, compresses text, and preserves semantic meaning. 
#we'll use the simplest: word level tokenizer: splits sentences by whitespace as word boundaries
#then map each word to 1 number, to build a vocabulary
#in the process we will build special tokens which will be used for the Transformer: padding, SOS: start-of-sentence, EOS: end-of-sentence, necessary for training the Transformer


import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

import datasets     #absent from original code
import tokenizers   #absent from original code
import tensorboard  #absent from original code

#import torch.utils.data: https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.Dataset

#import dataset and causal_mask from dataset.py
from dataset import BilingualDataset, causal_mask

#import model from file 01: model.py
from model import build_transformer

#import weights via config.py preload when training crashes
from config import get_weights_file_path, get_config

#get this from pip
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
#the class that will train the tokenizer, create the vocabulary given the list of sentences
from tokenizers.trainers import WorldLevelTrainer
#split words by whitespace
from tokenizers.pre_tokenizers import Whitespace

#import TensorBoard
from torch.utils.tensorboard import SummaryWriter

#import warnings, many will come from cuda
import warnings

#import tqdm (progress bar for training loop)
from tqdm import tqdm


#ability to create absolute paths from relative paths
from pathlib import Path

import time     #absent from original code

#run greedy decoding on our model, run the encoder only once
def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    #Precompute the encoder outut and reuse it for every token we get from the decoder
    #give it the source and source_mask, which is the encoder input and the encoder mask
    encoder_output = model.encode(source, source_mask)

    #how do we inference:
    #step 1: we give to the decoder the SOS token, so the decoder will output the 1st token of the translated sentence
    #step 2: at every iteration, we add the previous token to the Decoder input, so that the decoder can output the next token
    #step 3: we take the next token and put it again as input to the Decoder to get the successive token
    #build the decoder input for the 1st iteration, the SOS token
    #initialize the decoder input with the SOS token
    decoder_input = torch.empty(1,1).fill_(sos_idx).type_as(source).to(device)

    #we'll keep asking the Decoder to output the next token until either a) we reach EOS token, b) the max_len defined in fx's argument above
    #stopped 2:28
    while True:
        #1st stopping condition: if the decoder output, which becomes input of next step, reaches max_len
        #why do we have 2 dimensions: empty(1,1), 1 is for the batch, 1 is for the tokens of the decoder input
        if decoder_input.size(1) == max_len:
            break

        #build mask for the tgt (decoder input)
        #use causal_mask to prevent the decoder input from looking at future words
        #we don't need the other mask b/c here we don't have any PAD tokens
        #tensor.size(): numpy/pytorch to get the # of elements along the 2nd dimension (axis 1) of an array or tensor
        #thus decoder_input is a tensor, source_mask is a tensor
        #self.type_as(tensor).to(device): pytorch used to cast a tensor to the data type and device of another tensor
        #source_mask is the tensor with the desired type and device
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        #calculate the output of the decoder. reuse the output of the encoder for every iteration of the loop
        #.decode: pytorch to convert encoded data (model outputs, image, audio, video files) into tensors
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        #Get the next token, get the probabilities of the next token using the projection layer
        #but we only want the projection of the last token: -1
        #the next token after the last we have given the encoder
        #.project: pytorch custom method to project data into a lower dimensional space
        prob = model.project(out[:,-1])

        #Select the token with the maximum probability (using the greedy search)
        #_,: py tuple unpacking. _ (underscore): py for a 'throwaway' or dummy variable. 
        #_,: this variable indicates that the 1st returned tensor (the actual max probability values) is not needed for this function, and is being ignored
        #next_word: this variable is assigned the 2nd returned tensor, which holds the indices of the maximum probabilities.
        #these indices often represent the predicted ID of the most probable next word or class label for each item in a batch.
        #basically, the code identifies the most likely "next word" or "class" by its index, discarding the probability value itself.
        #the 'prob' tensor contains probability distributions across various possible words or classes for a batch of inputs.
        #torch.max finds max values and their corresponding indices along a specific dimension (dimension 1) of the tensor: prob

        _, next_word = torch.max(prob, dim=1)
        #torch.max(prob, dim=1): pytorch returns a tuple of 2 tensors
        #tensor 1: contains the max value found in each row of the 'prob' tensor along dim=1
        #tensor 2: contains the indices (positions) of these maximum values in each row. The indices are a LongTensor
        #take next_word and append it back to decoder_input, bc it will be the input of the next iteration
        decoder_input = torch.cat([decoder_input, torch.empty(1,1).type_as(source).fill_(next_word.item()).to(device)], dim=1)

        #if next word is equal to the End Of Sentence token, then stop 'break'
        if next_word == eos_idx:
            break

    #return the output, which is basically the decoder_input, bc everytime you're appending the next token to it
    #we remove the batch dim, so we squeeze it
    return decoder_input.squeeze(0)