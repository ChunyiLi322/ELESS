import sys
import argparse
import csv
import json
import time
import os
import numpy as np
import fnmatch
import criteria
import string
import pickle
import random
import math

import tensorflow as tf
import tensorflow_hub as hub

import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, SequentialSampler, TensorDataset
tf.compat.v1.disable_eager_execution()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from BERT.tokenization import BertTokenizer
from BERT.modeling import BertForSequenceClassification, BertConfig
torch.set_num_threads(6)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



#---------------------------------------------------------------
# from  texthoaxer.TextHoaxer_nli import texthoaxer_attack

from InferSent.models import NLINet

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
ESIM_ROOT = os.path.join(PROJECT_ROOT, "ESIM")

if ESIM_ROOT not in sys.path:
    sys.path.insert(0, ESIM_ROOT)

from esim.model import ESIM
# from esim.data import Preprocessor
# from esim.utils import correct_predictions

from pww.adversarial_tools import ForwardGradWrapper, PWW , PWW_nli
from pww.word_level_process import  get_tokenizer

from deepWordBug.attack import  deepWordBug_nli
from deepWordBug.dwbloaddata import loaddatawithtokenize, custom_collate_fn
from deepWordBug.dwbdataloader import Worddata

# from HLBB.classification_attack import HLBB

from hardlabel.nli_attack import hardlabelattack_nli


#
from limeattack.generate_embedding_mat import generate_embedding_mat,csim_matrix_nli
from limeattack.LimeAttack_classification import attacklime_nli



import pickle
import simcompute_nli 

seed = 6
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


class NLI_infer_InferSent(nn.Module):
    def __init__(self,
                 pretrained_file,
                 embedding_path,
                 data,
                 batch_size=32):
        super(NLI_infer_InferSent, self).__init__()

        #         self.device = torch.device("cuda:{}".format(local_rank) if local_rank > -1 else "cpu")
        # torch.cuda.set_device(local_rank)

        # Retrieving model parameters from checkpoint.
        config_nli_model = {
            'word_emb_dim': 200,
            'enc_lstm_dim': 2048,
            'n_enc_layers': 1,
            'dpout_model': 0.,
            'dpout_fc': 0.,
            'fc_dim': 512,
            'bsize': batch_size,
            'n_classes': 3,
            'pool_type': 'max',
            'nonlinear_fc': 1,
            'encoder_type': 'InferSent',
            'use_cuda': True,
            'use_target': False,
            'version': 1,
        }

        print("\t* Building model...")
        self.model = NLINet(config_nli_model).cuda()
        print("Reloading pretrained parameters...")
        self.model.load_state_dict(torch.load(pretrained_file, map_location='cuda:0'))

        # construct dataset loader
        print('Building vocab and embeddings...')
        self.dataset = NLIDataset_InferSent(embedding_path, data=data, batch_size=batch_size)

    def text_pred(self, text_data, datasize=0):
        """Predict all NLI pairs and return one probability tensor per input pair.

        The original implementation returned only the final mini-batch when
        datasize == 0. ELESS evaluates synonym sets in batches, so all batches
        must be concatenated. ``datasize`` is kept only for API compatibility.
        """
        self.model.eval()
        data_batches = self.dataset.transform_text(text_data)
        probs_all = []
        with torch.no_grad():
            for batch in data_batches:
                (s1_batch, s1_len), (s2_batch, s2_len) = batch
                s1_batch, s2_batch = s1_batch.cuda(), s2_batch.cuda()
                logits = self.model((s1_batch, s1_len), (s2_batch, s2_len))
                probs_all.append(nn.functional.softmax(logits, dim=-1))

        if not probs_all:
            return torch.empty((0, 3), device='cuda')
        return torch.cat(probs_all, dim=0)

    def dwb_text_pred(self, text_data, datasize = 0):
        # Switch the model to eval mode.
        self.model.eval()
        # transform text data into indices and create batches
        
        # print("____________dwb_text_pred_________text_data______",text_data)
        data_batches = self.dataset.transform_text(text_data)

        # Deactivate autograd for evaluation.
        probs_all = []

        with torch.no_grad():
            for batch in data_batches:
                # Move input and output data to the GPU if one is used.
                (s1_batch, s1_len), (s2_batch, s2_len) = batch
                s1_batch, s2_batch = s1_batch.cuda(), s2_batch.cuda()
                logits = self.model((s1_batch, s1_len), (s2_batch, s2_len))
                probs = F.log_softmax(logits, dim=-1)
                probs = 2 * torch.exp(probs)
                # print("___________________________",)

                # 将对数概率存入 probs_all
                probs_all.append(probs)

        # 将所有批次的概率拼接起来，返回形状为 [total_samples, num_classes]
        return torch.cat(probs_all, dim=0)
    
    def lime_text_pred(self, hypothese_str_list, premises, true_label, batch_size=128): 

        self.model.eval()

        # 1. 把单个 premise 变成 token 列表
        #    - 如果已经是 ['A','person',...] 这种 list，就直接用
        #    - 如果是字符串 "A person ..." 就先 .split()
        if isinstance(premises, str):
            premise_tokens = premises.split()
        else:
            premise_tokens = premises

                        
        # 2. 复制这个唯一的 premise，数量 = hypothesis 的个数
        premises_list = [premise_tokens for _ in range(len(hypothese_str_list))]

        #  确保 hypothese_str_list 是一个 list（有时候可能传进来的是单个字符串）
        if isinstance(hypothese_str_list, str):
            hypothese_str_list = [hypothese_str_list]
            
        # print("----hypothese_str_list---",hypothese_str_list)
        # 3. 处理每个 hypothesis：既支持 str 也支持已经是 token list 的情况
        hypotheses_list = []
        for h in hypothese_str_list:
            if isinstance(h, str):
                # 字符串 -> 按空格切成 tokens
                hypotheses_list.append(h.split())
            elif isinstance(h, (list, tuple)):
                # 已经是 token list/tuple -> 直接用
                hypotheses_list.append(h)
            else:
                # 其他奇怪类型，兜底：先转成字符串再 split
                hypotheses_list.append(str(h).split())
       
        # print("----hypotheses_list---",hypotheses_list)
        # 4. 构造 A，交给 dataset 处理
        A = {
            'premises': premises_list,
            'hypotheses': hypotheses_list
        }
        
        # print("----A---",A)
        
        # 如果你的 transform_text 支持 batch_size，就加上；否则就用 transform_text(A)
        dataloader = self.dataset.transform_text(A)
        
        

        probs_all = []

        with torch.no_grad():
            for batch in dataloader:
                # Move input and output data to the GPU if one is used.
                (s1_batch, s1_len), (s2_batch, s2_len) = batch
                s1_batch, s2_batch = s1_batch.cuda(), s2_batch.cuda()
                logits = self.model((s1_batch, s1_len), (s2_batch, s2_len))
                probs = F.log_softmax(logits, dim=-1)
                probs = 2 * torch.exp(probs)
                # print("___________________________",)

                # 将对数概率存入 probs_all
                probs_all.append(probs)

        probs_all_cat = torch.cat(probs_all, dim=0)      # [N, num_classes]
        max_indices = torch.argmax(probs_all_cat, dim=1) # [N]

        hard_label = torch.zeros_like(probs_all_cat, device=probs_all_cat.device)
        hard_label[torch.arange(hard_label.size(0)), max_indices] = 1
        
        # print("------------hard_label.cpu().numpy() -----------",hard_label.cpu().numpy() )

        # return hard_label.scpu().numpy()
        return hard_label.detach().cpu()







class NLI_infer_ESIM(nn.Module):
    def __init__(self,
                 pretrained_file,
                 worddict_path,
                 local_rank=-1,
                 batch_size=32):
        super(NLI_infer_ESIM, self).__init__()

        self.batch_size = batch_size
        self.device = torch.device("cpu")
        checkpoint = torch.load(pretrained_file)
        # Retrieving model parameters from checkpoint.
        vocab_size = checkpoint['model']['_word_embedding.weight'].size(0)
        embedding_dim = checkpoint['model']['_word_embedding.weight'].size(1)
        hidden_size = checkpoint['model']['_projection.0.weight'].size(0)
        num_classes = checkpoint['model']['_classification.4.weight'].size(0)

        print("\t* Building model...")
        self.model = ESIM(vocab_size,
                          embedding_dim,
                          hidden_size,
                          num_classes=num_classes,
                          device=self.device).to(self.device)

        self.model.load_state_dict(checkpoint['model'])

        # construct dataset loader
        self.dataset = NLIDataset_ESIM(worddict_path)

    def text_pred(self, text_data):
        # Switch the model to eval mode.
        self.model.eval()
        device = self.device

        # transform text data into indices and create batches
        self.dataset.transform_text(text_data)
        dataloader = DataLoader(self.dataset, shuffle=False, batch_size=self.batch_size)

        # Deactivate autograd for evaluation.
        probs_all = []
        with torch.no_grad():
            for batch in dataloader:
                # Move input and output data to the GPU if one is used.
                premises = batch['premise'].to(device)
                premises_lengths = batch['premise_length'].to(device)
                hypotheses = batch['hypothesis'].to(device)
                hypotheses_lengths = batch['hypothesis_length'].to(device)

                _, probs = self.model(premises,
                                      premises_lengths,
                                      hypotheses,
                                      hypotheses_lengths)
                probs_all.append(probs)

        return torch.cat(probs_all, dim=0)

    def dwb_text_pred(self, text_data):
        # Switch the model to eval mode.
        self.model.eval()
        device = self.device

        # transform text data into indices and create batches
        self.dataset.transform_text(text_data)
        dataloader = DataLoader(self.dataset, shuffle=False, batch_size=self.batch_size)

        # Deactivate autograd for evaluation.
        probs_all = []
        with torch.no_grad():
            for batch in dataloader:
                # Move input and output data to the GPU if one is used.
                premises = batch['premise'].to(device)
                premises_lengths = batch['premise_length'].to(device)
                hypotheses = batch['hypothesis'].to(device)
                hypotheses_lengths = batch['hypothesis_length'].to(device)

                _, probs = self.model(premises,
                                      premises_lengths,
                                      hypotheses,
                                      hypotheses_lengths)
                probs_all.append(probs)

        return torch.cat(probs_all, dim=1)

    def lime_text_pred(self, hypothese_str_list, premises, true_label, batch_size=128): 

        # Switch the model to eval mode.
        self.model.eval()
        device = self.device

        # 1. 把单个 premise 变成 token 列表
        #    - 如果已经是 ['A','person',...] 这种 list，就直接用
        #    - 如果是字符串 "A person ..." 就先 .split()
        if isinstance(premises, str):
            premise_tokens = premises.split()
        else:
            premise_tokens = premises

                        
        # 2. 复制这个唯一的 premise，数量 = hypothesis 的个数
        premises_list = [premise_tokens for _ in range(len(hypothese_str_list))]

        #  确保 hypothese_str_list 是一个 list（有时候可能传进来的是单个字符串）
        if isinstance(hypothese_str_list, str):
            hypothese_str_list = [hypothese_str_list]
            
        # print("----hypothese_str_list---",hypothese_str_list)
        # 3. 处理每个 hypothesis：既支持 str 也支持已经是 token list 的情况
        hypotheses_list = []
        for h in hypothese_str_list:
            if isinstance(h, str):
                # 字符串 -> 按空格切成 tokens
                hypotheses_list.append(h.split())
            elif isinstance(h, (list, tuple)):
                # 已经是 token list/tuple -> 直接用
                hypotheses_list.append(h)
            else:
                # 其他奇怪类型，兜底：先转成字符串再 split
                hypotheses_list.append(str(h).split())
       
        # print("----hypotheses_list---",hypotheses_list)
        # 4. 构造 A，交给 dataset 处理
        A = {
            'premises': premises_list,
            'hypotheses': hypotheses_list
        }
        
        # print("----A---",A)
        
        # 如果你的 transform_text 支持 batch_size，就加上；否则就用 transform_text(A)
        self.dataset.transform_text(A)
        dataloader = DataLoader(self.dataset, shuffle=False, batch_size=self.batch_size)

        # Deactivate autograd for evaluation.
        probs_all = []
        with torch.no_grad():
            for batch in dataloader:
                # Move input and output data to the GPU if one is used.
                premises = batch['premise'].to(device)
                premises_lengths = batch['premise_length'].to(device)
                hypotheses = batch['hypothesis'].to(device)
                hypotheses_lengths = batch['hypothesis_length'].to(device)

                _, probs = self.model(premises,
                                      premises_lengths,
                                      hypotheses,
                                      hypotheses_lengths)
                probs_all.append(probs)

        probs_all_cat = torch.cat(probs_all, dim=0)      # [N, num_classes]
        max_indices = torch.argmax(probs_all_cat, dim=1) # [N]

        hard_label = torch.zeros_like(probs_all_cat, device=probs_all_cat.device)
        hard_label[torch.arange(hard_label.size(0)), max_indices] = 1
        
        # print("------------hard_label.cpu().numpy() -----------",hard_label.cpu().numpy() )

        # return hard_label.scpu().numpy()
        return hard_label.detach().cpu() 
    

class NLI_infer_BERT(nn.Module):
    def __init__(self,
                 pretrained_dir,
                 max_seq_length=128,
                 batch_size=32):
        super(NLI_infer_BERT, self).__init__()
        self.model = BertForSequenceClassification.from_pretrained(pretrained_dir, num_labels=3).cuda()

        # construct dataset loader
        self.dataset = NLIDataset_BERT(pretrained_dir, max_seq_length=max_seq_length, batch_size=batch_size)

    def text_pred(self, text_data):
        # Switch the model to eval mode.
        self.model.eval()

        # transform text data into indices and create batches
        dataloader = self.dataset.transform_text(text_data)

        probs_all = []
        #         for input_ids, input_mask, segment_ids in tqdm(dataloader, desc="Evaluating"):
        for input_ids, input_mask, segment_ids in dataloader:
            input_ids = input_ids.cuda()
            input_mask = input_mask.cuda()
            segment_ids = segment_ids.cuda()

            with torch.no_grad():
                logits = self.model(input_ids, segment_ids, input_mask)
                probs = nn.functional.softmax(logits,  dim=-1)
                probs_all.append(probs)

        return torch.cat(probs_all, dim=0)
    
    def dwb_text_pred(self, text_data, batch_size=32):
        # Switch the model to eval mode.
        # print("__________model.eval()之前_______________")
        self.model.eval()
        # transform text data into indices and create batches
        # print("__________dataloader之前_______________")
        # text_data_str = [[str(i) for i in sublist] for sublist in text_data]
        dataloader = self.dataset.transform_text(text_data)

        probs_all = []
        #         for input_ids, input_mask, segment_ids in tqdm(dataloader, desc="Evaluating"):
        for input_ids, input_mask, segment_ids in dataloader:
            input_ids = input_ids.cuda()
            input_mask = input_mask.cuda()
            segment_ids = segment_ids.cuda()
            # print("__________输入数据已经转移到gpu上_______________")

            with torch.no_grad():
                logits = self.model(input_ids, segment_ids, input_mask)
                probs = nn.functional.softmax(logits, dim=-1)
                probs_all.append(probs)

        return torch.cat(probs_all, dim=0)

    def lime_text_pred(self, hypothese_str_list, premises, true_label, batch_size=128): 

        self.model.eval()

        # 1. 把单个 premise 变成 token 列表
        #    - 如果已经是 ['A','person',...] 这种 list，就直接用
        #    - 如果是字符串 "A person ..." 就先 .split()
        if isinstance(premises, str):
            premise_tokens = premises.split()
        else:
            premise_tokens = premises

                        
        # 2. 复制这个唯一的 premise，数量 = hypothesis 的个数
        premises_list = [premise_tokens for _ in range(len(hypothese_str_list))]
        true_label_list = [true_label for _ in range(len(hypothese_str_list))]

        #  确保 hypothese_str_list 是一个 list（有时候可能传进来的是单个字符串）
        if isinstance(hypothese_str_list, str):
            hypothese_str_list = [hypothese_str_list]
            
        # print("----hypothese_str_list---",hypothese_str_list)
        # 3. 处理每个 hypothesis：既支持 str 也支持已经是 token list 的情况
        hypotheses_list = []
        for h in hypothese_str_list:
            if isinstance(h, str):
                # 字符串 -> 按空格切成 tokens
                hypotheses_list.append(h.split())
            elif isinstance(h, (list, tuple)):
                # 已经是 token list/tuple -> 直接用
                hypotheses_list.append(h)
            else:
                # 其他奇怪类型，兜底：先转成字符串再 split
                hypotheses_list.append(str(h).split())
       
        # print("----hypotheses_list---",hypotheses_list)
        # 4. 构造 A，交给 dataset 处理
        A = {
            'premises': premises_list,
            'hypotheses': hypotheses_list
        }
        
        # print("----A---",A)
        
        # 如果你的 transform_text 支持 batch_size，就加上；否则就用 transform_text(A)
        dataloader = self.dataset.transform_text(A)
        
        

        probs_all = []

       
        for input_ids, input_mask, segment_ids in dataloader:
            input_ids = input_ids.cuda()
            input_mask = input_mask.cuda()
            segment_ids = segment_ids.cuda()
            # print("__________输入数据已经转移到gpu上_______________")

            with torch.no_grad():
                logits = self.model(input_ids, segment_ids, input_mask)
                probs = nn.functional.softmax(logits, dim=-1)
                probs_all.append(probs)
                
                

        probs_all_cat = torch.cat(probs_all, dim=0)      # [N, num_classes]
        max_indices = torch.argmax(probs_all_cat, dim=1) # [N]

        hard_label = torch.zeros_like(probs_all_cat, device=probs_all_cat.device)
        hard_label[torch.arange(hard_label.size(0)), max_indices] = 1
        
        # print("------------hard_label.cpu().numpy() -----------",hard_label.cpu().numpy() )

        # return hard_label.scpu().numpy()
        return hard_label.detach().cpu()  
    

class USE(object):
    def __init__(self, cache_path):
        super(USE, self).__init__()
        os.environ['TFHUB_CACHE_DIR'] = cache_path
        module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/3"
        self.embed = hub.Module(module_url)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.build_graph()
        self.sess.run([tf.global_variables_initializer(), tf.tables_initializer()])

    def build_graph(self):
        self.sts_input1 = tf.placeholder(tf.string, shape=(None))
        self.sts_input2 = tf.placeholder(tf.string, shape=(None))

        sts_encode1 = tf.nn.l2_normalize(self.embed(self.sts_input1), axis=1)
        sts_encode2 = tf.nn.l2_normalize(self.embed(self.sts_input2), axis=1)
        self.cosine_similarities = tf.reduce_sum(tf.multiply(sts_encode1, sts_encode2), axis=1)
        clip_cosine_similarities = tf.clip_by_value(self.cosine_similarities, -1.0, 1.0)
        self.sim_scores = 1.0 - tf.acos(clip_cosine_similarities)

    def semantic_sim(self, sents1, sents2):
        scores = self.sess.run(
            [self.sim_scores],
            feed_dict={
                self.sts_input1: sents1,
                self.sts_input2: sents2,
            })
        return scores

    def semantic_cosine(self, sents1, sents2):
        """Cosine similarity used by ELESSAttack semantic energy."""
        return self.sess.run(
            self.cosine_similarities,
            feed_dict={
                self.sts_input1: sents1,
                self.sts_input2: sents2,
            }
        )


def pick_most_similar_words_batch(src_words, sim_mat, idx2word, ret_count=10, threshold=0.):
    """
    embeddings is a matrix with (d, vocab_size)
    """
    sim_order = np.argsort(-sim_mat[src_words, :])[:, 1:1 + ret_count]
    sim_words, sim_values = [], []
    for idx, src_word in enumerate(src_words):
        sim_value = sim_mat[src_word][sim_order[idx]]
        mask = sim_value >= threshold
        sim_word, sim_value = sim_order[idx][mask], sim_value[mask]
        sim_word = [idx2word[id] for id in sim_word]
        sim_words.append(sim_word)
        sim_values.append(sim_value)
    return sim_words, sim_values


def read_data(filepath, data_size, target_model='infersent', lowercase=False, ignore_punctuation=False, stopwords=[]):
    """
    Read the premises, hypotheses and labels from some NLI dataset's
    file and return them in a dictionary. The file should be in the same
    form as SNLI's .txt files.

    Args:
        filepath: The path to a file containing some premises, hypotheses
            and labels that must be read. The file should be formatted in
            the same way as the SNLI (and MultiNLI) dataset.

    Returns:
        A dictionary containing three lists, one for the premises, one for
        the hypotheses, and one for the labels in the input data.
    """
    if target_model == 'bert':
        labeldict = {"contradiction": 0,
                      "entailment": 1,
                      "neutral": 2}
    else:
        labeldict = {"entailment": 0,
                     "neutral": 1,
                     "contradiction": 2}
    with open(filepath, 'r', encoding='utf8') as input_data:
        premises, hypotheses, labels = [], [], []

        # Translation tables to remove punctuation from strings.
        punct_table = str.maketrans({key: ' '
                                     for key in string.punctuation})

        for idx, line in enumerate(input_data):
            if idx >= data_size:
                break

            # Ensure that the line has enough columns
            if len(line) < 3:
                continue

            line = line.strip().split('\t')

            # Ignore sentences that have no gold label.
            if line[0] == '-':
                continue

            premise = line[1]
            hypothesis = line[2]

            if lowercase:
                premise = premise.lower()
                hypothesis = hypothesis.lower()

            if ignore_punctuation:
                premise = premise.translate(punct_table)
                hypothesis = hypothesis.translate(punct_table)

            # Each premise and hypothesis is split into a list of words.
            premises.append([w for w in premise.rstrip().split()
                             if w not in stopwords])
            hypotheses.append([w for w in hypothesis.rstrip().split()
                               if w not in stopwords])
            labels.append(labeldict[line[0]])

        return {"premises": premises,
                "hypotheses": hypotheses,
                "labels": labels}


class NLIDataset_ESIM(Dataset):
    """
    Dataset class for Natural Language Inference datasets.

    The class can be used to read preprocessed datasets where the premises,
    hypotheses and labels have been transformed to unique integer indices
    (this can be done with the 'preprocess_data' script in the 'scripts'
    folder of this repository).
    """

    def __init__(self,
                 worddict_path,
                 padding_idx=0,
                 bos="_BOS_",
                 eos="_EOS_"):
        """
        Args:
            data: A dictionary containing the preprocessed premises,
                hypotheses and labels of some dataset.
            padding_idx: An integer indicating the index being used for the
                padding token in the preprocessed data. Defaults to 0.
            max_premise_length: An integer indicating the maximum length
                accepted for the sequences in the premises. If set to None,
                the length of the longest premise in 'data' is used.
                Defaults to None.
            max_hypothesis_length: An integer indicating the maximum length
                accepted for the sequences in the hypotheses. If set to None,
                the length of the longest hypothesis in 'data' is used.
                Defaults to None.
        """
        self.bos = bos
        self.eos = eos
        self.padding_idx = padding_idx

        # build word dict
        with open(worddict_path, 'rb') as pkl:
            self.worddict = pickle.load(pkl)

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, index):
        return {
            "premise": self.data["premises"][index],
            "premise_length": min(self.premises_lengths[index],
                                  self.max_premise_length),
            "hypothesis": self.data["hypotheses"][index],
            "hypothesis_length": min(self.hypotheses_lengths[index],
                                     self.max_hypothesis_length)
        }

    def words_to_indices(self, sentence):
        """
        Transform the words in a sentence to their corresponding integer
        indices.

        Args:
            sentence: A list of words that must be transformed to indices.

        Returns:
            A list of indices.
        """
        indices = []
        # Include the beggining of sentence token at the start of the sentence
        # if one is defined.
        if self.bos:
            indices.append(self.worddict["_BOS_"])

        for word in sentence:
            if word in self.worddict:
                index = self.worddict[word]
            else:
                # Words absent from 'worddict' are treated as a special
                # out-of-vocabulary word (OOV).
                index = self.worddict['_OOV_']
            indices.append(index)
        # Add the end of sentence token at the end of the sentence if one
        # is defined.
        if self.eos:
            indices.append(self.worddict["_EOS_"])

        return indices

    def transform_to_indices(self, data):
        """
        Transform the words in the premises and hypotheses of a dataset, as
        well as their associated labels, to integer indices.

        Args:
            data: A dictionary containing lists of premises, hypotheses
                and labels, in the format returned by the 'read_data'
                method of the Preprocessor class.

        Returns:
            A dictionary containing the transformed premises, hypotheses and
            labels.
        """
        transformed_data = {"premises": [],
                            "hypotheses": []}

        for i, premise in enumerate(data['premises']):
            # Ignore sentences that have a label for which no index was
            # defined in 'labeldict'.

            indices = self.words_to_indices(premise)
            transformed_data["premises"].append(indices)

            indices = self.words_to_indices(data["hypotheses"][i])
            transformed_data["hypotheses"].append(indices)

        return transformed_data

    def transform_text(self, data):
        #         # standardize data format
        #         data = defaultdict(list)
        #         for hypothesis in hypotheses:
        #             data['premises'].append(premise)
        #             data['hypotheses'].append(hypothesis)

        # transform data into indices
        data = self.transform_to_indices(data)

        self.premises_lengths = [len(seq) for seq in data["premises"]]
        self.max_premise_length = max(self.premises_lengths)

        self.hypotheses_lengths = [len(seq) for seq in data["hypotheses"]]
        self.max_hypothesis_length = max(self.hypotheses_lengths)

        self.num_sequences = len(data["premises"])

        self.data = {
            "premises": torch.ones((self.num_sequences,
                                    self.max_premise_length),
                                   dtype=torch.long) * self.padding_idx,
            "hypotheses": torch.ones((self.num_sequences,
                                      self.max_hypothesis_length),
                                     dtype=torch.long) * self.padding_idx}

        for i, premise in enumerate(data["premises"]):
            end = min(len(premise), self.max_premise_length)
            self.data["premises"][i][:end] = torch.tensor(premise[:end])

            hypothesis = data["hypotheses"][i]
            end = min(len(hypothesis), self.max_hypothesis_length)
            self.data["hypotheses"][i][:end] = torch.tensor(hypothesis[:end])


class NLIDataset_InferSent(Dataset):
    """
    Dataset class for Natural Language Inference datasets.

    The class can be used to read preprocessed datasets where the premises,
    hypotheses and labels have been transformed to unique integer indices
    (this can be done with the 'preprocess_data' script in the 'scripts'
    folder of this repository).
    """

    def __init__(self,
                 embedding_path,
                 data,
                 word_emb_dim=200,
                 batch_size=32,
                 bos="<s>",
                 eos="</s>"):
        """
        Args:
            data: A dictionary containing the preprocessed premises,
                hypotheses and labels of some dataset.
            padding_idx: An integer indicating the index being used for the
                padding token in the preprocessed data. Defaults to 0.
            max_premise_length: An integer indicating the maximum length
                accepted for the sequences in the premises. If set to None,
                the length of the longest premise in 'data' is used.
                Defaults to None.
            max_hypothesis_length: An integer indicating the maximum length
                accepted for the sequences in the hypotheses. If set to None,
                the length of the longest hypothesis in 'data' is used.
                Defaults to None.
        """
        self.bos = bos
        self.eos = eos
        self.word_emb_dim = word_emb_dim
        self.batch_size = batch_size

        # build word dict
        self.word_vec = self.build_vocab(data['premises']+data['hypotheses'], embedding_path)

    def build_vocab(self, sentences, embedding_path):
        word_dict = self.get_word_dict(sentences)
        word_vec = self.get_embedding(word_dict, embedding_path)
        print('Vocab size : {0}'.format(len(word_vec)))
        return word_vec

    def get_word_dict(self, sentences):
        # create vocab of words
        word_dict = {}
        for sent in sentences:
            for word in sent:
                if word not in word_dict:
                    word_dict[word] = ''
        word_dict['<s>'] = ''
        word_dict['</s>'] = ''
        word_dict['<oov>'] = ''
        return word_dict

    def get_embedding(self, word_dict, embedding_path):
        # create word_vec with glove vectors
        word_vec = {}
        word_vec['<oov>'] = np.random.normal(size=(self.word_emb_dim))
        with open(embedding_path) as f:
            for line in f:
                word, vec = line.split(' ', 1)
                if word in word_dict:
                    word_vec[word] = np.array(list(map(float, vec.split())))
        print('Found {0}(/{1}) words with embedding vectors'.format(
            len(word_vec), len(word_dict)))
        return word_vec

    def get_batch(self, batch, word_vec, emb_dim=200):
        # sent in batch in decreasing order of lengths (bsize, max_len, word_dim)
        lengths = np.array([len(x) for x in batch])
        max_len = np.max(lengths)
        #         print(max_len)
        embed = np.zeros((max_len, len(batch), emb_dim))

        for i in range(len(batch)):
            for j in range(len(batch[i])):
                if batch[i][j] in word_vec:
                    embed[j, i, :] = word_vec[batch[i][j]]
                else:
                    embed[j, i, :] = word_vec['<oov>']
        #                     embed[j, i, :] = np.random.normal(size=(emb_dim))

        return torch.from_numpy(embed).float(), lengths

    def transform_text(self, data):
        # transform data into seq of embeddings
        premises = data['premises']
        hypotheses = data['hypotheses']

        # add bos and eos
        premises = [['<s>'] + premise + ['</s>'] for premise in premises]
        hypotheses = [['<s>'] + hypothese + ['</s>'] for hypothese in hypotheses]

        batches = []
        for stidx in range(0, len(premises), self.batch_size):
            # prepare batch
            s1_batch, s1_len = self.get_batch(premises[stidx:stidx + self.batch_size],
                                              self.word_vec, self.word_emb_dim)
            s2_batch, s2_len = self.get_batch(hypotheses[stidx:stidx + self.batch_size],
                                              self.word_vec, self.word_emb_dim)
            batches.append(((s1_batch, s1_len), (s2_batch, s2_len)))

        return batches


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids


class NLIDataset_BERT(Dataset):
    """
    Dataset class for Natural Language Inference datasets.

    The class can be used to read preprocessed datasets where the premises,
    hypotheses and labels have been transformed to unique integer indices
    (this can be done with the 'preprocess_data' script in the 'scripts'
    folder of this repository).
    """

    def __init__(self,
                 pretrained_dir,
                 max_seq_length=128,
                 batch_size=32):
        """
        Args:
            data: A dictionary containing the preprocessed premises,
                hypotheses and labels of some dataset.
            padding_idx: An integer indicating the index being used for the
                padding token in the preprocessed data. Defaults to 0.
            max_premise_length: An integer indicating the maximum length
                accepted for the sequences in the premises. If set to None,
                the length of the longest premise in 'data' is used.
                Defaults to None.
            max_hypothesis_length: An integer indicating the maximum length
                accepted for the sequences in the hypotheses. If set to None,
                the length of the longest hypothesis in 'data' is used.
                Defaults to None.
        """
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_dir, do_lower_case=True)
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size

    def _truncate_seq_pair(self, tokens_a, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""

        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()

    def convert_examples_to_features(self, examples, max_seq_length, tokenizer):
        """Loads a data file into a list of `InputBatch`s."""

        features = []
        for (ex_index, (text_a, text_b)) in enumerate(examples):
            tokens_a = tokenizer.tokenize(' '.join(text_a))

            tokens_b = None
            if text_b:
                tokens_b = tokenizer.tokenize(' '.join(text_b))
                # Modifies `tokens_a` and `tokens_b` in place so that the total
                # length is less than the specified length.
                # Account for [CLS], [SEP], [SEP] with "- 3"
                self._truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
            else:
                # Account for [CLS] and [SEP] with "- 2"
                if len(tokens_a) > max_seq_length - 2:
                    tokens_a = tokens_a[:(max_seq_length - 2)]

            tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
            segment_ids = [0] * len(tokens)

            if tokens_b:
                tokens += tokens_b + ["[SEP]"]
                segment_ids += [1] * (len(tokens_b) + 1)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding = [0] * (max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids))
        return features

    def transform_text(self, data):
        # transform data into seq of embeddings
        eval_features = self.convert_examples_to_features(list(zip(data['premises'], data['hypotheses'])),
                                                          self.max_seq_length, self.tokenizer)

        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)

        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=self.batch_size)

        return eval_dataloader



def _local_window_bounds(idx, len_text, window_size):
    """Return [start, end) bounds for a local semantic window centered at idx."""
    if len_text <= window_size:
        return 0, len_text

    half = (window_size - 1) // 2
    start = max(0, idx - half)
    end = min(len_text, idx + half + 1)

    # Keep a fixed-size window near sentence boundaries whenever possible.
    if end - start < window_size:
        if start == 0:
            end = min(len_text, window_size)
        elif end == len_text:
            start = max(0, len_text - window_size)
    return start, end


def _is_content_pos(tag):
    """ELESSAttack keeps nouns, verbs, adjectives and adverbs."""
    if tag is None:
        return False
    tag = str(tag).upper()
    return tag.startswith(('NN', 'VB', 'JJ', 'RB')) or tag in {'NOUN', 'VERB', 'ADJ', 'ADV'}


def attack_eless_nli(premise, hypothese, true_label, predictor, stop_words_set, word2idx, idx2word, cos_sim,
                     sim_predictor=None, import_score_threshold=-1., sim_score_threshold=0.5,
                     sim_score_window=15, synonym_num=50, batch_size=64, perturb_budget=0.40,
                     initial_temperature=0.10, cooling_factor=0.90, energy_zeta=0.0,
                     lambda_sem=0.03, use_ranking=True, use_local_semantics=True,
                     use_energy_search=True, max_query_budget=0, return_details=False):
    # Exact classification ELESS algorithm; NLI adaptation only wraps target-model queries.
    # Premise is fixed and only hypothesis tokens are perturbed.
    text_ls = list(hypothese)

    def _predict(candidate_hypotheses, batch_size=None):
        return predictor({
            'premises': [premise] * len(candidate_hypotheses),
            'hypotheses': candidate_hypotheses,
        })


    started = time.perf_counter()
    # Keep the same synonym search space as TextFooler.
    candidate_topk = int(synonym_num)

    # Revisit several high-importance positions after the first sweep.
    # This gives Metropolis acceptance a chance to escape a local optimum
    # instead of making only one proposal per token.
    revisit_topk = 10
    refine_steps = 3

    trajectory = []
    accepted_worse_moves = 0
    proposed_moves = 0
    accepted_moves = 0

    def normalize_label(x):
        return int(x.item()) if torch.is_tensor(x) else int(x)

    def _pack(text_out, n_changed, o_label, n_label, q, success=False,
              final_sim=1.0, final_energy=0.0, success_iter=-1):
        details = {
            'eligible': normalize_label(true_label) == normalize_label(o_label),
            'success': bool(success),
            'num_changed': int(n_changed),
            'perturb_ratio': float(n_changed / max(len(text_ls), 1)),
            'num_queries': int(q),
            'runtime_sec': float(time.perf_counter() - started),
            'final_similarity': float(final_sim),
            'final_energy': float(final_energy),
            'success_iteration': int(success_iter),
            'accepted_worse_moves': int(accepted_worse_moves),
            'accepted_moves': int(accepted_moves),
            'proposed_moves': int(proposed_moves),
            'trajectory': trajectory,
        }
        result = (text_out, n_changed, o_label, n_label, q)
        return result + (details,) if return_details else result

    if sim_predictor is None:
        raise ValueError('ELESS requires sim_predictor with semantic_cosine().')

    orig_probs = _predict([text_ls]).squeeze()
    orig_label = torch.argmax(orig_probs)
    orig_label_int = normalize_label(orig_label)
    orig_prob = float(orig_probs[orig_label].detach().cpu().item())
    num_queries = 1
    true_label_int = normalize_label(true_label)

    if true_label_int != orig_label_int:
        return _pack('', 0, orig_label, orig_label, num_queries, False, 1.0, orig_prob, -1)

    len_text = len(text_ls)
    if len_text == 0:
        return _pack('', 0, orig_label, orig_label, num_queries, False, 1.0, orig_prob, -1)

    effective_sim_threshold = float(sim_score_threshold)
    if len_text < sim_score_window:
        effective_sim_threshold = 0.1

    def changed_set(candidate):
        return {i for i, (a, b) in enumerate(zip(candidate, text_ls)) if a != b}

    def within_budget(candidate):
        return len(changed_set(candidate)) / float(max(len_text, 1)) <= float(perturb_budget)

    def full_similarity(candidate):
        return float(sim_predictor.semantic_cosine(
            [' '.join(text_ls)], [' '.join(candidate)]
        )[0])

    def calc_energy(score, similarity):
        return float(score) + float(lambda_sem) * (1.0 - float(similarity))

    def calc_search_energy(score, local_similarity):
        return float(score) + float(lambda_sem) * (1.0 - float(local_similarity))

    # Stage I: importance ranking
    pos_ls = criteria.get_pos(text_ls)
    candidate_positions = [i for i, word in enumerate(text_ls) if word not in stop_words_set]
    importance_scores = np.zeros(len_text, dtype=float)

    if use_ranking and candidate_positions:
        ranking_positions = list(candidate_positions)
        leave_1_texts = [
            text_ls[:i] + ['<oov>'] + text_ls[i + 1:]
            for i in ranking_positions
        ]

        if max_query_budget:
            remaining = max(0, int(max_query_budget) - num_queries)
            if remaining == 0:
                leave_1_texts, ranking_positions = [], []
            elif len(leave_1_texts) > remaining:
                leave_1_texts = leave_1_texts[:remaining]
                ranking_positions = ranking_positions[:remaining]

        if leave_1_texts:
            leave_1_probs = _predict(leave_1_texts, batch_size=batch_size)
            num_queries += len(leave_1_texts)
            leave_1_probs_argmax = torch.argmax(leave_1_probs, dim=-1)

            vals = (
                orig_probs[orig_label] - leave_1_probs[:, orig_label]
                + (leave_1_probs_argmax != orig_label).float()
                * (
                    leave_1_probs.max(dim=-1)[0]
                    - torch.index_select(orig_probs, 0, leave_1_probs_argmax)
                )
            ).detach().cpu().numpy()

            for i, v in zip(ranking_positions, vals):
                importance_scores[i] = float(v)

        words_perturb = [
            (i, text_ls[i])
            for i in sorted(candidate_positions, key=lambda j: importance_scores[j], reverse=True)
            if importance_scores[i] > import_score_threshold
        ]
    else:
        words_perturb = [(i, text_ls[i]) for i in candidate_positions]

    if not words_perturb:
        return _pack(' '.join(text_ls), 0, orig_label, orig_label,
                     num_queries, False, 1.0, orig_prob, -1)

    # Do not truncate attack positions here.  TextFooler considers every
    # eligible position; query control should be handled by max_query_budget.

    # Stage II: synonym sets
    valid_words_perturb = [(idx, word) for idx, word in words_perturb if word in word2idx]
    if not valid_words_perturb:
        return _pack(' '.join(text_ls), 0, orig_label, orig_label,
                     num_queries, False, 1.0, orig_prob, -1)

    words_perturb_idx = [word2idx[word] for _, word in valid_words_perturb]
    synonym_words, _ = pick_most_similar_words_batch(
        words_perturb_idx, cos_sim, idx2word, synonym_num, 0.5
    )

    synonyms_all = []
    for (idx, _), synonyms in zip(valid_words_perturb, synonym_words):
        if synonyms:
            synonyms_all.append((idx, list(synonyms[:candidate_topk])))

    if not synonyms_all:
        return _pack(' '.join(text_ls), 0, orig_label, orig_label,
                     num_queries, False, 1.0, orig_prob, -1)

    # Stage III: fast energy search
    text_prime = text_ls[:]
    changed_positions = set()
    temperature = max(float(initial_temperature), 1e-8)
    cooling_factor = min(max(float(cooling_factor), 1e-6), 1.0)
    zeta_initialized = False

    current_orig_score = orig_prob
    # IMPORTANT: state energy must be evaluated in a fixed reference frame.
    # We therefore track semantic similarity to the ORIGINAL text, not
    # current->candidate transition similarity from unrelated local windows.
    current_state_sim = 1.0
    current_search_energy = calc_search_energy(current_orig_score, current_state_sim)

    # Long-text stabilization: cool according to accepted/state-changing moves
    # rather than every visited token position. This prevents the temperature
    # from collapsing merely because the document contains many candidate words.
    cooling_steps = 0

    def adaptive_temperature():
        return max(float(initial_temperature) * (float(cooling_factor) ** cooling_steps), 1e-8)

    # Keep the best state ever visited.  Simulated annealing may accept a
    # temporarily worse move, so returning only the last state can discard a
    # much stronger adversarial direction discovered earlier.
    best_text = text_prime[:]
    best_orig_score = float(orig_prob)
    best_search_energy = float(current_search_energy)

    iteration = 0
    first_success_iteration = -1

    trajectory.append({
        'iteration': 0,
        'token_index': -1,
        'energy': float(current_search_energy),
        'similarity': 1.0,
        'orig_class_score': float(current_orig_score),
        'perturb_ratio': 0.0,
        'temperature': float(temperature),
        'accepted': 1,
        'worse_move': 0,
        'queries': int(num_queries),
        'success': 0,
    })

    def process_position(idx, synonyms, allow_original=False):
        nonlocal num_queries, temperature, initial_temperature
        nonlocal current_orig_score, current_state_sim, current_search_energy
        nonlocal text_prime, changed_positions
        nonlocal best_text, best_orig_score, best_search_energy
        nonlocal proposed_moves, accepted_moves, accepted_worse_moves
        nonlocal zeta_initialized, iteration, cooling_steps

        result = {
            'success': False, 'accepted': False,
            'success_text': None, 'success_label': None,
            'success_score': None, 'success_sim': None,
            'success_energy': None, 'token_index': idx,
        }

        candidate_words = list(synonyms)
        if allow_original and text_prime[idx] != text_ls[idx] and text_ls[idx] not in candidate_words:
            candidate_words = [text_ls[idx]] + candidate_words

        candidate_words = [w for w in candidate_words if w != text_prime[idx]][:candidate_topk]
        if not candidate_words:
            return result

        new_texts = []
        for synonym in candidate_words:
            candidate = text_prime[:]
            candidate[idx] = synonym
            if within_budget(candidate):
                new_texts.append(candidate)

        if not new_texts:
            return result

        if max_query_budget:
            remaining = max(0, int(max_query_budget) - num_queries)
            if remaining <= 0:
                return result
            new_texts = new_texts[:remaining]

        if not new_texts:
            return result

        new_probs = _predict(new_texts, batch_size=batch_size)
        num_queries += len(new_texts)

        if len(new_probs.shape) < 2:
            new_probs = new_probs.unsqueeze(0)

        pred_labels = torch.argmax(new_probs, dim=-1).detach().cpu().numpy()
        orig_class_scores = new_probs[:, orig_label].detach().cpu().numpy()

        # POS
        synonyms_pos_ls = []
        for new_text in new_texts:
            if len(new_text) > 10:
                local_start = max(idx - 4, 0)
                local_end = min(idx + 5, len(new_text))
                local_pos = criteria.get_pos(new_text[local_start:local_end])
                local_idx = idx - local_start
                synonyms_pos_ls.append(
                    local_pos[local_idx] if 0 <= local_idx < len(local_pos) else pos_ls[idx]
                )
            else:
                candidate_pos = criteria.get_pos(new_text)
                synonyms_pos_ls.append(
                    candidate_pos[idx] if idx < len(candidate_pos) else pos_ls[idx]
                )

        pos_mask = np.asarray(criteria.pos_filter(pos_ls[idx], synonyms_pos_ls), dtype=bool)

        # Local semantic similarity is used ONLY as a per-position hard constraint.
        # The reference is original-window -> candidate-window for this position;
        # these values are not compared as energies across different positions.
        text_range_min, text_range_max = _local_window_bounds(idx, len_text, sim_score_window)
        original_window = ' '.join(text_ls[text_range_min:text_range_max])
        candidate_windows = [
            ' '.join(candidate[text_range_min:text_range_max])
            for candidate in new_texts
        ]

        candidate_local_sims = np.asarray(
            sim_predictor.semantic_cosine(
                [original_window] * len(new_texts),
                candidate_windows
            ),
            dtype=float
        )

        semantic_mask = (
            candidate_local_sims >= effective_sim_threshold
            if use_local_semantics
            else np.ones(len(new_texts), dtype=bool)
        )

        valid_indices = np.where(pos_mask & semantic_mask)[0]
        if len(valid_indices) == 0:
            return result

        # Successful candidates first
        success_indices = valid_indices[
            pred_labels[valid_indices] != orig_label_int
        ]

        if len(success_indices) > 0:
            success_scores = orig_class_scores[success_indices]
            success_local_sims = candidate_local_sims[success_indices]

            # Among already successful candidates, prioritize attack strength first;
            # semantic similarity is a small tie-breaker because all candidates have
            # already passed the semantic threshold.
            success_rank = success_scores + float(lambda_sem) * (1.0 - success_local_sims)
            local_best = int(np.argmin(success_rank))
            best_idx = int(success_indices[local_best])

            best_text = new_texts[best_idx]
            best_score = float(orig_class_scores[best_idx])
            best_label = int(pred_labels[best_idx])

            # Full USE only for final winner
            best_sim = full_similarity(best_text)
            best_energy = calc_energy(best_score, best_sim)

            iteration += 1
            trajectory.append({
                'iteration': iteration,
                'token_index': int(idx),
                'energy': float(best_energy),
                'similarity': float(best_sim),
                'orig_class_score': float(best_score),
                'perturb_ratio': len(changed_set(best_text)) / float(len_text),
                'temperature': float(temperature),
                'accepted': 1,
                'worse_move': int(best_energy > current_search_energy),
                'queries': int(num_queries),
                'success': 1,
            })

            result.update({
                'success': True,
                'accepted': True,
                'success_text': best_text,
                'success_label': best_label,
                'success_score': best_score,
                'success_sim': best_sim,
                'success_energy': best_energy,
            })
            return result

        # Greedy exploitation over the complete valid candidate set.
        # We already know every candidate's classification score and local
        # semantic similarity, so randomly sampling a weaker candidate here
        # unnecessarily sacrifices TextFooler's strongest behavior.  The
        # Metropolis mechanism is kept below for accepting a temporarily worse
        # *best available* move and for revisit-based local-optimum escape.
        energy_indices = valid_indices
        candidate_scores = orig_class_scores[energy_indices]
        candidate_sims = candidate_local_sims[energy_indices]
        candidate_energies = candidate_scores + float(lambda_sem) * (1.0 - candidate_sims)

        # Preserve TextFooler's strongest behavior on long texts: primarily choose
        # the synonym that most decreases the original-class confidence. Joint
        # energy is only used as a deterministic tie-breaker.
        min_score = float(np.min(candidate_scores))
        near_best = np.where(candidate_scores <= min_score + 1e-12)[0]
        if len(near_best) == 1:
            local_choice = int(near_best[0])
        else:
            local_choice = int(near_best[np.argmin(candidate_energies[near_best])])

        proposal_idx = int(energy_indices[local_choice])
        proposal_text = new_texts[proposal_idx]
        proposal_score = float(orig_class_scores[proposal_idx])

        proposal_local_sim = float(candidate_local_sims[proposal_idx])
        proposal_energy = calc_search_energy(proposal_score, proposal_local_sim)

        delta_e = proposal_energy - current_search_energy
        proposed_moves += 1

        temperature = adaptive_temperature()
        if energy_zeta and energy_zeta > 1.0 and delta_e > 0 and not zeta_initialized:
            # Optional one-time calibration from the first uphill move.
            initial_temperature = max(delta_e / math.log(float(energy_zeta)), 1e-8)
            temperature = adaptive_temperature()
            zeta_initialized = True

        if use_energy_search:
            accept = (
                True if delta_e <= 0
                else random.random() < min(
                    1.0, math.exp(-delta_e / max(temperature, 1e-8))
                )
            )
        else:
            accept = delta_e <= 0

        iteration += 1
        worse_move = int(delta_e > 0)

        if accept:
            text_prime = proposal_text
            changed_positions = changed_set(text_prime)
            current_orig_score = proposal_score
            current_state_sim = proposal_local_sim
            current_search_energy = proposal_energy
            accepted_moves += 1
            cooling_steps += 1

            if worse_move:
                accepted_worse_moves += 1

            # Preserve the strongest state visited so far.  Classification
            # confidence is the primary attack objective; joint energy is used
            # as a tie-breaker so semantic quality is still respected.
            if (proposal_score < best_orig_score - 1e-12 or
                    (abs(proposal_score - best_orig_score) <= 1e-12 and
                     proposal_energy < best_search_energy)):
                best_text = proposal_text[:]
                best_orig_score = float(proposal_score)
                best_search_energy = float(proposal_energy)

            result['accepted'] = True

        trajectory.append({
            'iteration': iteration,
            'token_index': int(idx),
            'energy': float(current_search_energy),
            'similarity': float(current_state_sim),
            'orig_class_score': float(current_orig_score),
            'perturb_ratio': len(changed_positions) / float(len_text),
            'temperature': float(temperature),
            'accepted': int(accept),
            'worse_move': worse_move,
            'queries': int(num_queries),
            'success': 0,
        })

        temperature = adaptive_temperature()
        return result

    def refine_success(success_text, success_label_int, success_score,
                       success_sim, success_energy):
        nonlocal num_queries, iteration

        best_text = success_text[:]
        best_label = int(success_label_int)
        best_score = float(success_score)
        best_sim = float(success_sim)
        best_energy = float(success_energy)

        if refine_steps <= 0:
            return best_text, best_label, best_score, best_sim, best_energy

        modified_positions = list(changed_set(best_text))
        if not modified_positions:
            return best_text, best_label, best_score, best_sim, best_energy

        modified_positions.sort(key=lambda i: importance_scores[i])
        attempts = 0

        for idx in modified_positions:
            if attempts >= refine_steps:
                break
            if max_query_budget and num_queries >= int(max_query_budget):
                break

            candidate = best_text[:]
            candidate[idx] = text_ls[idx]

            if candidate == best_text:
                continue

            probs = _predict([candidate]).squeeze()
            num_queries += 1
            attempts += 1
            label = int(torch.argmax(probs).item())

            if label != orig_label_int:
                score = float(probs[orig_label].detach().cpu().item())
                sim = full_similarity(candidate)
                e = calc_energy(score, sim)

                best_text = candidate
                best_label = label
                best_score = score
                best_sim = sim
                best_energy = e

                iteration += 1
                trajectory.append({
                    'iteration': iteration,
                    'token_index': int(idx),
                    'energy': float(best_energy),
                    'similarity': float(best_sim),
                    'orig_class_score': float(best_score),
                    'perturb_ratio': len(changed_set(best_text)) / float(len_text),
                    'temperature': float(temperature),
                    'accepted': 1,
                    'worse_move': 0,
                    'queries': int(num_queries),
                    'success': 1,
                })

        return best_text, best_label, best_score, best_sim, best_energy

    def return_success(out):
        nonlocal first_success_iteration

        if first_success_iteration < 0:
            first_success_iteration = iteration

        best_text, best_label, _, best_sim, best_energy = refine_success(
            out['success_text'],
            out['success_label'],
            out['success_score'],
            out['success_sim'],
            out['success_energy']
        )

        final_changed = changed_set(best_text)

        if torch.is_tensor(orig_label):
            final_label = torch.tensor(
                best_label,
                dtype=orig_label.dtype,
                device=orig_label.device
            )
        else:
            final_label = best_label

        return _pack(
            ' '.join(best_text),
            len(final_changed),
            orig_label,
            final_label,
            num_queries,
            True,
            best_sim,
            best_energy,
            first_success_iteration
        )

    # First sweep
    for idx, synonyms in synonyms_all:
        if max_query_budget and num_queries >= int(max_query_budget):
            break

        out = process_position(idx, synonyms, allow_original=False)
        if out['success']:
            return return_success(out)

    # Revisit several important positions.  This is essential for the
    # energy-search variant: a temporarily worse accepted move now has a later
    # opportunity to lead into a better basin instead of ending exploration.
    if revisit_topk > 0:
        for idx, synonyms in synonyms_all[:min(revisit_topk, len(synonyms_all))]:
            if max_query_budget and num_queries >= int(max_query_budget):
                break

            out = process_position(idx, synonyms, allow_original=True)
            if out['success']:
                return return_success(out)

    # Final prediction uses the best state visited, not merely the last
    # simulated-annealing state.
    text_prime = best_text[:]
    changed_positions = changed_set(text_prime)

    if not max_query_budget or num_queries < int(max_query_budget):
        final_probs = _predict([text_prime]).squeeze()
        num_queries += 1
        final_label = torch.argmax(final_probs)
        final_orig_score = float(final_probs[orig_label].detach().cpu().item())
    else:
        # best_text was evaluated when it was proposed, so its stored original
        # class score is still valid when the target-model query budget is full.
        final_orig_score = best_orig_score
        final_label = orig_label

    final_label_int = normalize_label(final_label)
    success = final_label_int != orig_label_int

    final_sim = full_similarity(text_prime)
    final_energy = calc_energy(final_orig_score, final_sim)

    if success:
        out = {
            'success_text': text_prime,
            'success_label': final_label_int,
            'success_score': final_orig_score,
            'success_sim': final_sim,
            'success_energy': final_energy,
        }
        return return_success(out)

    changed_positions = changed_set(text_prime)

    return _pack(
        ' '.join(text_prime),
        len(changed_positions),
        orig_label,
        final_label,
        num_queries,
        False,
        final_sim,
        final_energy,
        -1
    )


def TextFooler_bert(premise, hypothese, true_label, predictor, stop_words_set, word2idx, idx2word, cos_sim, sim_predictor=None,
           import_score_threshold=-1., sim_score_threshold=0.5, sim_score_window=15, synonym_num=50, batch_size=32):



    orig_probs = predictor({'premises': [premise], 'hypotheses': [hypothese]}).squeeze()
    orig_label = torch.argmax(orig_probs, dim=-1).item()
    orig_prob = orig_probs.max()


    if true_label != orig_label:
        return '', 0, orig_label, orig_label, 0
    else:
        len_text = len(hypothese)
        if len_text < sim_score_window:
            sim_score_threshold = 0.1  # shut down the similarity thresholding function
        half_sim_score_window = (sim_score_window - 1) // 2
        num_queries = 1

        # get the pos and verb tense info
        pos_ls = criteria.get_pos(hypothese)

        # get importance score
        leave_1_texts = [hypothese[:ii]+['<oov>']+hypothese[min(ii+1, len_text):] for ii in range(len_text)]
        leave_1_probs = predictor({'premises':[premise]*len_text, 'hypotheses': leave_1_texts})
        num_queries += len(leave_1_texts)
        leave_1_probs_argmax = torch.argmax(leave_1_probs, dim=-1)
        import_scores = (orig_prob - leave_1_probs[:, orig_label] + (leave_1_probs_argmax != orig_label).float() * (
                leave_1_probs.max(dim=-1)[0] - torch.index_select(orig_probs, 0,
                                                                  leave_1_probs_argmax))).data.cpu().numpy()

        # get words to perturb ranked by importance scorefor word in words_perturb
        words_perturb = []
        for idx, score in sorted(enumerate(import_scores), key=lambda x: x[1], reverse=True):
            if score > import_score_threshold and hypothese[idx] not in stop_words_set:
                words_perturb.append((idx, hypothese[idx]))

        # find synonyms
        words_perturb_idx = [word2idx[word] for idx, word in words_perturb if word in word2idx]
        synonym_words, _ = pick_most_similar_words_batch(words_perturb_idx, cos_sim, idx2word, synonym_num, 0.5)
        synonyms_all = []
        for idx, word in words_perturb:
            if word in word2idx:
                synonyms = synonym_words.pop(0)
                if synonyms:
                    synonyms_all.append((idx, synonyms))

        # start replacing and attacking
        text_prime = hypothese[:]
        text_cache = text_prime[:]
        num_changed = 0
        for idx, synonyms in synonyms_all:
            new_texts = [text_prime[:idx] + [synonym] + text_prime[min(idx + 1, len_text):] for synonym in synonyms]
            new_probs = predictor({'premises': [premise] * len(synonyms), 'hypotheses': new_texts})

            # compute semantic similarity
            if idx >= half_sim_score_window and len_text - idx - 1 >= half_sim_score_window:
                text_range_min = idx - half_sim_score_window
                text_range_max = idx + half_sim_score_window + 1
            elif idx < half_sim_score_window and len_text - idx - 1 >= half_sim_score_window:
                text_range_min = 0
                text_range_max = sim_score_window
            elif idx >= half_sim_score_window and len_text - idx - 1 < half_sim_score_window:
                text_range_min = len_text - sim_score_window
                text_range_max = len_text
            else:
                text_range_min = 0
                text_range_max = len_text
            semantic_sims = \
                sim_predictor.semantic_sim([' '.join(text_cache[text_range_min:text_range_max])] * len(new_texts),
                                           list(map(lambda x: ' '.join(x[text_range_min:text_range_max]), new_texts)))[0]

            num_queries += len(new_texts)
            if len(new_probs.shape) < 2:
                new_probs = new_probs.unsqueeze(0)
            new_probs_mask = (orig_label != torch.argmax(new_probs, dim=-1)).data.cpu().numpy()
            # prevent bad synonyms
            new_probs_mask *= (semantic_sims >= sim_score_threshold)
            # prevent incompatible pos
            synonyms_pos_ls = [criteria.get_pos(new_text[max(idx - 4, 0):idx + 5])[min(4, idx)]
                               if len(new_text) > 10 else criteria.get_pos(new_text)[idx] for new_text in new_texts]
            pos_mask = np.array(criteria.pos_filter(pos_ls[idx], synonyms_pos_ls))
            new_probs_mask *= pos_mask

            if np.sum(new_probs_mask) > 0:
                text_prime[idx] = synonyms[(new_probs_mask * semantic_sims).argmax()]
                num_changed += 1
                break
            else:
                new_label_probs = new_probs[:, orig_label] + torch.from_numpy(
                    (semantic_sims < sim_score_threshold) + (1 - pos_mask).astype(float)).float().cuda()
                new_label_prob_min, new_label_prob_argmin = torch.min(new_label_probs, dim=-1)
                if new_label_prob_min < orig_prob:
                    text_prime[idx] = synonyms[new_label_prob_argmin]
                    num_changed += 1
            text_cache = text_prime[:]

        return ' '.join(text_prime), num_changed, orig_label, \
            torch.argmax(predictor({'premises':[premise], 'hypotheses': [text_prime]}), dim=-1).item(), num_queries






def attack(premise, hypothese, true_label, predictor, stop_words_set, word2idx, idx2word, cos_sim, sim_predictor=None,
                 import_score_threshold=-1., sim_score_threshold=0.5, sim_score_window=15, synonym_num=50, batch_size=32):



    orig_probs = predictor({'premises': [premise], 'hypotheses': [hypothese]}).squeeze()
    orig_label = torch.argmax(orig_probs, dim=-1).item()
    orig_prob = orig_probs.max()


    if true_label != orig_label:
        return '', 0, orig_label, orig_label, 0
    else:

        len_text = len(premise)
        if len_text > sim_score_window:
            half_sim_score_window = sim_score_window
        else:
            half_sim_score_window = (sim_score_window - 1) // 2
        num_queries = 1


        print("---------------------------")

        # get the pos and verb tense info
        pos_ls = criteria.get_pos(hypothese)

        # print("------------pos_ls---------------",pos_ls)

        # get importance score
        leave_1_texts = [hypothese[:ii]+[' ']+hypothese[min(ii+1,len_text):] for ii in range(len_text)]

        leave_1_probs = predictor({'premises':[premise]*len_text, 'hypotheses': leave_1_texts})

        num_queries += len(leave_1_texts)


        # Calculate simple importance scores based on change in probability of the original label
        affect_words = (orig_prob - leave_1_probs[:, orig_label]).data.cpu().numpy()


        # Select the top 50 words by simple importance score, excluding stopwords
        words_perturb = []
        score_indices = sorted(enumerate(affect_words), key=lambda x: x[1], reverse=True)
        for idx, score in score_indices:
            if idx < len(hypothese) and  score > 0 and hypothese[idx] not in stop_words_set:  # Assuming we only consider negative impact
                words_perturb.append((idx, hypothese[idx]))

        temperature = 0.1
        delay_temperature = 0.002
        probability_threshold = 1000
        liter_N=50
        # print("----------temperature---------------", temperature)

        # find synonyms   每个句子中的idx后面跟着同义词列表
        words_perturb_idx = [word2idx[word] for idx, word in words_perturb if word in word2idx]
        synonym_words, _ = pick_most_similar_words_batch(words_perturb_idx, cos_sim, idx2word, synonym_num, 0.)
        synonyms_all = []
        for idx, word in words_perturb:
            if word in word2idx:
                synonyms = synonym_words.pop(0)
                if synonyms:
                    synonyms_all.append((idx, synonyms))

        # start replacing and attacking
        text_prime = hypothese[:]
        text_cache = text_prime[:]
        num_changed = 0


        semantic_sims = \
            sim_predictor.semantic_sim([' '.join(text_cache[0:15])] * len(text_prime),
                                       list(map(lambda x: ' '.join(x[0:15]), text_prime)))[0]


        for idx, synonyms in synonyms_all:
            print("------------idx: ---------------",idx)
            new_texts = [text_prime[:idx] + [synonym] + text_prime[min(idx + 1, len_text):] for synonym in synonyms]
            new_probs = predictor({'premises': [premise]*len(synonyms), 'hypotheses': new_texts})

            #同义词数量的替换生成的句子和原有句子的相似度
            if idx >= half_sim_score_window and len_text - idx - 1 >= half_sim_score_window:
                text_range_min = idx - half_sim_score_window
                text_range_max = idx + half_sim_score_window + 1
            elif idx < half_sim_score_window and len_text - idx - 1 >= half_sim_score_window:
                text_range_min = 0
                text_range_max = sim_score_window
            elif idx >= half_sim_score_window and len_text - idx - 1 < half_sim_score_window:
                text_range_min = len_text - sim_score_window
                text_range_max = len_text
            else:
                text_range_min = 0
                text_range_max = len_text
            old_semantic_sims = semantic_sims
            semantic_sims = \
                sim_predictor.semantic_sim([' '.join(text_cache[text_range_min:text_range_max])] * len(new_texts),
                                           list(map(lambda x: ' '.join(x[text_range_min:text_range_max]), new_texts)))[0]

            num_queries += len(new_texts)

            if len(new_probs.shape) < 2:
                new_probs = new_probs.unsqueeze(0)

            new_probs_mask = (orig_label != torch.argmax(new_probs, dim=-1)).data.cpu().numpy()


            # 过滤单词
            synonyms_pos_ls = [criteria.get_pos(new_text)[idx] for new_text in new_texts]

            pos_mask = np.array(criteria.pos_filter(pos_ls[idx], synonyms_pos_ls))
            new_probs_mask *= pos_mask

            Energy_change = (sum(semantic_sims) / len(semantic_sims)) -  (sum(old_semantic_sims) / len(old_semantic_sims))



            # 替换的条件:如果多个替换攻击成功，则选同义词编号最小的
            if np.sum(new_probs_mask) > 0:
                text_prime[idx] = synonyms[(new_probs_mask).argmax()]
                num_changed += 1
                break
            # 有一定概率接受一个失败的攻击，使用最近的同义词减轻损失
            elif  math.exp( - Energy_change / temperature ) > probability_threshold :
                text_prime[idx] = " "
                num_changed += 1
                if  temperature > 0.01:
                    temperature = temperature-delay_temperature
            # 否则
            else:

                new_label_probs = new_probs[:, orig_label] + torch.from_numpy((1 - pos_mask).astype(float)).float().cuda()
                new_label_prob_min, new_label_prob_argmin = torch.min(new_label_probs, dim=-1)
                text_prime[idx] = synonyms[new_label_prob_argmin]
                num_changed += 1
            text_cache = text_prime[:]

            # print("---------------------------premise------------------------------",premise)

        return ' '.join(text_prime), num_changed, orig_label, \
            torch.argmax(predictor({'premises':[premise], 'hypotheses': [text_prime]}), dim=-1).item(), num_queries



def TextFooler_esmi(premise, hypothese, true_label, predictor, stop_words_set, word2idx, idx2word, cos_sim, sim_predictor=None,
                import_score_threshold=-1., sim_score_threshold=0.5, sim_score_window=15, synonym_num=50, batch_size=32):



    orig_probs = predictor({'premises': [premise], 'hypotheses': [hypothese]}).squeeze()
    orig_label = torch.argmax(orig_probs, dim=-1).item()
    orig_prob = orig_probs.max()


    if true_label != orig_label:
        return '', 0, orig_label, orig_label, 0
    else:
        len_text = len(hypothese)
        if len_text < sim_score_window:
            sim_score_threshold = 0.1  # shut down the similarity thresholding function
        half_sim_score_window = (sim_score_window - 1) // 2
        num_queries = 1

        # get the pos and verb tense info
        pos_ls = criteria.get_pos(hypothese)

        # get importance score
        leave_1_texts = [hypothese[:ii]+['<oov>']+hypothese[min(ii+1, len_text):] for ii in range(len_text)]
        leave_1_probs = predictor({'premises':[premise]*len_text, 'hypotheses': leave_1_texts})
        num_queries += len(leave_1_texts)
        leave_1_probs_argmax = torch.argmax(leave_1_probs, dim=-1)
        import_scores = (orig_prob - leave_1_probs[:, orig_label] + (leave_1_probs_argmax != orig_label).float() * (
                leave_1_probs.max(dim=-1)[0] - torch.index_select(orig_probs, 0,
                                                                  leave_1_probs_argmax))).data.cpu().numpy()

        # get words to perturb ranked by importance scorefor word in words_perturb
        words_perturb = []
        for idx, score in sorted(enumerate(import_scores), key=lambda x: x[1], reverse=True):
            if score > import_score_threshold and hypothese[idx] not in stop_words_set:
                words_perturb.append((idx, hypothese[idx]))

        # find synonyms
        words_perturb_idx = [word2idx[word] for idx, word in words_perturb if word in word2idx]
        synonym_words, _ = pick_most_similar_words_batch(words_perturb_idx, cos_sim, idx2word, synonym_num, 0.5)
        synonyms_all = []
        for idx, word in words_perturb:
            if word in word2idx:
                synonyms = synonym_words.pop(0)
                if synonyms:
                    synonyms_all.append((idx, synonyms))

        # start replacing and attacking
        text_prime = hypothese[:]
        text_cache = text_prime[:]
        num_changed = 0
        for idx, synonyms in synonyms_all:
            new_texts = [text_prime[:idx] + [synonym] + text_prime[min(idx + 1, len_text):] for synonym in synonyms]
            new_probs = predictor({'premises': [premise] * len(synonyms), 'hypotheses': new_texts})

            # compute semantic similarity
            if idx >= half_sim_score_window and len_text - idx - 1 >= half_sim_score_window:
                text_range_min = idx - half_sim_score_window
                text_range_max = idx + half_sim_score_window + 1
            elif idx < half_sim_score_window and len_text - idx - 1 >= half_sim_score_window:
                text_range_min = 0
                text_range_max = sim_score_window
            elif idx >= half_sim_score_window and len_text - idx - 1 < half_sim_score_window:
                text_range_min = len_text - sim_score_window
                text_range_max = len_text
            else:
                text_range_min = 0
                text_range_max = len_text
            semantic_sims = \
                sim_predictor.semantic_sim([' '.join(text_cache[text_range_min:text_range_max])] * len(new_texts),
                                           list(map(lambda x: ' '.join(x[text_range_min:text_range_max]), new_texts)))[0]

            num_queries += len(new_texts)
            if len(new_probs.shape) < 2:
                new_probs = new_probs.unsqueeze(0)
            new_probs_mask = (orig_label != torch.argmax(new_probs, dim=-1)).data.cpu().numpy()
            # prevent bad synonyms
            new_probs_mask *= (semantic_sims >= sim_score_threshold)
            # prevent incompatible pos
            synonyms_pos_ls = [criteria.get_pos(new_text[max(idx - 4, 0):idx + 5])[min(4, idx)]
                               if len(new_text) > 10 else criteria.get_pos(new_text)[idx] for new_text in new_texts]
            pos_mask = np.array(criteria.pos_filter(pos_ls[idx], synonyms_pos_ls))
            new_probs_mask *= pos_mask

            if np.sum(new_probs_mask) > 0:
                text_prime[idx] = synonyms[(new_probs_mask * semantic_sims).argmax()]
                num_changed += 1
                break
            else:
                new_label_probs = new_probs[:, orig_label] + torch.from_numpy(
                    (semantic_sims < sim_score_threshold) + (1 - pos_mask).astype(float)).float()
                new_label_prob_min, new_label_prob_argmin = torch.min(new_label_probs, dim=-1)
                if new_label_prob_min < orig_prob:
                    text_prime[idx] = synonyms[new_label_prob_argmin]
                    num_changed += 1
            text_cache = text_prime[:]

        return ' '.join(text_prime), num_changed, orig_label, \
            torch.argmax(predictor({'premises':[premise], 'hypotheses': [text_prime]}), dim=-1).item(), num_queries






def attack_esmi(premise, hypothese, true_label, predictor, stop_words_set, word2idx, idx2word, cos_sim, sim_predictor=None,
           import_score_threshold=-1., sim_score_threshold=0.5, sim_score_window=15, synonym_num=50, batch_size=32):

    orig_probs = predictor({'premises': [premise], 'hypotheses': [hypothese]}).squeeze()
    orig_label = torch.argmax(orig_probs, dim=-1).item()
    orig_prob = orig_probs.max()


    if true_label != orig_label:
        return '', 0, orig_label, orig_label, 0
    else:

        len_text = len(premise)
        if len_text > sim_score_window:
            half_sim_score_window = sim_score_window
        else:
            half_sim_score_window = (sim_score_window - 1) // 2
        num_queries = 1


        print("---------------------------")

        # get the pos and verb tense info
        pos_ls = criteria.get_pos(hypothese)

        # print("------------pos_ls---------------",pos_ls)

        # get importance score
        leave_1_texts = [hypothese[:ii]+[' ']+hypothese[min(ii+1,len_text):] for ii in range(len_text)]

        leave_1_probs = predictor({'premises':[premise]*len_text, 'hypotheses': leave_1_texts})

        num_queries += len(leave_1_texts)


        # Calculate simple importance scores based on change in probability of the original label
        affect_words = (orig_prob - leave_1_probs[:, orig_label]).data.cpu().numpy()


        # Select the top 50 words by simple importance score, excluding stopwords
        words_perturb = []
        score_indices = sorted(enumerate(affect_words), key=lambda x: x[1], reverse=True)
        for idx, score in score_indices:
            if idx < len(hypothese) and  score > 0 and hypothese[idx] not in stop_words_set:  # Assuming we only consider negative impact
                words_perturb.append((idx, hypothese[idx]))

        temperature = 0.1
        delay_temperature = 0.002
        probability_threshold = 1000
        liter_N=50
        # print("----------temperature---------------", temperature)

        # find synonyms   每个句子中的idx后面跟着同义词列表
        words_perturb_idx = [word2idx[word] for idx, word in words_perturb if word in word2idx]
        synonym_words, _ = pick_most_similar_words_batch(words_perturb_idx, cos_sim, idx2word, synonym_num, 0.)
        synonyms_all = []
        for idx, word in words_perturb:
            if word in word2idx:
                synonyms = synonym_words.pop(0)
                if synonyms:
                    synonyms_all.append((idx, synonyms))

        # start replacing and attacking
        text_prime = hypothese[:]
        text_cache = text_prime[:]
        num_changed = 0


        semantic_sims = \
            sim_predictor.semantic_sim([' '.join(text_cache[0:15])] * len(text_prime),
                                       list(map(lambda x: ' '.join(x[0:15]), text_prime)))[0]


        for idx, synonyms in synonyms_all:
            print("------------idx: ---------------",idx)
            new_texts = [text_prime[:idx] + [synonym] + text_prime[min(idx + 1, len_text):] for synonym in synonyms]
            new_probs = predictor({'premises': [premise]*len(synonyms), 'hypotheses': new_texts})

            #同义词数量的替换生成的句子和原有句子的相似度
            if idx >= half_sim_score_window and len_text - idx - 1 >= half_sim_score_window:
                text_range_min = idx - half_sim_score_window
                text_range_max = idx + half_sim_score_window + 1
            elif idx < half_sim_score_window and len_text - idx - 1 >= half_sim_score_window:
                text_range_min = 0
                text_range_max = sim_score_window
            elif idx >= half_sim_score_window and len_text - idx - 1 < half_sim_score_window:
                text_range_min = len_text - sim_score_window
                text_range_max = len_text
            else:
                text_range_min = 0
                text_range_max = len_text
            old_semantic_sims = semantic_sims
            semantic_sims = \
                sim_predictor.semantic_sim([' '.join(text_cache[text_range_min:text_range_max])] * len(new_texts),
                                           list(map(lambda x: ' '.join(x[text_range_min:text_range_max]), new_texts)))[0]

            num_queries += len(new_texts)

            if len(new_probs.shape) < 2:
                new_probs = new_probs.unsqueeze(0)

            new_probs_mask = (orig_label != torch.argmax(new_probs, dim=-1)).data.cpu().numpy()


            # 过滤单词
            synonyms_pos_ls = [criteria.get_pos(new_text)[idx] for new_text in new_texts]

            pos_mask = np.array(criteria.pos_filter(pos_ls[idx], synonyms_pos_ls))
            new_probs_mask *= pos_mask

            Energy_change = (sum(semantic_sims) / len(semantic_sims)) -  (sum(old_semantic_sims) / len(old_semantic_sims))



            # 替换的条件:如果多个替换攻击成功，则选同义词编号最小的
            if np.sum(new_probs_mask) > 0:
                text_prime[idx] = synonyms[(new_probs_mask).argmax()]
                num_changed += 1
                break
            # 有一定概率接受一个失败的攻击，使用最近的同义词减轻损失
            elif  math.exp( - Energy_change / temperature ) > probability_threshold :
                text_prime[idx] = " "
                num_changed += 1
                if  temperature > 0.01:
                    temperature = temperature-delay_temperature
            # 否则
            else:

                new_label_probs = new_probs[:, orig_label] + torch.from_numpy((1 - pos_mask).astype(float)).float()
                new_label_prob_min, new_label_prob_argmin = torch.min(new_label_probs, dim=-1)
                text_prime[idx] = synonyms[new_label_prob_argmin]
                num_changed += 1
            text_cache = text_prime[:]

            # print("---------------------------premise------------------------------",premise)

        return ' '.join(text_prime), num_changed, orig_label, \
            torch.argmax(predictor({'premises':[premise], 'hypotheses': [text_prime]}), dim=-1).item(), num_queries




def TextFooler(premise, hypothese, true_label, predictor, stop_words_set, word2idx, idx2word, cos_sim, sim_predictor=None,
           import_score_threshold=-1., sim_score_threshold=0.5, sim_score_window=15, synonym_num=50, batch_size=32):
    # first check the prediction of the original text
    orig_probs = predictor({'premises': [premise], 'hypotheses': [hypothese]}).squeeze()
    orig_label = torch.argmax(orig_probs, dim=-1).item()
    orig_prob = orig_probs.max()
    if true_label != orig_label:
        return '', 0, orig_label, orig_label, 0
    else:
        len_text = len(hypothese)
        if len_text < sim_score_window:
            sim_score_threshold = 0.1  # shut down the similarity thresholding function
        half_sim_score_window = (sim_score_window - 1) // 2
        num_queries = 1

        # get the pos and verb tense info
        pos_ls = criteria.get_pos(hypothese)

        # get importance score
        leave_1_texts = [hypothese[:ii]+['<oov>']+hypothese[min(ii+1, len_text):] for ii in range(len_text)]
        leave_1_probs = predictor({'premises':[premise]*len_text, 'hypotheses': leave_1_texts})
        num_queries += len(leave_1_texts)
        leave_1_probs_argmax = torch.argmax(leave_1_probs, dim=-1)
        import_scores = (orig_prob - leave_1_probs[:, orig_label] + (leave_1_probs_argmax != orig_label).float() * (
                leave_1_probs.max(dim=-1)[0] - torch.index_select(orig_probs, 0,
                                                                  leave_1_probs_argmax))).data.cpu().numpy()

        # get words to perturb ranked by importance scorefor word in words_perturb
        words_perturb = []
        for idx, score in sorted(enumerate(import_scores), key=lambda x: x[1], reverse=True):
            if score > import_score_threshold and hypothese[idx] not in stop_words_set:
                words_perturb.append((idx, hypothese[idx]))

        # find synonyms
        words_perturb_idx = [word2idx[word] for idx, word in words_perturb if word in word2idx]
        synonym_words, _ = pick_most_similar_words_batch(words_perturb_idx, cos_sim, idx2word, synonym_num, 0.5)
        synonyms_all = []
        for idx, word in words_perturb:
            if word in word2idx:
                synonyms = synonym_words.pop(0)
                if synonyms:
                    synonyms_all.append((idx, synonyms))

        # start replacing and attacking
        text_prime = hypothese[:]
        text_cache = text_prime[:]
        num_changed = 0
        for idx, synonyms in synonyms_all:
            new_texts = [text_prime[:idx] + [synonym] + text_prime[min(idx + 1, len_text):] for synonym in synonyms]
            new_probs = predictor({'premises': [premise] * len(synonyms), 'hypotheses': new_texts}, datasize = len(synonyms))

            # compute semantic similarity
            if idx >= half_sim_score_window and len_text - idx - 1 >= half_sim_score_window:
                text_range_min = idx - half_sim_score_window
                text_range_max = idx + half_sim_score_window + 1
            elif idx < half_sim_score_window and len_text - idx - 1 >= half_sim_score_window:
                text_range_min = 0
                text_range_max = sim_score_window
            elif idx >= half_sim_score_window and len_text - idx - 1 < half_sim_score_window:
                text_range_min = len_text - sim_score_window
                text_range_max = len_text
            else:
                text_range_min = 0
                text_range_max = len_text
            semantic_sims = \
                sim_predictor.semantic_sim([' '.join(text_cache[text_range_min:text_range_max])] * len(new_texts),
                                           list(map(lambda x: ' '.join(x[text_range_min:text_range_max]), new_texts)))[0]

            num_queries += len(new_texts)
            if len(new_probs.shape) < 2:
                new_probs = new_probs.unsqueeze(0)
            new_probs_mask = (orig_label != torch.argmax(new_probs, dim=-1)).data.cpu().numpy()
            # prevent bad synonyms
            new_probs_mask *= (semantic_sims >= sim_score_threshold)
            # prevent incompatible pos
            synonyms_pos_ls = [criteria.get_pos(new_text[max(idx - 4, 0):idx + 5])[min(4, idx)]
                               if len(new_text) > 10 else criteria.get_pos(new_text)[idx] for new_text in new_texts]
            pos_mask = np.array(criteria.pos_filter(pos_ls[idx], synonyms_pos_ls))
            new_probs_mask *= pos_mask

            if np.sum(new_probs_mask) > 0:
                text_prime[idx] = synonyms[(new_probs_mask * semantic_sims).argmax()]
                num_changed += 1
                break
            else:
                new_label_probs = new_probs[:, orig_label] + torch.from_numpy(
                    (semantic_sims < sim_score_threshold) + (1 - pos_mask).astype(float)).float().cuda()
                new_label_prob_min, new_label_prob_argmin = torch.min(new_label_probs, dim=-1)
                if new_label_prob_min < orig_prob:
                    text_prime[idx] = synonyms[new_label_prob_argmin]
                    num_changed += 1
            text_cache = text_prime[:]

        return ' '.join(text_prime), num_changed, orig_label, \
            torch.argmax(predictor({'premises':[premise], 'hypotheses': [text_prime]}), dim=-1).item(), num_queries









def attack_infer(premise, hypothese, true_label, predictor, stop_words_set, word2idx, idx2word, cos_sim, sim_predictor=None,
           import_score_threshold=-1., sim_score_threshold=0.5, sim_score_window=15, synonym_num=50, batch_size=32):

    orig_probs = predictor({'premises': [premise], 'hypotheses': [hypothese]})
    orig_label = torch.argmax(orig_probs, dim=-1).item()
    orig_prob = orig_probs.max()

    if true_label != orig_label:
        return '', 0, orig_label, orig_label, 0
    else:

        len_text = len(premise)
        if len_text > sim_score_window:
            half_sim_score_window = sim_score_window
        else:
            half_sim_score_window = (sim_score_window - 1) // 2
        num_queries = 1


        print("---------------------------")

        # get the pos and verb tense info
        pos_ls = criteria.get_pos(hypothese)

        # get importance score
        leave_1_texts = [hypothese[:ii]+['<oov>']+hypothese[min(ii+1,len_text):] for ii in range(len_text)]

        leave_1_probs = predictor({'premises':[premise]*len_text, 'hypotheses': leave_1_texts})
        
        num_queries += len(leave_1_texts)


        # Calculate simple importance scores based on change in probability of the original label
        affect_words = (orig_prob - leave_1_probs[:, orig_label]).data.cpu().numpy()


        # Select the top 50 words by simple importance score, excluding stopwords
        words_perturb = []
        score_indices = sorted(enumerate(affect_words), key=lambda x: x[1], reverse=True)
        for idx, score in score_indices:
            # if idx < len(hypothese) and hypothese[idx] not in stop_words_set:  # Assuming we only consider negative impact
            #     words_perturb.append((idx, hypothese[idx]))
            if idx < len(hypothese) and hypothese[idx] not in stop_words_set:  # Assuming we only consider negative impact
                words_perturb.append((idx, hypothese[idx]))

        temperature = 0.1
        delay_temperature = 0.002
        # probability_threshold = 1000
        probability_threshold = 10

        
        # print("----------temperature---------------", temperature)

        # find synonyms   每个句子中的idx后面跟着同义词列表
        words_perturb_idx = [word2idx[word] for idx, word in words_perturb if word in word2idx]
        synonym_words, _ = pick_most_similar_words_batch(words_perturb_idx, cos_sim, idx2word, synonym_num, 0.)
        synonyms_all = []
        for idx, word in words_perturb:
            if word in word2idx:
                synonyms = synonym_words.pop(0)
                if synonyms:
                    synonyms_all.append((idx, synonyms))

        # start replacing and attacking
        text_prime = hypothese[:]
        text_cache = text_prime[:]
        num_changed = 0


        semantic_sims = \
            sim_predictor.semantic_sim([' '.join(text_cache[0:15])] * len(text_prime),
                                       list(map(lambda x: ' '.join(x[0:15]), text_prime)))[0]


        for idx, synonyms in synonyms_all:
            print("------------idx: ---------------",idx)
            new_texts = [text_prime[:idx] + [synonym] + text_prime[min(idx + 1, len_text):] for synonym in synonyms]
            new_probs = predictor({'premises': [premise]*len(synonyms), 'hypotheses': new_texts}, datasize = len(synonyms) )

            #同义词数量的替换生成的句子和原有句子的相似度
            if idx >= half_sim_score_window and len_text - idx - 1 >= half_sim_score_window:
                text_range_min = idx - half_sim_score_window
                text_range_max = idx + half_sim_score_window + 1
            elif idx < half_sim_score_window and len_text - idx - 1 >= half_sim_score_window:
                text_range_min = 0
                text_range_max = sim_score_window
            elif idx >= half_sim_score_window and len_text - idx - 1 < half_sim_score_window:
                text_range_min = len_text - sim_score_window
                text_range_max = len_text
            else:
                text_range_min = 0
                text_range_max = len_text
            old_semantic_sims = semantic_sims
            semantic_sims = \
                sim_predictor.semantic_sim([' '.join(text_cache[text_range_min:text_range_max])] * len(new_texts),
                                           list(map(lambda x: ' '.join(x[text_range_min:text_range_max]), new_texts)))[0]

            num_queries += len(new_texts)

            if len(new_probs.shape) < 2:
                new_probs = new_probs.unsqueeze(0)


            new_probs_mask = (orig_label != torch.argmax(new_probs, dim=-1)).data.cpu().numpy()

            # 过滤单词
            synonyms_pos_ls = [criteria.get_pos(new_text)[idx] for new_text in new_texts]

            pos_mask = np.array(criteria.pos_filter(pos_ls[idx], synonyms_pos_ls))
            new_probs_mask *= pos_mask

            Energy_change = (sum(semantic_sims) / len(semantic_sims)) -  (sum(old_semantic_sims) / len(old_semantic_sims))

            # 替换的条件:如果多个替换攻击成功，则选同义词编号最小的
            if np.sum(new_probs_mask) > 0:
                text_prime[idx] = synonyms[(new_probs_mask).argmax()]
                num_changed += 1
                break
            # 有一定概率接受一个失败的攻击，使用最近的同义词减轻损失
            elif  math.exp( - Energy_change / temperature ) > probability_threshold :
                text_prime[idx] = "<oov>"
                num_changed += 1
                if  temperature > 0.01:
                    temperature = temperature-delay_temperature
            # 否则
            else:
                new_label_probs = new_probs[:, orig_label] + torch.from_numpy((1 - pos_mask).astype(float)).float().cuda()
                new_label_prob_min, new_label_prob_argmin = torch.min(new_label_probs, dim=-1)
                text_prime[idx] = synonyms[new_label_prob_argmin]
                num_changed += 1
            text_cache = text_prime[:]

            # print("---------------------------premise------------------------------",premise)

        return ' '.join(text_prime), num_changed, orig_label, \
               torch.argmax(predictor({'premises':[premise], 'hypotheses': [text_prime]}),dim=-1).item(), num_queries


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--dataset_path",
                        type=str,
                        required=True,
                        help="Which dataset to attack.")
    parser.add_argument("--target_model",
                        type=str,
                        required=True,
                        choices=['infersent', 'esim', 'bert'],
                        help="Target models for text classification: fasttext, charcnn, word level lstm "
                             "For NLI: InferSent, ESIM, bert-base-uncased")
    parser.add_argument("--output_dir",
                        type=str,
                        default='/root/autodl-fs/adv_results',
                        help="The output directory where the attack results will be written.")

    parser.add_argument("--target_model_name",
                        type=str,
                        required=True,
                        help="pre-trained target model name")

    parser.add_argument("--target_model_path",
                        type=str,
                        required=True,
                        help="pre-trained target model path")
    parser.add_argument("--word_embeddings_path",
                        type=str,
                        default='',
                        help="path to the word embeddings for the target model")
    parser.add_argument("--counter_fitting_embeddings_path",
                        type=str,
                        required=True,
                        help="path to the counter-fitting embeddings we used to find synonyms")
    parser.add_argument("--counter_fitting_cos_sim_path",
                        type=str,
                        default='',
                        help="pre-compute the cosine similarity scores based on the counter-fitting embeddings")
    parser.add_argument("--USE_cache_path",
                        type=str,
                        required=True,
                        help="Path to the USE encoder cache.")

    ## Model hyperparameters
    parser.add_argument("--sim_score_window",
                        default=15,
                        type=int,
                        help="Text length or token number to compute the semantic similarity score")
    parser.add_argument("--import_score_threshold",
                        default=-1.,
                        type=float,
                        help="Required mininum importance score.")
    parser.add_argument("--sim_score_threshold",
                        default=0.5,
                        type=float,
                        help="Required minimum semantic similarity score.")
    parser.add_argument("--synonym_num",
                        default=50,
                        type=int,
                        help="Number of synonyms to extract")
    parser.add_argument("--batch_size",
                        default=128,
                        type=int,
                        help="Batch size to get prediction")
    parser.add_argument("--data_size",
                        default=500,
                        type=int,
                        help="Data size to create adversaries")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="max sequence length for BERT target model")
    parser.add_argument("--result_model_path_file",
                        type=str,
                        required=True,
                        help="file name of output")

    # ELESSAttack parameters (ported from attack_classification).
    parser.add_argument("--perturb_budget", default=1.00, type=float,
                        help="Maximum hypothesis-token perturbation ratio rho; 1.0 removes the extra ELESS-only cap for fairer comparison with NLI TextFooler")
    parser.add_argument("--initial_temperature", default=0.10, type=float,
                        help="Initial temperature T0 for ELESSAttack; higher exploration while keeping the same Metropolis rule")
    parser.add_argument("--cooling_factor", default=0.95, type=float,
                        help="Multiplicative cooling factor alpha; slower cooling gives the unchanged ELESS search more opportunity to escape local optima")
    parser.add_argument("--energy_zeta", default=0.0, type=float,
                        help="Optional zeta > 1 for one-time temperature calibration")
    parser.add_argument("--lambda_sem", default=0.01, type=float,
                        help="Weight of semantic loss in the joint ELESS energy; local semantic threshold and POS constraints remain unchanged")
    parser.add_argument("--use_ranking", default=1, type=int, choices=[0, 1],
                        help="1=critical-token ranking; 0=NoRank ablation")
    parser.add_argument("--use_local_semantics", default=1, type=int, choices=[0, 1],
                        help="1=local semantic threshold; 0=NoLocal ablation")
    parser.add_argument("--use_energy_search", default=1, type=int, choices=[0, 1],
                        help="1=Metropolis energy search; 0=greedy NoEnergy ablation")
    parser.add_argument("--max_query_budget", default=0, type=int,
                        help="Maximum target-model queries per example; 0 means unlimited")
    parser.add_argument("--random_seed", default=1234, type=int,
                        help="Random seed for repeated-run stability experiments")
    parser.add_argument("--experiment_tag", default="main", type=str,
                        help="Free-form tag for experiment records")


    args = parser.parse_args()

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.random_seed)

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        print("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    else:
        os.makedirs(args.output_dir, exist_ok=True)

    # get data to attack, fetch first [args.data_size] data samples for adversarial attacking
    data = read_data(args.dataset_path, data_size=args.data_size, target_model=args.target_model)
    print("Data import finished!")

    # construct the model
    print("Building Model...")
    if args.target_model == 'esim':
        model = NLI_infer_ESIM(args.target_model_path,
                                args.word_embeddings_path,
                               batch_size=args.batch_size)
    elif args.target_model == 'infersent':
        model = NLI_infer_InferSent(args.target_model_path,
                                    args.word_embeddings_path,
                                    data=data,
                                    batch_size=args.batch_size)
    else:
        model = NLI_infer_BERT(args.target_model_path, max_seq_length=args.max_seq_length, batch_size=args.batch_size)
    predictor = model.text_pred
    print("Model built!")

    # prepare synonym extractor
    # build dictionary via the embedding file
    print("Building vocab...")
    idx2word = {}
    word2idx = {}

    with open(args.counter_fitting_embeddings_path, 'r') as ifile:
        for line in ifile:
            word = line.split()[0]
            if word not in idx2word:
                idx2word[len(idx2word)] = word
                word2idx[word] = len(idx2word) - 1

    # for cosine similarity matrix
    print("Building cos sim matrix...")
    if args.counter_fitting_cos_sim_path:
        # load pre-computed cosine similarity matrix if provided
        print('Load pre-computed cosine similarity matrix from {}'.format(args.counter_fitting_cos_sim_path))
        cos_sim = np.load(args.counter_fitting_cos_sim_path)
    else:
        # calculate the cosine similarity matrix
        print('Start computing the cosine similarity matrix!')
        embeddings = []
        with open(args.counter_fitting_embeddings_path, 'r') as ifile:
            for line in ifile:
                embedding = [float(num) for num in line.strip().split()[1:]]
                embeddings.append(embedding)
        embeddings = np.array(embeddings)
        product = np.dot(embeddings, embeddings.T)
        norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
        cos_sim = product / np.dot(norm, norm.T)
    print("Cos sim import finished!")

    # build the semantic similarity module
    use = USE(args.USE_cache_path)

    
    
    
    
    # start attacking-----------------------------------------------------------------------------------------------------------------------------------------
    orig_failures = 0.
    adv_failures = 0.
    changed_rates = []
    nums_queries = []
    log_file = open(os.path.join(args.output_dir, 'results_log'), 'a')
    orig_premises = []
    orig_hypotheses = []
    adv_hypotheses = []
    true_labels = []
    new_labels = []
    sample_records = []
    trajectory_records = []

    stop_words_set = criteria.get_stopwords()


    if args.target_model_name == 'Texthoaxer':
        fuzz = 0
        top_k_words = 50
        allowed_qrs = 1000000
        sim_lis = []
        random_sims = []

        if args.counter_fitting_cos_sim_path:
            print('Load pre-computed cosine similarity matrix from {}'.format(args.counter_fitting_cos_sim_path))
            with open(args.counter_fitting_cos_sim_path, "rb") as fp:
                sim_lis = np.load(fp, allow_pickle=True)

    if args.target_model_name == 'pww':
        grad_guide = model.text_pred
        tokenizer = get_tokenizer(data['hypotheses'])
        premises_list = [str(value) for value in data['premises']]


    if args.target_model_name == 'deepWordBug':

        vocab_size = 20000

        # IMPORTANT:
        # Do NOT use str(list), because that introduces
        # brackets, quotes and commas into DeepWordBug tokenization.
        #
        # Keep DeepWordBug input in exactly the same word space
        # as data['hypotheses'], which is also used by Pert.
        texts = [
            ' '.join(str(word) for word in value)
            if isinstance(value, (list, tuple))
            else str(value)
            for value in data['hypotheses']
        ]

        (test, dwbtokenizer, rawtest) = loaddatawithtokenize(
            texts[0:args.data_size],
            nb_words=vocab_size,
            datalen=args.data_size
        )

        test = np.array(test)

        dwbdata = {
            'content': test,
            'output': data['labels'][0:args.data_size]
        }

        word_index = dwbtokenizer.word_index

        testword = Worddata(
            dwbdata,
            getidx=True,
            rawdata=rawtest
        )

        test_loader = DataLoader(
            testword,
            batch_size=1,
            num_workers=4,
            shuffle=False,
            collate_fn=custom_collate_fn
        )

        wordinput = []
        tgt = []
        adv = []
        origsample = []
        origsampleidx = []

        grad_guide = model.dwb_text_pred

        index2word = {}
        index2word[0] = '[PADDING]'
        index2word[1] = '[START]'
        index2word[2] = '[UNKNOWN]'
        index2word[3] = ''

        for i in word_index:
            if word_index[i] + 3 < vocab_size:
                index2word[word_index[i] + 3] = i

        for dataid, datadwb in enumerate(test_loader):

            if dataid % 1 == 0:
                print('{} samples out of {} have been finished!'.format(dataid, args.data_size))

            # Match attack_classification: time every baseline attack and push every
            # example through the unified sample_records pipeline.  Without this,
            # Query/Time fall back to nan and INPUT LENGTH ANALYSIS cannot be built.
            attack_started = time.perf_counter()
            new_text, num_changed, orig_label, \
              new_label, true_label, text, num_queries = deepWordBug_nli(
                                                          premise=data['premises'][dataid],
                                                          hypothese=data['hypotheses'][dataid],
                                                          numclass=3,
                                                          dataid=dataid,
                                                          data=datadwb,
                                                          origsample=origsample,
                                                          origsampleidx=origsampleidx,
                                                          tgt=tgt,
                                                          adv=adv,
                                                          wordinput=wordinput,
                                                          model=grad_guide,
                                                          testword=testword,
                                                          index2word=index2word,
                                                          word_index=word_index,
                                                          dwbtokenizer=dwbtokenizer)
            runtime_sec = time.perf_counter() - attack_started

            # Normalize labels before comparisons/serialization.
            orig_label_int = int(orig_label.item() if torch.is_tensor(orig_label) else orig_label)
            new_label_int = int(new_label.item() if torch.is_tensor(new_label) else new_label)
            true_label_int = int(true_label.item() if torch.is_tensor(true_label) else true_label)
            eligible = (true_label_int == orig_label_int)
            success = eligible and (new_label_int != orig_label_int)

            # NLI paper metrics perturb only the hypothesis, so input length and
            # semantic similarity are both defined on the hypothesis (not premise+hypothesis).
            orig_hyp_tokens = data['hypotheses'][dataid]
            orig_text_str = ' '.join(str(x) for x in orig_hyp_tokens)
            adv_text_str = new_text if isinstance(new_text, str) else ' '.join(str(x) for x in new_text)
            text_length = max(len(orig_hyp_tokens), 1)
            changed_rate = float(num_changed) / float(text_length)
            semantic_similarity = float(use.semantic_cosine([orig_text_str], [adv_text_str])[0])

            if not eligible:
                orig_failures += 1
            else:
                nums_queries.append(int(num_queries))

            if true_label_int != new_label_int:
                adv_failures += 1

            changed_rates.append(changed_rate)
            orig_premises.append(' '.join(data['premises'][dataid]))
            orig_hypotheses.append(orig_text_str)
            adv_hypotheses.append(adv_text_str)
            true_labels.append(true_label_int)
            new_labels.append(new_label_int)

            # Same record schema as classification. Baseline-inapplicable ELESS
            # search fields stay None, but all paper metrics are real values.
            sample_records.append({
                'sample_id': dataid,
                'experiment_tag': args.experiment_tag,
                'dataset_path': args.dataset_path,
                'target_model': args.target_model,
                'attack_method': 'deepWordBug',
                'seed': args.random_seed,
                'text_length': text_length,
                'true_label': true_label_int,
                'orig_label': orig_label_int,
                'new_label': new_label_int,
                'eligible': int(eligible),
                'success': int(success),
                'num_changed': int(num_changed),
                'perturb_ratio': changed_rate,
                'semantic_similarity': semantic_similarity,
                'num_queries': int(num_queries),
                'runtime_sec': float(runtime_sec),
                'final_energy': None,
                'success_iteration': None,
                'accepted_worse_moves': None,
                'accepted_moves': None,
                'proposed_moves': None,
                'use_ranking': None,
                'use_local_semantics': None,
                'use_energy_search': None,
                'synonym_num': None,
                'sim_score_window': None,
                'sim_score_threshold': None,
                'perturb_budget': None,
                'initial_temperature': None,
                'cooling_factor': None,
                'energy_zeta': None,
                'max_query_budget': None,
            })

            print("------------deepWordBug 替换次数----------------", num_changed)
            print("------------deepWordBug 查询次数----------------", num_queries)
            print("------------deepWordBug USE相似度---------------", semantic_similarity)

    if args.target_model_name == 'texthoaxer' or args.target_model_name == 'hardlabel' or args.target_model_name == 'HLBB':
        fuzz = 0
        top_k_words = 1000000
        allowed_qrs = 1000000
        sim_lis = []
        random_sims = []

        if args.counter_fitting_cos_sim_path:
            print('Load pre-computed cosine similarity matrix from {}'.format(args.counter_fitting_cos_sim_path))
            with open(args.counter_fitting_cos_sim_path, "rb") as fp:
                    sim_lis = np.load(fp, allow_pickle=True)
                    
                    
    if args.target_model_name == 'limeattack':
        fail  = {} 
        lime_pred = model.lime_text_pred
        embeddings, word2idx_vocab, idx2word_vocab = generate_embedding_mat(args.counter_fitting_embeddings_path)
        cos_sim, word2idx_rev, idx2word_rev = csim_matrix_nli(data, embeddings, word2idx_vocab)
        pos = 1
        k = 5
        query_budget=2000


            
            


    for idx, premise in enumerate(data['premises'][0:args.data_size]):

        if idx % 1 == 0:
            print('{} samples out of {} have been finished!'.format(idx, args.data_size))

        hypothese, true_label = data['hypotheses'][idx], data['labels'][idx]
        
        # print("----------hypothese, true_label-----------", hypothese, true_label)

        if args.target_model_name == "Textfooler" and args.target_model == "infersent" :
                new_text, num_changed, orig_label, \
                new_label, num_queries = TextFooler(premise, hypothese, true_label, predictor, stop_words_set,
                                                word2idx, idx2word, cos_sim, sim_predictor=use,
                                                sim_score_threshold=args.sim_score_threshold,
                                                import_score_threshold=args.import_score_threshold,
                                                sim_score_window=args.sim_score_window,
                                                synonym_num=args.synonym_num,
                                                batch_size=args.batch_size)

        elif args.target_model_name == "Textfooler" and args.target_model == "bert":
            new_text, num_changed, orig_label, \
                new_label, num_queries = TextFooler_bert(premise, hypothese, true_label, predictor, stop_words_set,
                                                         word2idx, idx2word, cos_sim, sim_predictor=use,
                                                         sim_score_threshold=args.sim_score_threshold,
                                                         import_score_threshold=args.import_score_threshold,
                                                         sim_score_window=args.sim_score_window,
                                                         synonym_num=args.synonym_num,
                                                         batch_size=args.batch_size)


        elif args.target_model_name == "Textfooler" and args.target_model == "esim":
            new_text, num_changed, orig_label, \
                new_label, num_queries = TextFooler_esmi(premise, hypothese, true_label, predictor, stop_words_set,
                                                    word2idx, idx2word, cos_sim, sim_predictor=use,
                                                    sim_score_threshold=args.sim_score_threshold,
                                                    import_score_threshold=args.import_score_threshold,
                                                    sim_score_window=args.sim_score_window,
                                                    synonym_num=args.synonym_num,
                                                    batch_size=args.batch_size)

        elif args.target_model_name == "Texthoaxer":
            new_text, num_changed, random_changed, orig_label, \
                new_label, num_queries, sim, random_sim = texthoaxer_attack(fuzz,top_k_words,allowed_qrs,
                                                                            idx, hypothese, premise, true_label, predictor, stop_words_set,
                                                                            word2idx, idx2word, sim_lis , sim_predictor=use,
                                                                            sim_score_threshold=args.sim_score_threshold,
                                                                            import_score_threshold=args.import_score_threshold,
                                                                            sim_score_window=args.sim_score_window,
                                                                            synonym_num=args.synonym_num,
                                                                            batch_size=args.batch_size,embed_func = args.counter_fitting_embeddings_path,budget=1000)

        elif args.target_model_name == 'pww':
            new_text, num_changed, orig_label, \
                new_label, num_queries, runtime_sec = PWW_nli(
                                             premise,
                                             input_text=hypothese,
                                             true_y=true_label,
                                             grad_guide=grad_guide,
                                             tokenizer=tokenizer,
                                             dataset=premises_list,
                                             dataset_path=args.dataset_path
                                             )

        elif args.target_model_name == 'deepWordBug':
             break



        elif args.target_model_name == "TextRSS" and args.target_model == "infersent":
                new_text, num_changed, orig_label, \
                new_label, num_queries = attack_infer(premise, hypothese, true_label, predictor, stop_words_set,
                                                    word2idx, idx2word, cos_sim, sim_predictor=use,
                                                    sim_score_threshold=args.sim_score_threshold,
                                                    import_score_threshold=args.import_score_threshold,
                                                    sim_score_window=args.sim_score_window,
                                                    synonym_num=args.synonym_num,
                                                    batch_size=args.batch_size)


        elif args.target_model_name == "TextRSS" and args.target_model == "bert":
                new_text, num_changed, orig_label, \
                new_label, num_queries = attack(premise, hypothese, true_label, predictor, stop_words_set,
                                                      word2idx, idx2word, cos_sim, sim_predictor=use,
                                                      sim_score_threshold=args.sim_score_threshold,
                                                      import_score_threshold=args.import_score_threshold,
                                                      sim_score_window=args.sim_score_window,
                                                      synonym_num=args.synonym_num,
                                                      batch_size=args.batch_size)
        elif args.target_model_name == 'hardlabel':
                new_text, num_changed, random_changed, orig_label, \
                    new_label, num_queries, sim, random_sim = hardlabelattack_nli(fuzz, top_k_words, allowed_qrs,
                                                                     idx, hypothese, premise, true_label, predictor, stop_words_set,
                                                                     word2idx, idx2word, sim_lis, sim_predictor=use,
                                                                     sim_score_threshold=args.sim_score_threshold,
                                                                     import_score_threshold=args.import_score_threshold,
                                                                     sim_score_window=args.sim_score_window,
                                                                     synonym_num=args.synonym_num,
                                                                     batch_size=args.batch_size) 
        elif args.target_model_name == 'limeattack':
            res, new_text, num_changed, orig_label, \
            new_label, num_queries, runtime_sec = attacklime_nli(idx,fail,premise, hypothese, true_label, predictor, 
                                                                           lime_pred,stop_words_set, word2idx_rev, 
                                                                           idx2word_vocab, cos_sim,
                                                                           synonyms_num=args.synonym_num,
                                                                           sim_score_threshold=args.sim_score_threshold,
                                                                           batch_size=args.batch_size,
                                                                           import_score_threshold=args.import_score_threshold,
                                                                           pos=pos, k=k,query_budget=query_budget)                
        

        else:
                # ELESSAttack NLI: premise fixed, attack the hypothesis.
                new_text, num_changed, orig_label, \
                new_label, num_queries, attack_details = attack_eless_nli(
                                                    premise, hypothese, true_label, predictor, stop_words_set,
                                                    word2idx, idx2word, cos_sim, sim_predictor=use,
                                                    sim_score_threshold=args.sim_score_threshold,
                                                    import_score_threshold=args.import_score_threshold,
                                                    sim_score_window=args.sim_score_window,
                                                    synonym_num=args.synonym_num,
                                                    batch_size=args.batch_size,
                                                    perturb_budget=args.perturb_budget,
                                                    initial_temperature=args.initial_temperature,
                                                    cooling_factor=args.cooling_factor,
                                                    energy_zeta=args.energy_zeta,
                                                    lambda_sem=args.lambda_sem,
                                                    use_ranking=bool(args.use_ranking),
                                                    use_local_semantics=bool(args.use_local_semantics),
                                                    use_energy_search=bool(args.use_energy_search),
                                                    max_query_budget=args.max_query_budget,
                                                    return_details=True)

                rec = {
                    'sample_id': idx,
                    'experiment_tag': args.experiment_tag,
                    'dataset_path': args.dataset_path,
                    'target_model': args.target_model,
                    'attack_method': args.target_model_name,
                    'seed': args.random_seed,
                    'text_length': len(hypothese),
                    'true_label': int(true_label),
                    'orig_label': int(orig_label.item() if torch.is_tensor(orig_label) else orig_label),
                    'new_label': int(new_label.item() if torch.is_tensor(new_label) else new_label),
                    'eligible': int(attack_details['eligible']),
                    'success': int(attack_details['success']),
                    'num_changed': attack_details['num_changed'],
                    'perturb_ratio': attack_details['perturb_ratio'],
                    'semantic_similarity': attack_details['final_similarity'],
                    'final_energy': attack_details['final_energy'],
                    'num_queries': attack_details['num_queries'],
                    'runtime_sec': attack_details['runtime_sec'],
                    'success_iteration': attack_details['success_iteration'],
                    'accepted_worse_moves': attack_details['accepted_worse_moves'],
                    'accepted_moves': attack_details['accepted_moves'],
                    'proposed_moves': attack_details['proposed_moves'],
                    'use_ranking': args.use_ranking,
                    'use_local_semantics': args.use_local_semantics,
                    'use_energy_search': args.use_energy_search,
                    'synonym_num': args.synonym_num,
                    'sim_score_window': args.sim_score_window,
                    'sim_score_threshold': args.sim_score_threshold,
                    'perturb_budget': args.perturb_budget,
                    'initial_temperature': args.initial_temperature,
                    'cooling_factor': args.cooling_factor,
                    'energy_zeta': args.energy_zeta,
                    'lambda_sem': args.lambda_sem,
                    'max_query_budget': args.max_query_budget,
                }
                sample_records.append(rec)
                for tr in attack_details['trajectory']:
                    row = dict(rec)
                    row.update(tr)
                    trajectory_records.append(row)

        if true_label != orig_label:
            orig_failures += 1
        else:
            nums_queries.append(num_queries)
        if true_label != new_label:
            adv_failures += 1

        changed_rate = 1.0 * num_changed / max(len(hypothese), 1)
        # print('orig sentence ({}):'.format(orig_label), ' '.join(text), '\nto new sentence ({}):'.format(new_label),
        #       new_text, '\n{}/{} changed at {:.2f}%'.format(num_changed, len(text), changed_rate * 100))

        # print("---------------------true_label-------------------------",true_label)
        # print("---------------------new_label-------------------------",new_label)
        # print("---------------------orig_label-------------------------",orig_label)
        # print("----------------------!!!s------------------------")

        if not isinstance(new_label, str):
            if isinstance(new_label, torch.Tensor):
                new_label = new_label.cpu().item()
        if not isinstance(orig_label, str):
            if isinstance(orig_label, torch.Tensor):
                orig_label = orig_label.cpu().item()


        # Record PWWS/PWW metrics in the same unified pipeline used by ELESS/deepWordBug.
        # PWW_nli already returns the real target-model query count and runtime_sec;
        # without this block those values never reach sample_records, so paper metrics
        # such as Time_sec/Query/ASR/Pert/Sim become NaN/None for PWW runs.
        if args.target_model_name == 'pww':
            true_label_int = int(true_label.item() if torch.is_tensor(true_label) else true_label)
            orig_label_int = int(orig_label)
            new_label_int = int(new_label)

            eligible = (true_label_int == orig_label_int)
            success = eligible and (new_label_int != orig_label_int)

            orig_text_str = ' '.join(str(x) for x in hypothese)
            adv_text_str = (
                new_text if isinstance(new_text, str)
                else ' '.join(str(x) for x in new_text)
            )

            semantic_similarity = float(
                use.semantic_cosine([orig_text_str], [adv_text_str])[0]
            )

            sample_records.append({
                'sample_id': idx,
                'experiment_tag': args.experiment_tag,
                'dataset_path': args.dataset_path,
                'target_model': args.target_model,
                'attack_method': 'pww',
                'seed': args.random_seed,
                'text_length': len(hypothese),
                'true_label': true_label_int,
                'orig_label': orig_label_int,
                'new_label': new_label_int,
                'eligible': int(eligible),
                'success': int(success),
                'num_changed': int(num_changed),
                'perturb_ratio': float(num_changed) / max(len(hypothese), 1),
                'semantic_similarity': semantic_similarity,
                'num_queries': int(num_queries),
                'runtime_sec': float(runtime_sec),
                'final_energy': None,
                'success_iteration': None,
                'accepted_worse_moves': None,
                'accepted_moves': None,
                'proposed_moves': None,
                'use_ranking': None,
                'use_local_semantics': None,
                'use_energy_search': None,
                'synonym_num': None,
                'sim_score_window': None,
                'sim_score_threshold': None,
                'perturb_budget': None,
                'initial_temperature': None,
                'cooling_factor': None,
                'energy_zeta': None,
                'lambda_sem': None,
                'max_query_budget': None,
            })
        elif args.target_model_name == 'limeattack':
            true_label_int = int(
                true_label.item() if torch.is_tensor(true_label) else true_label
            )
            orig_label_int = int(orig_label)
            new_label_int = int(new_label)

            eligible = (true_label_int == orig_label_int)
            success = eligible and (new_label_int != orig_label_int)

            orig_text_str = ' '.join(str(x) for x in hypothese)

            adv_text_str = (
                new_text
                if isinstance(new_text, str)
                else ' '.join(str(x) for x in new_text)
            )

            semantic_similarity = float(
                use.semantic_cosine(
                    [orig_text_str],
                    [adv_text_str]
                )[0]
            )

            sample_records.append({
                'sample_id': idx,
                'experiment_tag': args.experiment_tag,
                'dataset_path': args.dataset_path,
                'target_model': args.target_model,
                'attack_method': 'limeattack',
                'seed': args.random_seed,

                'text_length': len(hypothese),

                'true_label': true_label_int,
                'orig_label': orig_label_int,
                'new_label': new_label_int,

                'eligible': int(eligible),
                'success': int(success),

                'num_changed': int(num_changed),
                'perturb_ratio':
                    float(num_changed) / max(len(hypothese), 1),

                'semantic_similarity': semantic_similarity,

                'num_queries': int(num_queries),
                'runtime_sec': float(runtime_sec),

                # ELESS-specific metrics do not apply
                'final_energy': None,
                'success_iteration': None,
                'accepted_worse_moves': None,
                'accepted_moves': None,
                'proposed_moves': None,

                'use_ranking': None,
                'use_local_semantics': None,
                'use_energy_search': None,
                'synonym_num': args.synonym_num,
                'sim_score_window': args.sim_score_window,
                'sim_score_threshold': args.sim_score_threshold,
                'perturb_budget': None,
                'initial_temperature': None,
                'cooling_factor': None,
                'energy_zeta': None,
                'lambda_sem': None,
                'max_query_budget': query_budget,
        })

        changed_rates.append(changed_rate)
        orig_premises.append(' '.join(premise))
        orig_hypotheses.append(' '.join(hypothese))
        adv_hypotheses.append(new_text)
        true_labels.append(true_label)
        new_labels.append(new_label)

    # ------------------------------------------------------------------
    # Final result writing follows the classification script standard.
    # Keep the original NLI adversarial-example text file, compute paper-ready
    # metrics in memory, then print and append one unified complete log.
    # No additional CSV/JSON files are created.
    # ------------------------------------------------------------------
    if args.target_model == 'bert':
        labeldict = {0: "contradiction", 1: "entailment", 2: "neutral"}
    else:
        labeldict = {0: "entailment", 1: "neutral", 2: "contradiction"}

    with open(os.path.join(args.output_dir, args.result_model_path_file), 'w', encoding="utf-8") as ofile:
        for orig_premise, orig_hypothesis, adv_hypothesis, true_label, new_label in zip(
                orig_premises, orig_hypotheses, adv_hypotheses, true_labels, new_labels):
            ofile.write(
                'orig premise:\t{}\norig hypothesis ({}):\t{}\nadv hypothesis ({}):\t{}\n\n'.format(
                    orig_premise,
                    labeldict[int(true_label)],
                    orig_hypothesis,
                    labeldict[int(new_label)],
                    adv_hypothesis,
                )
            )

    # ------------------------------------------------------------------
    # Paper-ready aggregate metrics are computed in memory only, exactly as
    # in attack_classification. NLI text_length means hypothesis token length.
    # ------------------------------------------------------------------
    if sample_records:
        eligible = [r for r in sample_records if r['eligible'] == 1]
        successful = [r for r in eligible if r['success'] == 1]

        def mean_or_nan(rows, key):
            vals = []
            for r in rows:
                v = r.get(key, None)
                if v is None:
                    continue
                try:
                    fv = float(v)
                except (TypeError, ValueError):
                    continue
                if not np.isnan(fv):
                    vals.append(fv)
            return float(np.mean(vals)) if vals else float('nan')

        n_total = len(sample_records)
        n_corr = len(eligible)
        n_succ = len(successful)
        baseline_attacks = ['deepWordBug', 'pww', 'TextFooler', 'Textfooler',
                            'Texthoaxer', 'texthoaxer', 'hardlabel', 'HLBB', 'limeattack']

        if args.target_model_name == 'deepWordBug':
            summary_config = {
                'attack_method': 'deepWordBug',
                'query_definition': 'target-model prediction queries reported by NLI baseline',
                'semantic_metric': 'USE full-sentence cosine on hypothesis',
            }
        elif args.target_model_name == 'pww':
            summary_config = {
                'attack_method': 'PWWS',
                'query_definition': 'target-model prediction queries reported by NLI baseline',
                'semantic_metric': 'USE full-sentence cosine on hypothesis',
            }
        elif args.target_model_name in ['TextFooler', 'Textfooler']:
            summary_config = {
                'attack_method': 'TextFooler',
                'query_definition': 'target-model prediction queries reported by NLI attack',
                'semantic_metric': 'USE full-sentence cosine on hypothesis',
                'num_changed_definition': 'number of changed hypothesis token positions',
                'synonym_num': args.synonym_num,
                'sim_score_window': args.sim_score_window,
                'sim_score_threshold': args.sim_score_threshold,
                'perturb_budget': args.perturb_budget,
                'import_score_threshold': args.import_score_threshold,
            }
        elif args.target_model_name == 'hardlabel':
            summary_config = {
                'attack_method': 'HardLabel',
                'query_definition': 'target-model prediction queries reported by NLI baseline',
                'semantic_metric': 'USE full-sentence cosine on hypothesis',
                'num_changed_definition': 'changed hypothesis token positions',
            }
        elif args.target_model_name == 'limeattack':
            summary_config = {
                'attack_method': 'LimeAttack',
                'query_definition': 'target-model prediction queries reported by NLI baseline',
                'semantic_metric': 'USE full-sentence cosine on hypothesis',
                'num_changed_definition': 'changed hypothesis token positions',
            }
        else:
            summary_config = {
                'use_ranking': args.use_ranking,
                'use_local_semantics': args.use_local_semantics,
                'use_energy_search': args.use_energy_search,
                'synonym_num': args.synonym_num,
                'sim_score_window': args.sim_score_window,
                'sim_score_threshold': args.sim_score_threshold,
                'perturb_budget': args.perturb_budget,
                'initial_temperature': args.initial_temperature,
                'cooling_factor': args.cooling_factor,
                'energy_zeta': args.energy_zeta,
                'lambda_sem': args.lambda_sem,
                'max_query_budget': args.max_query_budget,
            }

        summary = {
            'experiment_tag': args.experiment_tag,
            'dataset_path': args.dataset_path,
            'target_model': args.target_model,
            'attack_method': args.target_model_name,
            'seed': args.random_seed,
            'N_total': n_total,
            'N_corr': n_corr,
            'N_succ': n_succ,
            'Orig_percent': (100.0 * n_corr / n_total) if n_total else float('nan'),
            'ASR_percent': (100.0 * n_succ / n_corr) if n_corr else float('nan'),
            'Pert_percent': 100.0 * mean_or_nan(successful, 'perturb_ratio'),
            'Sim_percent': 100.0 * mean_or_nan(successful, 'semantic_similarity'),
            'Query': mean_or_nan(eligible, 'num_queries'),
            'Time_sec': mean_or_nan(eligible, 'runtime_sec'),
            'Avg_changed_tokens_success': mean_or_nan(successful, 'num_changed'),
            'Accepted_worse_moves': mean_or_nan(eligible, 'accepted_worse_moves')
                if args.target_model_name not in baseline_attacks else None,
            'Success_iteration': mean_or_nan(successful, 'success_iteration')
                if args.target_model_name not in baseline_attacks else None,
            'Avg_input_length': mean_or_nan(sample_records, 'text_length'),
            'config': summary_config,
        }

        lengths = np.asarray([r['text_length'] for r in eligible], dtype=float)
        length_groups = {}
        if len(lengths):
            q1, q2, q3 = np.quantile(lengths, [0.25, 0.50, 0.75])
            groups = {'Short': [], 'Medium': [], 'Long': [], 'Very long': []}
            for r in eligible:
                L = r['text_length']
                g = 'Short' if L <= q1 else ('Medium' if L <= q2 else ('Long' if L <= q3 else 'Very long'))
                groups[g].append(r)
            for g, rows in groups.items():
                succ = [r for r in rows if r['success'] == 1]
                length_groups[g] = {
                    'N': len(rows),
                    'Avg_tokens': mean_or_nan(rows, 'text_length'),
                    'ASR_percent': (100.0 * len(succ) / len(rows)) if rows else float('nan'),
                    'Pert_percent': 100.0 * mean_or_nan(succ, 'perturb_ratio'),
                    'Sim_percent': 100.0 * mean_or_nan(succ, 'semantic_similarity'),
                    'Query': mean_or_nan(rows, 'num_queries'),
                    'Time_sec': mean_or_nan(rows, 'runtime_sec'),
                }
        summary['length_groups'] = length_groups

        print('[PAPER METRICS] Orig={:.3f}% ASR={:.3f}% Pert={:.3f}% Sim={:.3f}% Query={:.2f} Time={:.4f}s'.format(
            summary['Orig_percent'], summary['ASR_percent'], summary['Pert_percent'],
            summary['Sim_percent'], summary['Query'], summary['Time_sec']))

    average_similarity, average_length = simcompute_nli.compute_sim(
        os.path.join(args.output_dir, args.result_model_path_file)
    )

    # ==================================================================
    # Unified console + log output, matching attack_classification.
    # ==================================================================
    def _safe_mean(values):
        values = list(values)
        return float(np.mean(values)) if values else float('nan')

    def _fmt(v, digits=4):
        try:
            if v is None:
                return 'None'
            fv = float(v)
            if np.isnan(fv):
                return 'nan'
            return ('{:.%df}' % digits).format(fv)
        except (TypeError, ValueError):
            return str(v)

    old_orig_accuracy = (1.0 - orig_failures / args.data_size) * 100.0 if args.data_size else float('nan')
    old_adv_accuracy = (adv_failures / args.data_size) * 100.0 if args.data_size else float('nan')
    old_avg_changed_rate = _safe_mean(changed_rates) * 100.0
    old_avg_queries = _safe_mean(nums_queries)

    baseline_attacks = ['deepWordBug', 'pww', 'TextFooler', 'Textfooler',
                        'Texthoaxer', 'texthoaxer', 'hardlabel', 'HLBB', 'limeattack']

    if sample_records:
        eligible_records = [r for r in sample_records if int(r.get('eligible', 0)) == 1]
        successful_records = [r for r in eligible_records if int(r.get('success', 0)) == 1]

        n_total_log = len(sample_records)
        n_corr_log = len(eligible_records)
        n_succ_log = len(successful_records)
        orig_percent_log = 100.0 * n_corr_log / n_total_log if n_total_log else float('nan')
        asr_percent_log = 100.0 * n_succ_log / n_corr_log if n_corr_log else float('nan')
        pert_percent_log = 100.0 * _safe_mean(r['perturb_ratio'] for r in successful_records)
        sim_percent_log = 100.0 * _safe_mean(r['semantic_similarity'] for r in successful_records)
        query_log = _safe_mean(r['num_queries'] for r in eligible_records)
        time_log = _safe_mean(r['runtime_sec'] for r in eligible_records)
        success_time_log = _safe_mean(r['runtime_sec'] for r in successful_records)
        avg_changed_tokens_log = _safe_mean(r['num_changed'] for r in successful_records)
        is_eless_run = args.target_model_name not in baseline_attacks
        if is_eless_run:
            avg_final_energy_log = _safe_mean(r['final_energy'] for r in successful_records)
            avg_success_iteration_log = _safe_mean(
                r['success_iteration'] for r in successful_records
                if r.get('success_iteration') is not None and float(r.get('success_iteration', -1)) >= 0
            )
            avg_accepted_worse_log = _safe_mean(r['accepted_worse_moves'] for r in eligible_records)
            total_accepted_worse_log = int(sum(int(r['accepted_worse_moves']) for r in eligible_records))
            avg_accepted_moves_log = _safe_mean(r['accepted_moves'] for r in eligible_records)
            avg_proposed_moves_log = _safe_mean(r['proposed_moves'] for r in eligible_records)
        else:
            avg_final_energy_log = None
            avg_success_iteration_log = None
            avg_accepted_worse_log = None
            total_accepted_worse_log = None
            avg_accepted_moves_log = None
            avg_proposed_moves_log = None
        avg_input_length_log = _safe_mean(r['text_length'] for r in sample_records)
    else:
        eligible_records = []
        successful_records = []
        n_total_log = args.data_size
        n_corr_log = int(args.data_size - orig_failures)
        n_succ_log = int(adv_failures - orig_failures)
        n_succ_log = max(0, n_succ_log)
        orig_percent_log = old_orig_accuracy
        asr_percent_log = 100.0 * n_succ_log / n_corr_log if n_corr_log else float('nan')
        pert_percent_log = old_avg_changed_rate
        sim_percent_log = float(average_similarity) * 100.0
        query_log = old_avg_queries
        time_log = float('nan')
        success_time_log = float('nan')
        avg_changed_tokens_log = float('nan')
        avg_final_energy_log = float('nan')
        avg_success_iteration_log = float('nan')
        avg_accepted_worse_log = float('nan')
        total_accepted_worse_log = 0
        avg_accepted_moves_log = float('nan')
        avg_proposed_moves_log = float('nan')
        avg_input_length_log = _safe_mean(len(h) for h in data['hypotheses'][:args.data_size])

    log_lines = []
    log_lines.append('')
    log_lines.append('=' * 100)
    log_lines.append('ELESSAttack NLI EXPERIMENT - COMPLETE OUTPUT')
    log_lines.append('=' * 100)
    log_lines.append('[RUN INFORMATION]')
    log_lines.append('experiment_tag              : {}'.format(args.experiment_tag))
    log_lines.append('dataset_path                : {}'.format(args.dataset_path))
    log_lines.append('target_model                : {}'.format(args.target_model))
    log_lines.append('target_model_path           : {}'.format(args.target_model_path))
    log_lines.append('perturb_model               : {}'.format(args.target_model_name))
    log_lines.append('random_seed                 : {}'.format(args.random_seed))
    log_lines.append('data_size                   : {}'.format(args.data_size))
    log_lines.append('result_model_path_file      : {}'.format(args.result_model_path_file))
    log_lines.append('output_dir                  : {}'.format(args.output_dir))

    log_lines.append('')
    log_lines.append('[DATASET / SAMPLE STATISTICS]')
    log_lines.append('N_total                     : {}'.format(n_total_log))
    log_lines.append('N_corr                      : {}'.format(n_corr_log))
    log_lines.append('N_succ                      : {}'.format(n_succ_log))
    log_lines.append('original_failures           : {}'.format(int(orig_failures)))
    log_lines.append('avg_input_length            : {}'.format(_fmt(avg_input_length_log, 3)))

    log_lines.append('')
    log_lines.append('[MAIN PAPER METRICS]')
    log_lines.append('Orig_percent                : {}%'.format(_fmt(orig_percent_log, 3)))
    log_lines.append('ASR_percent                 : {}%'.format(_fmt(asr_percent_log, 3)))
    log_lines.append('Pert_percent                : {}%'.format(_fmt(pert_percent_log, 3)))
    log_lines.append('Sim_percent                 : {}%'.format(_fmt(sim_percent_log, 3)))
    log_lines.append('Query                       : {}'.format(_fmt(query_log, 3)))
    log_lines.append('Time_sec                    : {}'.format(_fmt(time_log, 6)))

    log_lines.append('')
    log_lines.append('[SEARCH BEHAVIOR / LOCAL OPTIMUM DATA]')
    if args.target_model_name in baseline_attacks:
        log_lines.append('ELESS-specific search metrics: N/A for baseline attack {}'.format(args.target_model_name))
        log_lines.append('Avg_changed_tokens_success  : {}'.format(_fmt(avg_changed_tokens_log, 3)))
        log_lines.append('Avg_success_runtime_sec     : {}'.format(_fmt(success_time_log, 6)))
    else:
        log_lines.append('Avg_final_energy            : {}'.format(_fmt(avg_final_energy_log, 6)))
        log_lines.append('Avg_success_iteration       : {}'.format(_fmt(avg_success_iteration_log, 3)))
        log_lines.append('Avg_accepted_moves          : {}'.format(_fmt(avg_accepted_moves_log, 3)))
        log_lines.append('Avg_proposed_moves          : {}'.format(_fmt(avg_proposed_moves_log, 3)))
        log_lines.append('Avg_accepted_worse_moves    : {}'.format(_fmt(avg_accepted_worse_log, 3)))
        log_lines.append('Total_accepted_worse_moves  : {}'.format(total_accepted_worse_log))
        log_lines.append('Avg_changed_tokens_success  : {}'.format(_fmt(avg_changed_tokens_log, 3)))
        log_lines.append('Avg_success_runtime_sec     : {}'.format(_fmt(success_time_log, 6)))

    log_lines.append('')
    log_lines.append('[LEGACY / COMPATIBILITY METRICS]')
    log_lines.append('Old original accuracy       : {}%'.format(_fmt(old_orig_accuracy, 3)))
    log_lines.append('Old adv accuracy            : {}%'.format(_fmt(old_adv_accuracy, 3)))
    log_lines.append('Old avg changed rate        : {}%'.format(_fmt(old_avg_changed_rate, 3)))
    log_lines.append('Old avg num queries         : {}'.format(_fmt(old_avg_queries, 3)))
    log_lines.append('simcompute avg similarity   : {}'.format(_fmt(average_similarity, 6)))
    log_lines.append('simcompute avg text length  : {}'.format(_fmt(average_length, 3)))

    log_lines.append('')
    log_lines.append('[ATTACK CONFIGURATION]')
    if args.target_model_name == 'deepWordBug':
        log_lines.append('attack_method                : deepWordBug')
        log_lines.append('semantic_metric              : USE full-sentence cosine on hypothesis')
        log_lines.append('query_definition             : NLI baseline-reported target-model queries')
    elif args.target_model_name == 'pww':
        log_lines.append('attack_method                : PWWS')
        log_lines.append('semantic_metric              : USE full-sentence cosine on hypothesis')
        log_lines.append('query_definition             : NLI baseline-reported target-model queries')
    elif args.target_model_name in ['TextFooler', 'Textfooler']:
        log_lines.append('attack_method                : TextFooler')
        log_lines.append('semantic_metric              : USE full-sentence cosine on hypothesis')
        log_lines.append('num_changed_definition       : changed hypothesis token positions')
        log_lines.append('import_score_threshold       : {}'.format(args.import_score_threshold))
        log_lines.append('sim_score_threshold          : {}'.format(args.sim_score_threshold))
        log_lines.append('sim_score_window             : {}'.format(args.sim_score_window))
        log_lines.append('synonym_num_K                : {}'.format(args.synonym_num))
        log_lines.append('perturb_budget_rho           : {}'.format(args.perturb_budget))
        log_lines.append('batch_size                   : {}'.format(args.batch_size))
    elif args.target_model_name == 'hardlabel':
        log_lines.append('attack_method                : HardLabel')
        log_lines.append('semantic_metric              : USE full-sentence cosine on hypothesis')
        log_lines.append('num_changed_definition       : changed hypothesis token positions')
    elif args.target_model_name == 'limeattack':
        log_lines.append('attack_method                : LimeAttack')
        log_lines.append('semantic_metric              : USE full-sentence cosine on hypothesis')
        log_lines.append('num_changed_definition       : changed hypothesis token positions')
    else:
        log_lines.append('use_ranking                 : {}'.format(args.use_ranking))
        log_lines.append('use_local_semantics         : {}'.format(args.use_local_semantics))
        log_lines.append('use_energy_search           : {}'.format(args.use_energy_search))
        log_lines.append('import_score_threshold      : {}'.format(args.import_score_threshold))
        log_lines.append('sim_score_threshold         : {}'.format(args.sim_score_threshold))
        log_lines.append('sim_score_window            : {}'.format(args.sim_score_window))
        log_lines.append('synonym_num_K               : {}'.format(args.synonym_num))
        log_lines.append('perturb_budget_rho          : {}'.format(args.perturb_budget))
        log_lines.append('initial_temperature_T0      : {}'.format(args.initial_temperature))
        log_lines.append('cooling_factor_alpha        : {}'.format(args.cooling_factor))
        log_lines.append('energy_zeta                 : {}'.format(args.energy_zeta))
        log_lines.append('lambda_sem                  : {}'.format(args.lambda_sem))
        log_lines.append('max_query_budget            : {}'.format(args.max_query_budget))
        log_lines.append('batch_size                  : {}'.format(args.batch_size))
        log_lines.append('max_seq_length              : {}'.format(args.max_seq_length))

    if sample_records and 'summary' in locals():
        log_lines.append('')
        log_lines.append('[INPUT LENGTH ANALYSIS]')
        if summary.get('length_groups'):
            for group_name in ['Short', 'Medium', 'Long', 'Very long']:
                g = summary['length_groups'].get(group_name, {})
                log_lines.append(
                    '{} | N={} | Avg_tokens={} | ASR={} | Pert={} | Sim={} | Query={} | Time={}'.format(
                        group_name,
                        g.get('N', 0),
                        _fmt(g.get('Avg_tokens'), 3),
                        _fmt(g.get('ASR_percent'), 3),
                        _fmt(g.get('Pert_percent'), 3),
                        _fmt(g.get('Sim_percent'), 3),
                        _fmt(g.get('Query'), 3),
                        _fmt(g.get('Time_sec'), 6),
                    )
                )
        else:
            log_lines.append('(no eligible samples)')

    log_lines.append('')
    log_lines.append('[OUTPUT FILES]')
    log_lines.append('adversarial_examples        : {}'.format(
        os.path.join(args.output_dir, args.result_model_path_file)))

    dataset_name = os.path.basename(args.dataset_path)
    log_path = os.path.join(
        args.output_dir,
        f'log_console_{dataset_name}_{args.target_model}.txt'
    )
    log_lines.append('console_log                 : {}'.format(log_path))
    log_lines.append('=' * 100)

    complete_message = '\n'.join(log_lines) + '\n'
    print(complete_message, end='')

    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(complete_message)
        f.flush()

    print('[LOG SAVED] {}'.format(log_path))

if __name__ == "__main__":
    main()