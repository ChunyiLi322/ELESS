import argparse
import os
import time
import numpy as np
from pathlib import Path
from scipy.special import softmax
np.random.seed(1234)
import pickle
import dataloader
from train_classifier import Model
from itertools import zip_longest
import criteria
import random
random.seed(0)
import csv
import math
import sys
csv.field_size_limit(sys.maxsize)

from collections import defaultdict

import tensorflow as tf
import tensorflow_hub as hub

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, SequentialSampler, TensorDataset


from BERT.tokenization import BertTokenizer
from BERT.modeling import BertForSequenceClassification, BertConfig
tf.compat.v1.disable_eager_execution()
class USE(object):
    def __init__(self, cache_path):
        super(USE, self).__init__()
        os.environ['TFHUB_CACHE_DIR'] = cache_path
        module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/3"
        self.embed = hub.Module(module_url)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        # Cache repeated USE sentence-pair evaluations. This is especially useful
        # in crossover(), where best_attack windows are evaluated repeatedly.
        self._sim_cache = {}
        self._sim_cache_max = 50000
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
        # Preserve the original return format: [np.ndarray].
        # Only uncached sentence pairs are sent through USE, and all misses are
        # evaluated together in one session.run().
        n = len(sents1)
        if n != len(sents2):
            raise ValueError("sents1 and sents2 must have the same length")

        result = np.empty(n, dtype=np.float32)
        miss_keys, miss_s1, miss_s2, miss_pos = [], [], [], []

        for i, (a, b) in enumerate(zip(sents1, sents2)):
            key = (a, b)
            cached = self._sim_cache.get(key)
            if cached is None:
                miss_keys.append(key)
                miss_s1.append(a)
                miss_s2.append(b)
                miss_pos.append(i)
            else:
                result[i] = cached

        if miss_keys:
            miss_scores = self.sess.run(
                self.sim_scores,
                feed_dict={self.sts_input1: miss_s1, self.sts_input2: miss_s2}
            )
            for pos, key, score in zip(miss_pos, miss_keys, miss_scores):
                score = float(score)
                result[pos] = score
                self._sim_cache[key] = score

            # Cheap bounded cache; clearing avoids unbounded RAM growth on long runs.
            if len(self._sim_cache) > self._sim_cache_max:
                self._sim_cache.clear()

        return [result]

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


class NLI_infer_BERT(nn.Module):
    def __init__(self,
                 pretrained_dir,
                 nclasses,
                 max_seq_length=128,
                 batch_size=32):
        super(NLI_infer_BERT, self).__init__()
        self.model = BertForSequenceClassification.from_pretrained(pretrained_dir, num_labels=nclasses).cuda()

        # construct dataset loader
        self.dataset = NLIDataset_BERT(pretrained_dir, max_seq_length=max_seq_length, batch_size=batch_size)

    def text_pred(self, text_data, batch_size=128):
        # Switch the model to eval mode.
        self.model.eval()

        # transform text data into indices and create batches
        dataloader = self.dataset.transform_text(text_data, batch_size=batch_size)

        probs_all = []
        #print (text_data)
        #         for input_ids, input_mask, segment_ids in tqdm(dataloader, desc="Evaluating"):
        for input_ids, input_mask, segment_ids in dataloader:
            input_ids = input_ids.cuda(non_blocking=True)
            input_mask = input_mask.cuda(non_blocking=True)
            segment_ids = segment_ids.cuda(non_blocking=True)

            with torch.inference_mode():
                logits = self.model(input_ids, segment_ids, input_mask)
                probs = nn.functional.softmax(logits, dim=-1)
                probs_all.append(probs)

        return torch.cat(probs_all, dim=0)


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

    def convert_examples_to_features(self, examples, max_seq_length, tokenizer):
        """Loads a data file into a list of `InputBatch`s."""

        features = []
        for (ex_index, text_a) in enumerate(examples):
            tokens_a = tokenizer.tokenize(' '.join(text_a))

            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

            tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
            segment_ids = [0] * len(tokens)

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

    def transform_text(self, data, batch_size=32):
        # transform data into seq of embeddings
        eval_features = self.convert_examples_to_features(data,
                                                          self.max_seq_length, self.tokenizer)

        # Building a single contiguous NumPy array first is much faster than
        # torch.tensor(list_of_lists) for large attack batches.
        all_input_ids = torch.from_numpy(np.asarray([f.input_ids for f in eval_features], dtype=np.int64))
        all_input_mask = torch.from_numpy(np.asarray([f.input_mask for f in eval_features], dtype=np.int64))
        all_segment_ids = torch.from_numpy(np.asarray([f.segment_ids for f in eval_features], dtype=np.int64))
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)

        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(
            eval_data, sampler=eval_sampler, batch_size=batch_size,
            pin_memory=True
        )

        return eval_dataloader


# It calculates semantic similarity between two text inputs.
# text_ls (list): First text input either original text input or previous text.
# new_texts (list): Updated text inputs.
# idx (int): Index of the word that has been changed.
# sim_score_window (int): The number of words to consider around idx. If idx = -1 consider the whole text.
def calc_sim(text_ls, new_texts, idx, sim_score_window, sim_predictor):

    len_text = len(text_ls)
    half_sim_score_window = (sim_score_window - 1) // 2

    # Compute the starting and ending indices of the window.
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

    if text_range_min < 0:
        text_range_min = 0
    if text_range_max > len_text:
        text_range_max = len_text

    if idx == -1:
        text_rang_min = 0
        text_range_max = len_text
    # Calculate semantic similarity using USE.
    semantic_sims = \
        sim_predictor.semantic_sim([' '.join(text_ls[text_range_min:text_range_max])],
            list(map(lambda x: ' '.join(x[text_range_min:text_range_max]), new_texts)))[0]

    return semantic_sims

# Returns the hard label prediction of the target model.
# new_text (list): Text to be fed to target model.
# predictor: Target Model.
# orig_label (int): Original label.
# batch_size (int): Batch size.
def get_attack_result(new_text, predictor, orig_label, batch_size):

    new_probs = predictor(new_text, batch_size=batch_size)
    pr=(orig_label!= torch.argmax(new_probs, dim=-1)).data.cpu().numpy()
    return pr

# It changes the inputt text at the specified index.
# rand_idx (int): Index to be mutated.
# text_ls (list): Original text.
# pos_ls (list): POS tage list.
# new_attack (list): The changed text during genetic optimization.
# best_attack (list): The best attack until now.
# remaining_indices (list): The indices in text input different from original input.
# synonyms_dict (dict): Synonym dict for each word.
# orig_label (int): Original prediction of the target model.
# sim_score_window (int): The number of words to consider around idx.
# predictor: Target model.
# sim_predictor: USE to compute semantic similarity.
# batch_size (int): batch size.
def mutate(rand_idx, text_ls, pos_ls, new_attack, best_attack, remaining_indices,
           synonyms_dict, old_syns, orig_label, sim_score_window,
           predictor, sim_predictor, batch_size):

    # Calculates the semantic similarity before mutation.
    random_text = new_attack[:]
    syns = synonyms_dict[text_ls[rand_idx]]
    prev_semantic_sims = calc_sim(text_ls, [best_attack], rand_idx, sim_score_window, sim_predictor)
    # Gives Priority to Original Word
    orig_word = 0
    if random_text[rand_idx] != text_ls[rand_idx]:

        temp_text = random_text[:]
        temp_text[rand_idx] = text_ls[rand_idx]
        pr = get_attack_result([temp_text], predictor, orig_label, batch_size)
        semantic_sims = calc_sim(text_ls, [temp_text], rand_idx, sim_score_window, sim_predictor)
        if np.sum(pr) > 0:
            orig_word = 1
            return temp_text, 1  #(updated_text, queries_taken)

    # If replacing with original word does not yield adversarial text, then try to replace with other synonyms.
    if orig_word == 0:
        final_mask = []
        new_texts = []
        final_texts = []

        # Replace with synonyms.
        for syn in syns:

            # Ignore the synonym already present at position rand_idx.
            if syn == best_attack[rand_idx]:
                final_mask.append(0)
            else:
                final_mask.append(1)
            temp_text = random_text[:]
            temp_text[rand_idx] = syn
            new_texts.append(temp_text[:])

        # Filter out mutated texts that: (1) are not having same POS tag of the synonym, (2) lowers Semantic Similarity and (3) Do not satisfy adversarial criteria.
        synonyms_pos_ls = [criteria.get_pos(new_text[max(rand_idx - 4, 0):rand_idx + 5])[min(4, rand_idx)]
                               if len(new_text) > 10 else criteria.get_pos(new_text)[rand_idx] for new_text in new_texts]
        pos_mask = np.array(criteria.pos_filter(pos_ls[rand_idx], synonyms_pos_ls))
        semantic_sims = calc_sim(text_ls, new_texts, rand_idx, sim_score_window, sim_predictor)
        pr = get_attack_result(new_texts, predictor, orig_label, batch_size)
        final_mask = np.asarray(final_mask)
        sem_filter = semantic_sims >= prev_semantic_sims[0]
        prediction_filter = pr > 0
        final_mask = final_mask*sem_filter
        final_mask = final_mask*prediction_filter
        final_mask = final_mask*pos_mask
        sem_vals = final_mask*semantic_sims

        for i in range(len(sem_vals)):
            if sem_vals[i] > 0:
                final_texts.append((new_texts[i], sem_vals[i]))

        # Return mutated text with best semantic similarity.
        final_texts.sort(key =  lambda x : x[1])
        final_texts.reverse()

        if len(final_texts) > 0:
            #old_syns[rand_idx].append(final_texts[0][0][rand_idx])
            return final_texts[0][0], len(new_texts)
        else:
            return [], len(new_texts)

# It generates children texts from the parent texts using crossover.
# population_size (int): Size of population used.
# population (list): The population currently in the optimization process.
# parent1_idx (int): The index of parent text input 1.
# parent2_idx (int): The index of parent text input 2.
# text_ls (list): Original text.
# best_attack (list): The best attack until now in the optimization.
# max_changes (int): The number of words substituted in the best_attack.
# changed_indices (list): The indices in text input different from original input.
# sim_score_window (int): The number of words to consider around idx.
# predictor: Target model.
# sim_predictor: USE to compute semantic similarity.
# orig_label (int): Original prediction of the target model.
# batch_size (int): batch size.
def crossover(population_size, population, parent1_idx, parent2_idx,
              text_ls, best_attack, max_changes, changed_indices,
              sim_score_window, sim_predictor,
              predictor, orig_label, batch_size):

    childs = []
    changes = []

    # Do crossover till population_size-1.
    for i in range(population_size-1):

        # Generates new child.
        p1 = population[parent1_idx[i]]
        p2 = population[parent2_idx[i]]
        assert len(p1) == len(p2)
        new_child = []
        for j in range(len(p1)):
            if np.random.uniform() < 0.5:
                new_child.append(p1[j])
            else:
                new_child.append(p2[j])
        change = 0
        cnt = 0
        mismatches = 0
        # Filter out crossover child which (1) Do not improve semantic similarity, (2) Have number of words substituted
        # more than the current best_attack.
        for k in range(len(changed_indices)):
            j = changed_indices[k]
            if new_child[j] == text_ls[j]:
                change+=1
                cnt+=1
            elif new_child[j] == best_attack[j]:
                change+=1
                cnt+=1
            elif new_child[j] != best_attack[j]:
                change+=1
                prev_semantic_sims = calc_sim(text_ls, [best_attack], j, sim_score_window, sim_predictor)
                semantic_sims = calc_sim(text_ls, [new_child], j, sim_score_window, sim_predictor)
                if semantic_sims[0] >= prev_semantic_sims[0]:
                    mismatches+=1
                    cnt+=1
        if cnt==change and mismatches<=max_changes:
            childs.append(new_child)
        changes.append(change)

    if len(childs) == 0:
        return [], 0

    # Filter out childs whoch do not satisfy the adversarial criteria.
    pr = get_attack_result(childs, predictor, orig_label, batch_size)
    final_childs = [childs[i] for i in range(len(pr)) if pr[i] > 0]
    return final_childs, len(final_childs)

        
def hardlabelattack(fuzz_val, top_k_words, qrs, sample_index, text_ls, true_label,
           predictor, stop_words_set, word2idx, idx2word, cos_sim, sim_predictor=None,
           import_score_threshold=-1., sim_score_threshold=0.5, sim_score_window=15, synonym_num=50,
           batch_size=128):

    started = time.perf_counter()

    # Step 0: original prediction
    orig_probs = predictor([text_ls]).squeeze()
    orig_label = torch.argmax(orig_probs)
    orig_prob = orig_probs.max()

    # One prediction for one example = one target-model query.
    num_queries = 1

    if true_label != orig_label:
        runtime_sec = float(time.perf_counter() - started)
        return '', 0, 0, orig_label, orig_label, num_queries, 0.0, 0.0, runtime_sec

    pos_ls = criteria.get_pos(text_ls)
    len_text = len(text_ls)

    if len_text < sim_score_window:
        sim_score_threshold = 0.1

    # Step 1: prepare perturbable words and synonym sets
    words_perturb = []
    pos_pref = ['NOUN', 'VERB', 'ADJ', 'ADV', 'PRON', 'DET', 'ADP', 'NUM', 'CONJ', 'PRT']

    for pos in pos_pref:
        for i in range(len(pos_ls)):
            if (
                pos_ls[i] == pos
                and len(text_ls[i]) > 2
                and text_ls[i] not in stop_words_set
            ):
                words_perturb.append((i, text_ls[i]))

    random.shuffle(words_perturb)
    words_perturb = words_perturb[:top_k_words]

    words_perturb_idx = [
        word2idx[word]
        for idx, word in words_perturb
        if word in word2idx
    ]

    synonym_words = []
    synonym_values = []

    # BERT speed optimization: synonym_num controls candidate synonyms.
    # Do not use top_k_words here; it may be extremely large and causes a
    # full-vocabulary sort for every perturbable token.
    synonym_k = max(1, int(synonym_num))
    for idx in words_perturb_idx:
        sim_vector = cos_sim[idx]
        vocab_size = int(sim_vector.shape[0])
        k = min(synonym_k + 1, vocab_size)

        # O(V) partial top-k selection instead of a complete O(V log V) sort.
        if k < vocab_size:
            top_k_indices = np.argpartition(-sim_vector, k - 1)[:k]
            top_k_indices = top_k_indices[np.argsort(-sim_vector[top_k_indices])]
        else:
            top_k_indices = np.argsort(-sim_vector)

        top_k_indices = [i for i in top_k_indices if int(i) != int(idx)][:synonym_k]
        top_k_scores = sim_vector[top_k_indices]

        temp_words = [idx2word[int(i)] for i in top_k_indices]
        temp_values = list(top_k_scores)

        synonym_words.append(temp_words)
        synonym_values.append(temp_values)

    synonyms_all = []
    synonyms_dict = defaultdict(list)

    for idx, word in words_perturb:
        if word in word2idx:
            synonyms = synonym_words.pop(0)
            if synonyms:
                synonyms_all.append((idx, synonyms))
                synonyms_dict[word] = synonyms

    print("----------------------STEP 1: Random initialization-----------------------")

    # qrs controls the algorithm loop, num_queries is the paper metric.
    qrs = 0
    flag = 0
    th = 0
    random_text = text_ls[:]

    # Build the same progressive random-initialization candidates first, then
    # evaluate them in BERT batches instead of launching one GPU inference per
    # candidate. No synonym count or function interface is changed.
    init_candidates = []
    init_limit = min(len_text, len(synonyms_all))
    for cur_th in range(init_limit):
        candidate_text = text_ls[:]
        for i in range(cur_th + 1):
            idx, syn = synonyms_all[i]
            candidate_text[idx] = random.choice(syn)
        init_candidates.append(candidate_text)

    if init_candidates:
        pr = get_attack_result(init_candidates, predictor, orig_label, batch_size)
        pr = np.asarray(pr).reshape(-1)
        success_idx = np.where(pr > 0)[0]

        if len(success_idx) > 0:
            first = int(success_idx[0])
            random_text = init_candidates[first]
            # Logical query count: candidates up to and including first success.
            qrs = first + 1
            num_queries += first + 1
            flag = 1
        else:
            qrs = len(init_candidates)
            num_queries += len(init_candidates)

    th = qrs

    old_qrs = qrs

    # BERT speed optimization: batch the random restarts.  Each candidate is
    # still counted as one target-model query, preserving the paper metric.
    max_random_restarts = 500
    restart_batch_size = max(1, int(batch_size))
    restart_done = 0

    while restart_done < max_random_restarts and flag == 0:
        cur_bs = min(restart_batch_size, max_random_restarts - restart_done)
        random_batch = []

        for _ in range(cur_bs):
            candidate_text = text_ls[:]
            for j in range(len(synonyms_all)):
                idx = synonyms_all[j][0]
                syn = synonyms_all[j][1]
                if syn:
                    candidate_text[idx] = random.choice(syn)
                if j >= len_text:
                    break
            random_batch.append(candidate_text)

        pr = get_attack_result(random_batch, predictor, orig_label, batch_size)
        pr = np.asarray(pr).reshape(-1)

        qrs += len(random_batch)
        num_queries += len(random_batch)
        restart_done += len(random_batch)

        success_idx = np.where(pr > 0)[0]
        if len(success_idx) > 0:
            first = int(success_idx[0])
            random_text = random_batch[first]
            flag = 1
            break

    if flag == 1:
        changed_indices = [
            i for i in range(len(text_ls))
            if text_ls[i] != random_text[i]
        ]
        num_changed = len(changed_indices)

        if sim_predictor:
            sim = calc_sim(
                text_ls,
                [random_text],
                -1,
                sim_score_window,
                sim_predictor
            )[0]
        else:
            sim = 0.0

        final_probs = predictor([random_text]).squeeze()
        num_queries += 1
        new_label = torch.argmax(final_probs)

        runtime_sec = float(time.perf_counter() - started)


        return (
            ' '.join(random_text),
            num_changed,
            num_changed,
            orig_label,
            new_label,
            num_queries,
            sim,
            sim,
            runtime_sec
        )

    print("Not Found")
    runtime_sec = float(time.perf_counter() - started)

    return (
        '',
        0,
        0,
        orig_label,
        orig_label,
        num_queries,
        0.0,
        0.0,
        runtime_sec
    )

