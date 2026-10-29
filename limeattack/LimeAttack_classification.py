import pickle
import argparse
import os
import numpy as np
import dataloader
import criteria
import torch
import torch.nn as nn
from limeattack.train_classifier import Model,NLI_infer_BERT
from limeattack.generate_embedding_mat import generate_embedding_mat,csim_matrix
import datetime
import time
import tqdm
from limeattack.lime.lime_text import LimeTextExplainer,LimeTextExplainer_nli
import copy

parser = argparse.ArgumentParser()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def split_token(seq):
    return seq.split()


def _bert_visible_word_count(predictor, text):
    """Return how many original whitespace words can affect a truncated BERT input.

    This helper is intentionally internal. Existing public method names and
    signatures are kept unchanged.
    """
    try:
        owner = getattr(predictor, '__self__', None)
        dataset = getattr(owner, 'dataset', None)
        tokenizer = getattr(dataset, 'tokenizer', None)
        max_seq_length = int(getattr(dataset, 'max_seq_length', 0))

        if tokenizer is None or max_seq_length <= 2:
            return len(text)

        wordpiece_budget = max_seq_length - 2  # [CLS] + [SEP]
        used = 0
        visible = 0

        for word in text:
            if used >= wordpiece_budget:
                break
            pieces = tokenizer.tokenize(str(word))
            # Keep behavior conservative for unusual/empty tokens.
            piece_count = max(1, len(pieces))
            visible += 1
            used += piece_count

        return min(visible, len(text))
    except Exception:
        # Non-BERT predictor or a custom wrapper: do not filter.
        return len(text)


#lime word ranking 
def lime_word_ranking(text,predictor,true_label):
    # For BERT with max_seq_length=128, attacking/ranking words after the
    # truncation boundary is wasted work because the classifier never sees them.
    visible_len = _bert_visible_word_count(predictor, text)
    ranking_text = text[:visible_len]

    if len(ranking_text) == 0:
        return [], []

    data = " ".join(ranking_text)
    class_names = ['negative','positive']
    explainer = LimeTextExplainer(class_names=class_names,bow=False,split_expression=split_token)

    # Preserve the original setting num_samples=len(text), but apply it only to
    # the BERT-visible prefix. This is the main long-IMDB LIME speedup.
    exp = explainer.explain_instance(
        data,
        predictor,
        num_features=len(ranking_text),
        labels=[true_label],
        num_samples=max(2, len(ranking_text))
    )

    ant = exp.as_index(true_label)
    import_scores = [0.0] * len(ranking_text)
    word_dict = exp.as_list(true_label)

    # Keep true token positions instead of compacting sparse LIME entries.
    for item in ant:
        idx, score = int(item[0]), float(item[1])
        if 0 <= idx < len(import_scores):
            import_scores[idx] = score

    return import_scores,word_dict


def lime_word_ranking_nli(text,premise,predictor,true_label):
    data = " ".join(text)
    class_names = ['negative','positive']
    explainer = LimeTextExplainer_nli(class_names=class_names,bow=False,split_expression=split_token)
    exp = explainer.explain_instance_nli(data,premise,predictor,num_features=len(text),labels=[true_label],num_samples=len(text))
    ant = exp.as_index(true_label)
    import_scores = []
    word_index_dict = {}
    word_dict= exp.as_list(true_label)
    for i in ant:
        word_index_dict[i[0]]=i[1]
    word_index_dict = sorted(word_index_dict.items(),key=lambda x:x[0])
    for i in word_index_dict:
        import_scores.append(i[-1])
    return import_scores,word_dict



def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

# semantic function
#import tensorflow as tf
#import tensorflow_hub as hub
#from scipy.spatial.distance import pdist
#USE = hub.load('https://tfhub.dev/google/universal-sentence-encoder')
def semantic_sim(text1, text2):
    return 1
    a = USE([text1,text2])
    simi = 1.0 - pdist([a['outputs'][0], a['outputs'][1]], 'cosine') 
    return simi.item()





def pick_most_similar_words_batch(src_words, sim_mat, idx2word, ret_count=10, threshold=0.):
    """
    embeddings is a matrix with (d, vocab_size)

    Optimized implementation:
      * keeps the original method name/signature;
      * uses argpartition instead of sorting the entire vocabulary;
      * still returns candidates sorted by cosine similarity.
    """
    if len(src_words) == 0 or ret_count <= 0:
        return [], []

    src_words = np.asarray(src_words, dtype=np.int64)
    rows = np.asarray(sim_mat[src_words, :])
    vocab_size = rows.shape[1]

    # +1 leaves room for the source word itself, which is removed below.
    take_count = min(vocab_size, int(ret_count) + 1)
    if take_count <= 0:
        return [[] for _ in src_words], [[] for _ in src_words]

    if take_count >= vocab_size:
        candidate_idx = np.argsort(-rows, axis=1)[:, :take_count]
    else:
        candidate_idx = np.argpartition(
            -rows, kth=take_count - 1, axis=1
        )[:, :take_count]
        candidate_val = np.take_along_axis(rows, candidate_idx, axis=1)
        local_order = np.argsort(-candidate_val, axis=1)
        candidate_idx = np.take_along_axis(candidate_idx, local_order, axis=1)

    sim_words, sim_values = [], []

    for row_no, src_word in enumerate(src_words):
        ids = candidate_idx[row_no]
        ids = ids[ids != src_word][:int(ret_count)]

        vals = rows[row_no, ids]
        mask = vals >= threshold
        ids = ids[mask]
        vals = vals[mask]

        sim_words.append([idx2word[int(i)] for i in ids])
        sim_values.append(vals)

    return sim_words, sim_values




def issuccess(predictor,batch_size,all_text_ls,text_ls,true_label,num_queries,k=2,query_budget=100):
    # Keep one query in reserve for attacklimeattack()'s final verification.
    if query_budget is not None and query_budget > 0:
        remaining = int(query_budget) - int(num_queries) - 1
        if remaining <= 0:
            return [0, [], []], num_queries
        eval_texts = all_text_ls[:remaining]
    else:
        eval_texts = all_text_ls

    if len(eval_texts) == 0:
        return [0, [], []], num_queries

    # One batched forward pass is shared by success checking and beam pruning.
    _probs = predictor(eval_texts, batch_size=batch_size)
    if not torch.is_tensor(_probs):
        _probs = torch.tensor(_probs)

    if _probs.dim() == 1:
        _probs = _probs.unsqueeze(0)

    num_queries += len(eval_texts)

    label_int = int(true_label.item()) if torch.is_tensor(true_label) else int(true_label)
    pred_labels = torch.argmax(_probs, dim=-1).detach().cpu().numpy()
    success_idx = np.where(pred_labels != label_int)[0]

    # Successful candidates: keep the original method's semantic preference.
    if len(success_idx) > 0:
        semantic = [
            semantic_sim(" ".join(eval_texts[int(i)]), " ".join(text_ls))
            for i in success_idx
        ]
        best_local = int(np.argmax(semantic))
        best_idx = int(success_idx[best_local])
        best_text = eval_texts[best_idx]
        return [1, text_ls, num_queries, best_text], num_queries

    # No success yet. Strictly keep at most k beams.
    # This fixes k=1 -> idx[-0:] -> "keep everything".
    beam_width = max(1, int(k))
    beam_width = min(beam_width, len(eval_texts))

    # Reuse the same probabilities: lower P(original_label) is a better
    # continuation for the attack. No second BERT forward is needed.
    orig_class_prob = _probs[:, label_int].detach().cpu().numpy()
    selected = np.argsort(orig_class_prob)[:beam_width]

    k_text_ls = [eval_texts[int(i)] for i in selected]
    tmp_k = []
    return [0, k_text_ls, tmp_k], num_queries



def auto_beamsearch(predictor,batch_size,text_ls,true_label,num_queries,k_text_ls,words_perturb_idx,start,synonym_words,k=2,query_budget=100):
    curr = []
    tmp = []
    end = min(start + k, len(synonym_words))

    # Safety cap only; with strict beam retention this normally stays far below it.
    max_candidates_per_step = 1000
    suc = [0, k_text_ls, []]

    for idx in range(start, end):
        if query_budget is not None and query_budget > 0:
            # One final verification query is reserved in issuccess().
            if num_queries >= int(query_budget) - 1:
                return [0, [], []], num_queries

        if idx == start:
            tmp = k_text_ls
        else:
            tmp = curr

        if not tmp:
            return [0, [], []], num_queries

        curr = []
        seen = set()

        # Expand only the already-pruned beam, instead of allowing exponential
        # growth across positions.
        for text in tmp:
            base = list(text)

            # Keep unchanged path as a candidate.
            key = tuple(base)
            if key not in seen:
                seen.add(key)
                curr.append(base)

            for synonym in synonym_words[idx]:
                aa = base.copy()
                aa[words_perturb_idx[idx]] = synonym
                key = tuple(aa)
                if key in seen:
                    continue
                seen.add(key)
                curr.append(aa)

                if len(curr) >= max_candidates_per_step:
                    break

            if len(curr) >= max_candidates_per_step:
                break

        if query_budget is not None and query_budget > 0:
            remaining = int(query_budget) - int(num_queries) - 1
            if remaining <= 0:
                return [0, [], []], num_queries
            curr = curr[:remaining]

        # issuccess() performs ONE predictor call and reuses its probabilities
        # for both success checking and beam pruning.
        suc, num_queries = issuccess(
            predictor, batch_size, curr, text_ls, true_label,
            num_queries, k, query_budget
        )

        if suc[0] == 1:
            return suc, num_queries

        curr = suc[1]
        if not curr:
            return suc, num_queries

    return suc,num_queries




#===================================attack=============================================
def attacklimeattack(text_id,fail,text_ls,true_label,predictor,lime_pred,stop_word_set,word2idx, idx2word, cos_sim,synonyms_num=50,sim_score_threshold=0, batch_size=32,import_score_threshold=0.005,pos=1,k=1,query_budget=1000):
    started = time.perf_counter()
    num_queries = 0
    runtime_sec = 0.0

    # ------------------------------------------------------------------
    # Original prediction
    # ------------------------------------------------------------------
    orig_probs = predictor([text_ls]).squeeze()
    num_queries += 1
    orig_label = torch.argmax(orig_probs)

    if true_label != orig_label:
        fail[text_id] = 0
        print("-------------------1------------------原始文本被模型预测为错误标签了，根本不用攻击。----")
        runtime_sec = float(time.perf_counter() - started)
        return [2, orig_label, 0], " ", 0, orig_label, orig_label, num_queries, runtime_sec

    if query_budget is not None and query_budget > 0 and num_queries >= int(query_budget):
        fail[text_id] = 3
        runtime_sec = float(time.perf_counter() - started)
        return [0], " ", 0, orig_label, orig_label, num_queries, runtime_sec

    # ------------------------------------------------------------------
    # LIME ranking.
    # lime_word_ranking() keeps its original name, but for a bound BERT
    # predictor it now ranks only the WordPiece-visible prefix.
    # ------------------------------------------------------------------
    import_scores, lime_word = lime_word_ranking(
        text_ls, lime_pred, orig_label.item()
    )

    visible_len = min(
        _bert_visible_word_count(predictor, text_ls),
        len(import_scores),
        len(text_ls)
    )

    words_perturb = []
    for idx, score in sorted(
        enumerate(import_scores[:visible_len]),
        key=lambda x: x[1],
        reverse=True
    ):
        if score > import_score_threshold and text_ls[idx] not in stop_word_set:
            words_perturb.append((idx, text_ls[idx]))

    word_perturb_text_idx = [idx for idx, word in words_perturb]
    word_list = [word for idx, word in words_perturb]

    # ------------------------------------------------------------------
    # Synonym lookup.
    # Batch unique source words so repeated IMDB words do not repeatedly
    # sort/search the same cosine-similarity row.
    # ------------------------------------------------------------------
    synonym_cache = {}
    unique_words = []
    unique_ids = []

    for word in word_list:
        if word in word2idx and word not in synonym_cache:
            synonym_cache[word] = []
            unique_words.append(word)
            unique_ids.append(word2idx[word])

    if unique_ids:
        unique_synonyms, _ = pick_most_similar_words_batch(
            unique_ids,
            cos_sim,
            idx2word,
            synonyms_num,
            sim_score_threshold
        )
        for word, syns in zip(unique_words, unique_synonyms):
            synonym_cache[word] = syns

    synonym_words = [
        list(synonym_cache.get(word, []))
        for word in word_list
    ]

    # Filter empty synonym lists while preserving original token positions.
    tmp_syn = []
    new_word_1 = []
    new_word_2 = []

    for i in range(len(word_perturb_text_idx)):
        if synonym_words[i]:
            new_word_2.append(word_perturb_text_idx[i])
            new_word_1.append(word_list[i])
            tmp_syn.append(synonym_words[i])

    word_perturb_text_idx = new_word_2
    word_list = new_word_1
    synonym_words = tmp_syn

    if not word_perturb_text_idx:
        fail[text_id] = 4
        print("-------------------2----------所有高重要性的词都被过滤掉了（停用词/重要性太低），或这些词没有可用的同义词。无法构建任何攻击词表。------------")
        runtime_sec = float(time.perf_counter() - started)
        return [0], " ", 0, orig_label, orig_label, num_queries, runtime_sec

    # ------------------------------------------------------------------
    # Local helper for the unchanged 7-value return interface.
    # issuccess() reserves one target-model query for this verification.
    # ------------------------------------------------------------------
    def _finish_success(suc, message):
        nonlocal num_queries
        print(message)

        if len(suc) > 3:
            candidate = suc[3]
            new_text = candidate.tolist() if hasattr(candidate, 'tolist') else list(candidate)
        else:
            if isinstance(suc[1][0], list):
                new_text = suc[1][0]
            else:
                new_text = suc[1]

        # Because issuccess() reserves this query, query_budget remains a hard cap.
        if query_budget is None or query_budget <= 0 or num_queries < int(query_budget):
            _probs_new = predictor([new_text]).squeeze()
            num_queries += 1
            new_label = torch.argmax(_probs_new)
        else:
            # This branch should normally be unreachable because of the reserve.
            new_label = orig_label

        number_changed = sum(
            w1 != w2 for w1, w2 in zip(new_text, text_ls)
        )
        runtime_sec = float(time.perf_counter() - started)

        return (
            suc,
            ' '.join(new_text),
            number_changed,
            orig_label,
            new_label,
            num_queries,
            runtime_sec
        )

    # If fewer perturbable words than k exist, this is the only possible block.
    if len(word_perturb_text_idx) < k:
        suc, num_queries = auto_beamsearch(
            predictor, batch_size, text_ls, true_label, num_queries,
            [text_ls], word_perturb_text_idx, 0, synonym_words,
            k=len(word_perturb_text_idx), query_budget=query_budget
        )

        if suc[0] == 1:
            return _finish_success(
                suc,
                "-------------------3-------可替换词数不足 k，但整体攻击成功，提前返回。---------------"
            )

        fail[text_id] = 3
        runtime_sec = float(time.perf_counter() - started)
        return [0], " ", 0, orig_label, orig_label, num_queries, runtime_sec

    # First block.
    suc, num_queries = auto_beamsearch(
        predictor, batch_size, text_ls, true_label, num_queries,
        [text_ls], word_perturb_text_idx, 0, synonym_words,
        k=k, query_budget=query_budget
    )

    if suc[0] == 1:
        return _finish_success(
            suc,
            "-------------------4---------首批 k 词攻击成功，提前返回。---------"
        )

    if query_budget is not None and query_budget > 0 and num_queries >= int(query_budget) - 1:
        fail[text_id] = 3
        runtime_sec = float(time.perf_counter() - started)
        return [0], " ", 0, orig_label, orig_label, num_queries, runtime_sec

    k_text_ls = suc[1]
    if not k_text_ls:
        fail[text_id] = 3
        runtime_sec = float(time.perf_counter() - started)
        return [0], " ", 0, orig_label, orig_label, num_queries, runtime_sec

    # Remaining blocks.
    for start in range(k, len(word_perturb_text_idx), k):
        if query_budget is not None and query_budget > 0 and num_queries >= int(query_budget) - 1:
            break

        end = min(start + k, len(word_perturb_text_idx))
        curr_k = end - start

        suc, num_queries = auto_beamsearch(
            predictor, batch_size, text_ls, true_label, num_queries,
            k_text_ls, word_perturb_text_idx, start, synonym_words,
            k=curr_k, query_budget=query_budget
        )

        if suc[0] == 1:
            return _finish_success(
                suc,
                "-------------------5--------------分段攻击成功，提前返回。---------"
            )

        k_text_ls = suc[1]
        if not k_text_ls:
            break

        if end == len(word_perturb_text_idx):
            break

    fail[text_id] = 3
    print("-------------------7-----------多轮 beamsearch 都未能找到有效的对抗样本，攻击失败。-----------")
    runtime_sec = float(time.perf_counter() - started)
    return [0], " ", 0, orig_label, orig_label, num_queries, runtime_sec



def evaluation(res,query_budget):
    sim = [0]
    change_num = 0
    atk=0
    change_all = 0
    query = [0]
    for i in res:
        if i[0] == 2:
            atk+=1
            #count+=1
        if i[0] == 1:
            sim.append(semantic_sim(" ".join(i[-1])," ".join(i[1])))
            change_num += np.sum(np.array(i[-1]) != np.array(i[1]))
            change_all+=len(i[1])
            if len(i[1])+i[2]<=query_budget and (np.sum(np.array(i[-1]) != np.array(i[1])) / len(i[1]))<=0.1: #query budget and perturb limit
                atk+=1
                query.append(i[2])
    print("asr:{},sim:{},pert:{},query:{}".format(atk,np.mean(sim),change_num/change_all,np.mean(query)))


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    #===================================attack============nli=============================================

    
def issuccess_nli(predictor,lime_pred,batch_size,all_text_ls,premise,text_ls,true_label,num_queries,k=2,query_budget=100):
    tmp_prob = []
    success_text_ls = []
    # print("------------premise------------",premise)
    
    _probs = lime_pred(all_text_ls,premise,true_label)
    _probs_argmax = torch.argmax(_probs, dim=-1).cpu().numpy()
    np_true_label = np.array([true_label]*len(all_text_ls))
    success_text_ls = list(np.array(all_text_ls)[np_true_label!=_probs_argmax])
    
    if len(success_text_ls) == 0:
        num_queries += len(all_text_ls)
        semantic_sims = []
        for simi_text in all_text_ls:
            # print("------------text_ls------------",text_ls)
            # print("------------simi_text------------",simi_text)
            semantic_sims.append(semantic_sim(" ".join(text_ls)," ".join(simi_text)))
        semantic_sims = torch.tensor(semantic_sims)
        a,idx = torch.sort(semantic_sims,descending=True)
        k_text_ls = []
        tmp_k = []
        if len(idx)>k:
            for i in idx[:int(k/3)]:
                k_text_ls.append(all_text_ls[i])
            for i in idx[-int(k/3):]:
                k_text_ls.append(all_text_ls[i])
            idx = idx.numpy()[int(k/3):-int(k/3)]
            #b = b[int(k/3):-int(k/3)]
            np.random.shuffle(idx)
            for i in idx[:int(k/3)]:
                k_text_ls.append(all_text_ls[i])
        else:
            for i in idx:
                k_text_ls.append(all_text_ls[i])
        
        return [0,k_text_ls,tmp_k],num_queries

    semantic = []
    bb = success_text_ls[0].tolist()
    num_queries+=all_text_ls.index(bb)
    for i in success_text_ls:
        semantic.append(semantic_sim(" ".join(i)," ".join(text_ls)))
    if num_queries>query_budget:
        return [1,text_ls,num_queries,success_text_ls[semantic.index(max(semantic))]],num_queries
    return [1,text_ls,num_queries,success_text_ls[semantic.index(max(semantic))]],num_queries


def _unwrap_candidates(cands):
    # 目标：返回 List[List[str]]
    # 允许输入：
    #   1) [tokens]                -> [[tokens]]
    #   2) [[tokens1],[tokens2]]   -> 原样
    #   3) [[[tokens1],[tokens2]]] -> 去一层
    if not isinstance(cands, list):
        return [cands]

    # 情况1：单条句子 tokens: ['The','angry',...]
    if len(cands) > 0 and isinstance(cands[0], str):
        return [cands]

    # 情况3：外面多包了一层: [ [ [..],[..] ] ]
    if len(cands) == 1 and isinstance(cands[0], list) and len(cands[0]) > 0 and isinstance(cands[0][0], list):
        return cands[0]

    # 情况2：正常候选集合
    return cands


def auto_beamsearch_nli(predictor,lime_pred,batch_size,premise,text_ls,true_label,num_queries,k_text_ls,words_perturb_idx,start,synonym_words,k=2,query_budget=100):
    

    # --- 统一形状 ---
    k_text_ls = _unwrap_candidates(k_text_ls)

    # text_ls 应该是单句 tokens，如果误传成候选集合，就取第一条当“原句参考”
    if isinstance(text_ls, list) and len(text_ls) > 0 and isinstance(text_ls[0], list):
        text_ls = text_ls[0]

    
    curr = []
    tmp = []
    end = min(start + k, len(synonym_words))   # ★ 最多不超过列表长度
    max_beam = 1000  # 你可以根据自己显存/效果权衡调整
    for idx in range(start, end):
        if idx == start:
            tmp = k_text_ls
        else:
            tmp = curr
            curr = []
        for text in tmp:
            curr.append(text)
            for j in synonym_words[idx]:
                aa = text.copy()
                aa[words_perturb_idx[idx]] = j
                curr.append(aa)
        # --- 剪枝！ ---
        if len(curr) > max_beam:
            # 用概率剪枝，先算一遍分数，取top-N
            # 下面代码只作为范例，你实际可直接用predictor（分批）求softmax最大概率
            scores = []
            for i in range(0, len(curr), batch_size):
                batch = curr[i:i+batch_size]
                
                # print("-------------batch-----------",batch)
                
                probs = predictor({'premises': [premise]*len(batch), 'hypotheses': batch}).squeeze()
                probs = probs if isinstance(probs, torch.Tensor) else torch.tensor(probs)
                max_probs = torch.max(probs, dim=-1)[0]
                scores.extend(max_probs.reshape(-1).tolist())
            top_idx = np.argsort(scores)[-max_beam:]
            curr = [curr[i] for i in top_idx]
        suc, num_queries = issuccess_nli(predictor,lime_pred, batch_size, curr, premise,text_ls, true_label, num_queries, k, query_budget)
        if suc[0] == 1:
            return suc, num_queries
        curr = suc[1]
    return suc,num_queries
    
    
    
    
    
    
    
    
def attacklime_nli(text_id,fail,premise, hypothese,true_label,predictor,lime_pred,stop_word_set,word2idx, idx2word, cos_sim,synonyms_num=50,sim_score_threshold=0, batch_size=32,import_score_threshold=0.005,pos=1,k=1,query_budget=1000):
    """NLI version aligned with attacklimeattack()'s return/statistics interface.

    Always returns 7 values:
        suc, adv_text, number_changed, orig_label, new_label,
        num_queries, runtime_sec

    This prevents downstream statistics from receiving None/missing runtime values.
    """
    started = time.perf_counter()
    num_queries = 0
    runtime_sec = 0.0

    # Keep an immutable reference for changed-token counting.
    original_hypothese = list(hypothese)

    # ------------------------------------------------------------------
    # Original prediction
    # ------------------------------------------------------------------
    orig_probs = predictor({'premises': [premise], 'hypotheses': [original_hypothese]}).squeeze()
    num_queries += 1
    orig_label = torch.argmax(orig_probs)

    if true_label != orig_label:
        fail[text_id] = 0
        print("-------------------1------------------原始文本被模型预测为错误标签了，根本不用攻击。----")
        runtime_sec = float(time.perf_counter() - started)
        return [2, orig_label, 0], " ", 0, orig_label, orig_label, num_queries, runtime_sec

    if query_budget is not None and query_budget > 0 and num_queries >= int(query_budget):
        fail[text_id] = 3
        runtime_sec = float(time.perf_counter() - started)
        return [0], " ", 0, orig_label, orig_label, num_queries, runtime_sec

    # ------------------------------------------------------------------
    # LIME ranking
    # ------------------------------------------------------------------
    import_scores, lime_word = lime_word_ranking_nli(
        original_hypothese, premise, lime_pred, orig_label.item()
    )

    words_perturb = []
    max_rank_len = min(len(import_scores), len(original_hypothese))
    for idx, score in sorted(
        enumerate(import_scores[:max_rank_len]),
        key=lambda x: x[1],
        reverse=True
    ):
        if score > import_score_threshold and original_hypothese[idx] not in stop_word_set:
            words_perturb.append((idx, original_hypothese[idx]))

    word_perturb_text_idx = [idx for idx, word in words_perturb]
    word_list = [word for idx, word in words_perturb]

    # ------------------------------------------------------------------
    # Synonym lookup
    # ------------------------------------------------------------------
    synonym_words = []
    for word in word_list:
        if word in word2idx:
            t, _ = pick_most_similar_words_batch(
                [word2idx[word]], cos_sim, idx2word,
                synonyms_num, sim_score_threshold
            )
            synonym_words.append(t[0])
        else:
            synonym_words.append([])

    # Filter empty synonym lists while preserving original positions.
    tmp_syn = []
    new_word_1 = []
    new_word_2 = []
    for i in range(len(word_perturb_text_idx)):
        if synonym_words[i]:
            new_word_2.append(word_perturb_text_idx[i])
            new_word_1.append(word_list[i])
            tmp_syn.append(synonym_words[i])

    word_perturb_text_idx = new_word_2
    word_list = new_word_1
    synonym_words = tmp_syn

    if not word_perturb_text_idx:
        fail[text_id] = 4
        print("-------------------2----------所有高重要性的词都被过滤掉了（停用词/重要性太低），或这些词没有可用的同义词。无法构建任何攻击词表。------------")
        runtime_sec = float(time.perf_counter() - started)
        return [0], " ", 0, orig_label, orig_label, num_queries, runtime_sec

    # ------------------------------------------------------------------
    # Success finalizer: same 7-value interface as classification.
    # ------------------------------------------------------------------
    def _finish_success_nli(suc, message):
        nonlocal num_queries
        print(message)

        if len(suc) > 3:
            candidate = suc[3]
            new_text = candidate.tolist() if hasattr(candidate, 'tolist') else list(candidate)
        else:
            cand = suc[1]
            cand = _unwrap_candidates(cand)
            new_text = list(cand[0]) if cand else list(original_hypothese)

        # Final verification query, if budget permits.
        if query_budget is None or query_budget <= 0 or num_queries < int(query_budget):
            _probs_new = predictor({
                'premises': [premise],
                'hypotheses': [new_text]
            }).squeeze()
            num_queries += 1
            new_label = torch.argmax(_probs_new)
        else:
            new_label = orig_label

        number_changed = sum(
            w1 != w2 for w1, w2 in zip(new_text, original_hypothese)
        )
        runtime_sec = float(time.perf_counter() - started)

        return (
            suc,
            ' '.join(new_text),
            int(number_changed),
            orig_label,
            new_label,
            int(num_queries),
            float(runtime_sec)
        )

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------
    if len(word_perturb_text_idx) < k:
        suc, num_queries = auto_beamsearch_nli(
            predictor, lime_pred, batch_size, premise, original_hypothese,
            true_label, num_queries, [original_hypothese],
            word_perturb_text_idx, 0, synonym_words,
            k=len(word_perturb_text_idx), query_budget=query_budget
        )
        if suc[0] == 1:
            return _finish_success_nli(
                suc,
                "-------------------3-------可替换词数不足 k，但整体攻击成功，提前返回。---------------"
            )

        fail[text_id] = 3
        runtime_sec = float(time.perf_counter() - started)
        return [0], " ", 0, orig_label, orig_label, int(num_queries), runtime_sec

    # First block.
    suc, num_queries = auto_beamsearch_nli(
        predictor, lime_pred, batch_size, premise, original_hypothese,
        true_label, num_queries, [original_hypothese],
        word_perturb_text_idx, 0, synonym_words,
        k=k, query_budget=query_budget
    )

    if suc[0] == 1:
        return _finish_success_nli(
            suc,
            "-------------------4---------首批 k 词攻击成功，提前返回。---------"
        )

    k_text_ls = _unwrap_candidates(suc[1]) if len(suc) > 1 else []
    if not k_text_ls:
        fail[text_id] = 3
        runtime_sec = float(time.perf_counter() - started)
        return [0], " ", 0, orig_label, orig_label, int(num_queries), runtime_sec

    # Remaining blocks. Keep original hypothesis fixed; only the beam changes.
    for start_idx in range(k, len(word_perturb_text_idx), k):
        if query_budget is not None and query_budget > 0 and num_queries >= int(query_budget):
            break

        end = min(start_idx + k, len(word_perturb_text_idx))
        curr_k = end - start_idx

        suc, num_queries = auto_beamsearch_nli(
            predictor, lime_pred, batch_size, premise, original_hypothese,
            true_label, num_queries, k_text_ls,
            word_perturb_text_idx, start_idx, synonym_words,
            k=curr_k, query_budget=query_budget
        )

        if suc[0] == 1:
            return _finish_success_nli(
                suc,
                "-------------------5--------------分段攻击成功，提前返回。---------"
            )

        k_text_ls = _unwrap_candidates(suc[1]) if len(suc) > 1 else []
        if not k_text_ls:
            break

        if end == len(word_perturb_text_idx):
            break

    fail[text_id] = 3
    print("-------------------7-----------多轮 beamsearch 都未能找到有效的对抗样本，攻击失败。-----------")
    runtime_sec = float(time.perf_counter() - started)
    return [0], " ", 0, orig_label, orig_label, int(num_queries), runtime_sec

