# coding: utf-8
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import sys
import attr
import nltk
import spacy
from functools import partial
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
# nltk.download('wordnet')
from nltk.corpus import wordnet as wn
from pww.get_NE_list import NE_list
import numpy as np
from pww.unbuffered import Unbuffered
import torch

sys.stdout = Unbuffered(sys.stdout)
# from pywsd.lesk import simple_lesk as disambiguate

nlp = spacy.load('en_core_web_sm', disable=['parser'])
# nlp = spacy.load("en_core_web_sm", disable=["tagger","parser","ner","lemmatizer"])
# Penn TreeBank POS tags:
# http://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html

supported_pos_tags = [
    'CC',  # coordinating conjunction, like "and but neither versus whether yet so"
    'CD',   # Cardinal number, like "mid-1890 34 forty-two million dozen"
    'DT',   # Determiner, like all "an both those"
    'EX',   # Existential there, like "there"
    'FW',   # Foreign word
    'IN',   # Preposition or subordinating conjunction, like "among below into"
    'JJ',  # Adjective, like "second ill-mannered"
    'JJR',  # Adjective, comparative, like "colder"
    'JJS',  # Adjective, superlative, like "cheapest"
    'LS',   # List item marker, like "A B C D"
    'MD',   # Modal, like "can must shouldn't"
    'NN',  # Noun, singular or mass
    'NNS',  # Noun, plural
    'NNP',  # Proper noun, singular
    'NNPS',  # Proper noun, plural
    'PDT',  # Predeterminer, like "all both many"
    'POS',  # Possessive ending, like "'s"
    'PRP',  # Personal pronoun, like "hers herself ours they theirs"
    'PRP$',  # Possessive pronoun, like "hers his mine ours"
    'RB',  # Adverb
    'RBR',  # Adverb, comparative, like "lower heavier"
    'RBS',  # Adverb, superlative, like "best biggest"
    'RP',   # Particle, like "board about across around"
    'SYM',  # Symbol
    'TO',   # to
    'UH',   # Interjection, like "wow goody"
    'VB',  # Verb, base form
    'VBD',  # Verb, past tense
    'VBG',  # Verb, gerund or present participle
    'VBN',  # Verb, past participle
    'VBP',  # Verb, non-3rd person singular present
    'VBZ',  # Verb, 3rd person singular present
    'WDT',  # Wh-determiner, like "that what whatever which whichever"
    'WP',   # Wh-pronoun, like "that who"
    'WP$',  # Possessive wh-pronoun, like "whose"
    'WRB',  # Wh-adverb, like "however wherever whenever"
]


# Cache WordNet candidate strings across samples. Token position is attached later.
_SYNONYM_CACHE = {}


@attr.s
class SubstitutionCandidate:
    token_position = attr.ib()
    similarity_rank = attr.ib()
    original_token = attr.ib()
    candidate_word = attr.ib()


def vsm_similarity(doc, original, synonym):
    window_size = 3
    start = max(0, original.i - window_size)
    return doc[start: original.i + window_size].similarity(synonym)


def _get_wordnet_pos(spacy_token):
    '''Wordnet POS tag'''
    pos = spacy_token.tag_[0].lower()
    if pos in ['r', 'n', 'v']:  # adv, noun, verb
        return pos
    elif pos == 'j':
        return 'a'  # adj


# def _synonym_prefilter_fn(token, synonym):
#     '''
#     Similarity heuristics go here
#     '''
#     if (len(synonym.text.split()) > 2 or (  # the synonym produced is a phrase
#             synonym.lemma == token.lemma) or (  # token and synonym are the same
#             synonym.tag != token.tag) or (  # the pos of the token synonyms are different
#             token.text.lower() == 'be')):  # token is be
#         return False
#     else:
#         return True

def _synonym_prefilter_fn(token, synonym):
    '''
    Similarity heuristics go here
    '''
    # 放宽对短语的限制，允许较长的同义词
    # if len(synonym.text.split()) > 2:
    #     return False

    # 允许相同词干的词（如变形）
    if synonym.lemma == token.lemma:
        return False

    # 放宽词性标签的要求，允许不同词性的同义词
    # if synonym.tag != token.tag:
    #     return False

    # 仍然排除“be”这个特殊情况
    if token.text.lower() == 'be':
        return False

    return True



def _generate_synonym_candidates(token, token_position, rank_fn=None):
    """Generate WordNet candidates efficiently.

    The original code called ``nlp()`` once for every WordNet lemma. Here the
    lemma strings are de-duplicated and parsed with ``nlp.pipe`` in one batch,
    then cached across examples. Candidate semantics are unchanged.
    """
    if rank_fn is None:
        rank_fn = vsm_similarity

    if token.tag_ not in supported_pos_tags:
        return []

    cache_key = (token.text.lower(), token.tag_)
    cached_words = _SYNONYM_CACHE.get(cache_key)

    if cached_words is None:
        wordnet_pos = _get_wordnet_pos(token)
        lemma_names = []
        seen_names = set()
        for synset in wn.synsets(token.text, pos=wordnet_pos):
            for lemma in synset.lemmas():
                name = lemma.name().replace('_', ' ')
                if name not in seen_names:
                    seen_names.add(name)
                    lemma_names.append(name)

        parsed_synonyms = []
        if lemma_names:
            for synonym_doc in nlp.pipe(lemma_names, batch_size=128):
                if len(synonym_doc):
                    parsed_synonyms.append(synonym_doc[0])

        candidate_words = []
        candidate_set = set()
        for synonym in parsed_synonyms:
            if not _synonym_prefilter_fn(token, synonym):
                continue
            candidate_word = synonym.text
            if candidate_word in candidate_set:
                continue
            candidate_set.add(candidate_word)
            candidate_words.append(candidate_word)

        _SYNONYM_CACHE[cache_key] = candidate_words
        cached_words = candidate_words

    return [
        SubstitutionCandidate(
            token_position=token_position,
            similarity_rank=None,
            original_token=token,
            candidate_word=candidate_word,
        )
        for candidate_word in cached_words
    ]


def _compile_perturbed_tokens(doc, accepted_candidates):
    '''
    Traverse the list of accepted candidates and do the token substitutions.
    '''
    candidate_by_position = {}
    for candidate in accepted_candidates:
        candidate_by_position[candidate.token_position] = candidate

    final_tokens = []
    for position, token in enumerate(doc):
        word = token.text
        if position in candidate_by_position:
            candidate = candidate_by_position[position]
            word = candidate.candidate_word.replace('_', ' ')
        final_tokens.append(word)

    return final_tokens


def PWWS_nli(
        doc,
        premise,
        true_y,
        dataset,
        dataset_path,
        word_saliency_list=None,
        rank_fn=None,
        heuristic_fn=None,  # Defined in adversarial_tools.py
        halt_condition_fn=None,  # Defined in adversarial_tools.py
        verbose=True):

    print("-------------pww-------------------1.5-----") 
    # defined in Eq.(8)
    def softmax(x):
        x = x.astype(np.float32)
        exp_x = np.exp(x)
        softmax_x = exp_x / np.sum(exp_x)
        return softmax_x

    print("-------------pww-------------------2-----") 
    heuristic_fn = heuristic_fn or (lambda _, candidate: candidate.similarity_rank)
    halt_condition_fn = halt_condition_fn or (lambda perturbed_text: False)

    perturbed_doc = doc
    perturbed_text = perturbed_doc.text
    original_length = len(doc)   # ✅ 固定原始 token 数，用于计算 sub_rate
    print("-------------pww-------------------3-----")

    substitute_count = 0  # calculate how many substitutions used in a doc
    substitute_tuple_list = []  # save the information of substitute word

    word_saliency_array = np.array([word_tuple[2] for word_tuple in word_saliency_list])
    # print("-------------word_saliency_array-----------------",type(word_saliency_array))
    word_saliency_array = softmax(word_saliency_array)

    NE_candidates = {}

    if dataset_path == "data/imdb":
        NE_candidates = NE_list.L['imdb'][true_y]
    elif dataset_path == "data/ag":
        NE_candidates = NE_list.L['agnews'][true_y]
    else:
        NE_candidates == NE_list.L['agnews'][true_y]

    NE_tags = list(NE_candidates.keys())
    use_NE = True  # whether use NE as a substitute


    # for each word w_i in x, use WordNet to build a synonym set L_i
    for (position, token, word_saliency, tag) in word_saliency_list:

        candidates = []
        candidates = _generate_synonym_candidates(token=token, token_position=position, rank_fn=rank_fn)

        if len(candidates) == 0:
            continue
        perturbed_text = perturbed_doc.text

        # The substitute word selection method R(w_i;L_i) defined in Eq.(4)
        sorted_candidates = zip(map(partial(heuristic_fn, doc.text), candidates), candidates)
        # Sorted according to the return value of heuristic_fn function, that is, \Delta P defined in Eq.(4)
        sorted_candidates = list(sorted(sorted_candidates, key=lambda t: t[0]))

        # delta_p_star is defined in Eq.(5); substitute is w_i^*premise
        delta_p_star, substitute = sorted_candidates.pop()

        # delta_p_star * word_saliency_array[position] equals H(x, x_i^*, w_i) defined in Eq.(7)
        substitute_tuple_list.append(
            (position, token.text, substitute, delta_p_star * word_saliency_array[position], token.tag_))

    # sort all the words w_i in x in descending order based on H(x, x_i^*, w_i)
    sorted_substitute_tuple_list = sorted(substitute_tuple_list, key=lambda t: t[3], reverse=True)

    print("-------------pww-------------------5-----")
    # replace w_i in x^(i-1) with w_i^* to craft x^(i)
    NE_count = 0  # calculate how many NE used in a doc
    change_tuple_list = []
    # 依次替换生成对抗文本
    for (position, token, substitute, score, tag) in sorted_substitute_tuple_list:
        if doc[position].ent_type_ in NE_tags:
            NE_count += 1
        # print("-------------pww-------------------5.1-----")

        change_tuple_list.append((position, token, substitute, score, tag))
        # print("-------------pww-------------------5.2-----")
        perturbed_text = ' '.join(_compile_perturbed_tokens(perturbed_doc, [substitute]))
        # print("-------------pww-------------------5.3-----")
        perturbed_doc = nlp(perturbed_text)
        # print("-------------pww-------------------5.4-----")
        substitute_count += 1

        # 若满足终止条件
        if halt_condition_fn(premise, perturbed_text):
            if verbose:
                print("use", substitute_count, "substitution; use", NE_count, 'NE')
            sub_rate = substitute_count / original_length   # ✅ 改为用原始长度
            NE_rate = 0.0 if substitute_count == 0 else NE_count / substitute_count
            return perturbed_text, substitute_count, NE_rate, change_tuple_list

    if verbose:
        print("use", substitute_count, "substitution; use", NE_count, 'NE')
    sub_rate = substitute_count / original_length   # ✅ 始终保持一致
    NE_rate = 0.0 if substitute_count == 0 else NE_count / substitute_count
       
    return perturbed_text, substitute_count, NE_rate, change_tuple_list


def PWWS(
        doc,
        true_y,
        dataset,
        dataset_path,
        word_saliency_list=None,
        rank_fn=None,
        heuristic_fn=None,  # scalar fallback
        heuristic_batch_fn=None,  # batched candidate scorer for BERT
        halt_condition_fn=None,
        verbose=True):
    print("-------------pww-------------------1.5-----") 
    # defined in Eq.(8)
    def softmax(x):
        x = x.astype(np.float32)
        exp_x = np.exp(x)
        softmax_x = exp_x / np.sum(exp_x)
        return softmax_x

    print("-------------pww-------------------2-----") 
    heuristic_fn = heuristic_fn or (lambda _, candidate: candidate.similarity_rank)
    halt_condition_fn = halt_condition_fn or (lambda perturbed_text: False)

    perturbed_doc = doc
    perturbed_text = perturbed_doc.text
    original_length = len(doc)   # ✅ 固定原始 token 数，用于计算 sub_rate
    print("-------------pww-------------------3-----")

    substitute_count = 0  # 替换的单词数
    substitute_tuple_list = []  # 记录替换细节

    word_saliency_array = np.array([word_tuple[2] for word_tuple in word_saliency_list])
    word_saliency_array = softmax(word_saliency_array)

    NE_candidates = {}
    if dataset_path == "data/imdb":
        NE_candidates = NE_list.L['imdb'][true_y]
    elif dataset_path == "data/ag":
        NE_candidates = NE_list.L['agnews'][true_y]
    else:
        NE_candidates = NE_list.L['agnews'][true_y]

    NE_tags = list(NE_candidates.keys())
    use_NE = True  # 是否使用命名实体替换
    print("-------------pww-------------------4-----")

    # 为每个 token 生成候选替换词
    for (position, token, word_saliency, tag) in word_saliency_list:
        candidates = _generate_synonym_candidates(token=token, token_position=position, rank_fn=rank_fn)
        if len(candidates) == 0:
            continue

        perturbed_text = perturbed_doc.text
        # Eq.(4): score all synonyms for this position in one BERT batch.
        if heuristic_batch_fn is not None:
            candidate_scores = heuristic_batch_fn(doc.text, candidates)
            scored_candidates = list(zip(candidate_scores, candidates))
        else:
            scored_candidates = list(zip(
                map(partial(heuristic_fn, doc.text), candidates),
                candidates
            ))

        # Only the maximum is needed; avoid sorting the whole candidate list.
        delta_p_star, substitute = max(scored_candidates, key=lambda t: float(t[0]))

        # Eq.(7): H(x, x_i^*, w_i)
        substitute_tuple_list.append(
            (position, token.text, substitute, delta_p_star * word_saliency_array[position], token.tag_)
        )

    # 按影响力从高到低排序
    sorted_substitute_tuple_list = sorted(substitute_tuple_list, key=lambda t: t[3], reverse=True)
    print("-------------pww-------------------5-----")

    NE_count = 0  # 命名实体替换计数
    change_tuple_list = []

    # 依次替换生成对抗文本
    for (position, token, substitute, score, tag) in sorted_substitute_tuple_list:
        if nlp(token)[0].ent_type_ in NE_tags:
            NE_count += 1
        # print("-------------pww-------------------5.1-----")

        change_tuple_list.append((position, token, substitute, score, tag))
        # print("-------------pww-------------------5.2-----")
        perturbed_text = ' '.join(_compile_perturbed_tokens(perturbed_doc, [substitute]))
        # print("-------------pww-------------------5.3-----")
        perturbed_doc = nlp(perturbed_text)
        # print("-------------pww-------------------5.4-----")
        substitute_count += 1

        # 若满足终止条件
        if halt_condition_fn(perturbed_text):
            if verbose:
                print("use", substitute_count, "substitution; use", NE_count, 'NE')
            sub_rate = substitute_count / original_length   # ✅ 改为用原始长度
            NE_rate = 0.0 if substitute_count == 0 else NE_count / substitute_count
            return perturbed_text, substitute_count, NE_rate, change_tuple_list
    
    

    # 没有提前终止的情况
    if verbose:
        print("use", substitute_count, "substitution; use", NE_count, 'NE')

    sub_rate = substitute_count / original_length   # ✅ 始终保持一致
    NE_rate = 0.0 if substitute_count == 0 else NE_count / substitute_count
       
    return perturbed_text, substitute_count, NE_rate, change_tuple_list
