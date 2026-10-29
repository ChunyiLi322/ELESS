# coding: utf-8
import copy
import spacy
import torch
from pww.word_level_process import text_to_vector, vector_to_text

# PWW needs POS/NER elsewhere, but saliency itself does not use spaCy pipeline here.
nlp = spacy.load('en_core_web_sm', disable=['parser'])


def _ensure_2d_probs(probs):
    if torch.is_tensor(probs) and probs.dim() == 1:
        return probs.unsqueeze(0)
    return probs


def evaluate_word_saliency(doc, input_text, grad_guide, tokenizer, input_y, dataset):
    """PWWS word saliency with batched target-model inference.

    This is mathematically equivalent to the original implementation, but all
    leave-one-word-out examples are sent to BERT in one batch instead of one
    forward pass per token. Query counting is unchanged because a batch of B
    examples still counts as B model queries.
    """
    text = ' '.join(doc[position].text for position in range(len(doc)))
    origin_vector = text_to_vector(text, tokenizer)
    origin_prob = grad_guide([input_text]).squeeze()

    masked_inputs = []
    for position in range(len(doc)):
        without_word_vector = copy.deepcopy(origin_vector)
        without_word_vector[0][position] = 0
        masked_inputs.append(vector_to_text(without_word_vector, tokenizer))

    if masked_inputs:
        probs_without = _ensure_2d_probs(grad_guide(masked_inputs))
    else:
        probs_without = None

    word_saliency_list = []
    for position in range(len(doc)):
        word_saliency = origin_prob[input_y] - probs_without[position, input_y]
        word_saliency_list.append(
            (position, doc[position], word_saliency, doc[position].tag_)
        )

    position_word_list = [(word[0], word[1]) for word in word_saliency_list]
    return position_word_list, word_saliency_list


def evaluate_word_saliency_nli(doc, premise, input_text, grad_guide, tokenizer, input_y, dataset):
    """Batched NLI saliency variant; keeps the original public interface."""
    text = ' '.join(doc[position].text for position in range(len(doc)))
    origin_vector = text_to_vector(text, tokenizer)
    origin_prob = grad_guide({
        'premises': [premise],
        'hypotheses': [input_text]
    }).squeeze()

    masked_hypotheses = []
    for position in range(len(doc)):
        without_word_vector = copy.deepcopy(origin_vector)
        without_word_vector[0][position] = 0
        masked_hypotheses.append(vector_to_text(without_word_vector, tokenizer))

    if masked_hypotheses:
        probs_without = _ensure_2d_probs(grad_guide({
            'premises': [premise] * len(masked_hypotheses),
            'hypotheses': masked_hypotheses
        }))
    else:
        probs_without = None

    word_saliency_list = []
    for position in range(len(doc)):
        word_saliency = origin_prob[input_y] - probs_without[position, input_y]
        word_saliency_list.append(
            (position, doc[position], word_saliency, doc[position].tag_)
        )

    position_word_list = [(word[0], word[1]) for word in word_saliency_list]
    return position_word_list, word_saliency_list
