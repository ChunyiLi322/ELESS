import sys
import time
import keras
import spacy
import numpy as np
import tensorflow as tf

tf.compat.v1.disable_eager_execution()

import os
from keras import backend as K
from pww.paraphrase import _compile_perturbed_tokens, PWWS, PWWS_nli
from pww.word_level_process import text_to_vector, vector_to_text
from pww.evaluate_word_saliency import evaluate_word_saliency, evaluate_word_saliency_nli
from keras.backend.tensorflow_backend import set_session
from pww.unbuffered import Unbuffered

import torch
import torch.nn as nn
import torch.nn.functional as F


sys.stdout = Unbuffered(sys.stdout)

# Keep the original spaCy pipeline used by the project.
# nlp = spacy.load('en_core_web_sm', disable=['tagger', 'ner'])
nlp = spacy.load('en_core_web_sm', disable=['parser'])
# nlp = spacy.load('en_core_web_sm', disable=['tagger', 'parser', 'ner', 'lemmatizer', 'attribute_ruler'])


class ModelQueryCounter:
    """
    Wrapper for counting target-model queries.

    Counting convention used by the paper-ready evaluation:
        - prediction for one example = 1 query
        - prediction for a batch of B examples = B queries

    The wrapper is passed into evaluate_word_saliency(), heuristic_fn(),
    and halt_condition_fn(), so model calls inside those stages are counted.
    """

    def __init__(self, model):
        self.model = model
        self.num_queries = 0

    def reset(self):
        self.num_queries = 0

    def _infer_batch_size(self, inputs, outputs):
        # Prefer output shape when possible.
        if torch.is_tensor(outputs):
            if outputs.dim() <= 1:
                return 1
            return int(outputs.size(0))

        if isinstance(outputs, np.ndarray):
            if outputs.ndim <= 1:
                return 1
            return int(outputs.shape[0])

        # Fallback to input structure.
        if torch.is_tensor(inputs):
            if inputs.dim() <= 1:
                return 1
            return int(inputs.size(0))

        if isinstance(inputs, dict):
            for value in inputs.values():
                if isinstance(value, (list, tuple)):
                    return max(1, len(value))
                if torch.is_tensor(value):
                    if value.dim() <= 1:
                        return 1
                    return int(value.size(0))

        if isinstance(inputs, (list, tuple)):
            return max(1, len(inputs))

        return 1

    def __call__(self, inputs, *args, **kwargs):
        outputs = self.model(inputs, *args, **kwargs)
        self.num_queries += self._infer_batch_size(inputs, outputs)
        return outputs


class ForwardGradWrapper(nn.Module):
    def __init__(self,
                 pretrained_dir,
                 nclasses,
                 max_seq_length=128,
                 batch_size=32):
        # NOTE: this class is retained from the original project.  It is not
        # used by the WordCNN/WordLSTM PWW path in attack_classification.py.
        super(ForwardGradWrapper, self).__init__()
        print("---------------------pretrained_dir---------", pretrained_dir)
        self.model = BertForSequenceClassification.from_pretrained(
            pretrained_dir,
            num_labels=nclasses
        ).cuda()

        # construct dataset loader
        self.dataset = NLIDataset_BERT(
            pretrained_dir,
            max_seq_length=max_seq_length,
            batch_size=batch_size
        )

    def predict_prob(self, text_data, batch_size=32):
        # Switch the model to eval mode.
        self.model.eval()
        # transform text data into indices and create batches
        dataloader = self.dataset.transform_text(text_data, batch_size=batch_size)

        probs_all = []
        for input_ids, input_mask, segment_ids in dataloader:
            input_ids = input_ids.cuda()
            input_mask = input_mask.cuda()
            segment_ids = segment_ids.cuda()

            with torch.no_grad():
                logits = self.model(input_ids, segment_ids, input_mask)
                probs = nn.functional.softmax(logits, dim=-1)
                probs_all.append(probs)

        return torch.cat(probs_all, dim=0)

    def predict_classes(self, text_data, batch_size=32):
        orig_probs = self.predict_prob(text_data, batch_size=batch_size).squeeze()
        orig_label = torch.argmax(orig_probs)
        return orig_label


def PWW(input_text, true_y, grad_guide, tokenizer, dataset, dataset_path, verbose=True):
    """
    PWWS attack for text classification.

    The attack greedily chooses synonym substitutions according to saliency
    and probability change.  This version additionally records paper-ready
    target-model query count and runtime.

    Returns
    -------
    perturbed_text : str
        Generated adversarial example.
    num_changed : int
        Number of actually substituted word positions.
    orig_label : torch.Tensor/int
        Prediction on the original input.
    perturbed_y : torch.Tensor/int
        Prediction on the final adversarial input.
    num_queries : int
        Total number of target-model example predictions used by the attack.
    runtime_sec : float
        Wall-clock runtime of the attack for this example.
    """

    started = time.perf_counter()

    # Count all target-model predictions made by PWW.
    counted_grad_guide = ModelQueryCounter(grad_guide)

    def halt_condition_fn(perturbed_text):
        """
        Halt if model output is changed.
        """
        perturbed_vector = text_to_vector(perturbed_text, tokenizer)
        perturbed_o = counted_grad_guide(
            [vector_to_text(perturbed_vector, tokenizer)]
        ).squeeze()
        adv_y = torch.argmax(perturbed_o)

        if adv_y != true_y:
            return True
        else:
            return False

    def heuristic_fn(text, candidate):
        """Scalar fallback scorer; uses cached original probability."""
        perturbed_tokens = _compile_perturbed_tokens(doc, [candidate])
        perturbed_vector = text_to_vector(' '.join(perturbed_tokens), tokenizer)
        perturbed_prob = counted_grad_guide(
            [vector_to_text(perturbed_vector, tokenizer)]
        ).squeeze()
        return orig_proc[true_y] - perturbed_prob[true_y]

    def heuristic_batch_fn(text, candidates):
        """Score all synonyms of one token using a single BERT batch."""
        if not candidates:
            return []

        model_inputs = []
        for candidate in candidates:
            perturbed_tokens = _compile_perturbed_tokens(doc, [candidate])
            perturbed_vector = text_to_vector(' '.join(perturbed_tokens), tokenizer)
            model_inputs.append(vector_to_text(perturbed_vector, tokenizer))

        perturbed_probs = counted_grad_guide(model_inputs)
        if torch.is_tensor(perturbed_probs) and perturbed_probs.dim() == 1:
            perturbed_probs = perturbed_probs.unsqueeze(0)

        deltas = orig_proc[true_y] - perturbed_probs[:, true_y]
        return [float(x.detach().cpu().item()) for x in deltas]

    # -------------------------------------------------------------
    # Original prediction: 1 target-model query for this example.
    # -------------------------------------------------------------
    orig_proc = counted_grad_guide([input_text]).squeeze()
    orig_label = torch.argmax(orig_proc)

    doc = nlp(" ".join(input_text))
    print("-------------pww-------------------1-----")

    # -------------------------------------------------------------
    # Word-saliency stage.
    # IMPORTANT: pass counted_grad_guide instead of grad_guide so
    # model calls made inside evaluate_word_saliency() are counted.
    # -------------------------------------------------------------
    position_word_list, word_saliency_list = evaluate_word_saliency(
        doc,
        input_text,
        counted_grad_guide,
        tokenizer,
        true_y,
        dataset
    )

    print("-------------pww-------------------1.2-----")

    # -------------------------------------------------------------
    # PWWS synonym replacement stage.
    # heuristic_fn() and halt_condition_fn() both use the counted model.
    # -------------------------------------------------------------
    perturbed_text, sub_rate, NE_rate, change_tuple_list = PWWS(
        doc,
        true_y,
        dataset,
        dataset_path,
        word_saliency_list=word_saliency_list,
        heuristic_fn=heuristic_fn,
        heuristic_batch_fn=heuristic_batch_fn,
        halt_condition_fn=halt_condition_fn,
        verbose=verbose
    )

    # -------------------------------------------------------------
    # Final adversarial prediction.
    # -------------------------------------------------------------
    perturbed_vector = text_to_vector(perturbed_text, tokenizer)
    perturbed_o = counted_grad_guide(
        [vector_to_text(perturbed_vector, tokenizer)]
    ).squeeze()
    perturbed_y = torch.argmax(perturbed_o)

    # IMPORTANT:
    # sub_rate is a replacement RATE, not the number of changed tokens.
    # change_tuple_list contains the substitutions actually made.
    num_changed = len(change_tuple_list)

    # Real target-model query count, not number of replacements.
    num_queries = int(counted_grad_guide.num_queries)

    runtime_sec = float(time.perf_counter() - started)

    return (
        perturbed_text,
        num_changed,
        orig_label,
        perturbed_y,
        num_queries,
        runtime_sec
    )




def PWW_nli(
        premise,
        input_text,
        true_y,
        grad_guide,
        tokenizer,
        dataset,
        dataset_path,
        verbose=True):
    """
    PWWS attack for NLI.

    Statistics are aligned with the classification PWW implementation.

    Returns
    -------
    perturbed_text : str
        Generated adversarial hypothesis.
    num_changed : int
        Number of actually changed token positions.
    orig_label : torch.Tensor/int
        Prediction on the original premise-hypothesis pair.
    perturbed_y : torch.Tensor/int
        Prediction on the final adversarial pair.
    num_queries : int
        Total number of target-model example predictions.
    runtime_sec : float
        Wall-clock runtime for this example.
    """

    # -------------------------------------------------------------
    # Start timing exactly as classification PWW does.
    # -------------------------------------------------------------
    started = time.perf_counter()

    # Count every target-model prediction.
    counted_grad_guide = ModelQueryCounter(grad_guide)

    # -------------------------------------------------------------
    # Stop condition
    # -------------------------------------------------------------
    def halt_condition_fn(premise, perturbed_text):
        """
        Halt once the prediction changes.
        """
        perturbed_vector = text_to_vector(
            perturbed_text,
            tokenizer
        )

        perturbed_o = counted_grad_guide({
            'premises': [premise],
            'hypotheses': [
                vector_to_text(
                    perturbed_vector,
                    tokenizer
                )
            ]
        }).squeeze()

        adv_y = torch.argmax(perturbed_o)

        return adv_y != true_y

    # -------------------------------------------------------------
    # Candidate scoring
    #
    # Important:
    # classification PWW caches orig_proc instead of repeatedly
    # evaluating the original sentence. Do the same here.
    # -------------------------------------------------------------
    def heuristic_fn(text, candidate):
        perturbed_tokens = _compile_perturbed_tokens(
            doc,
            [candidate]
        )

        perturbed_vector = text_to_vector(
            ' '.join(perturbed_tokens),
            tokenizer
        )

        perturbed_prob = counted_grad_guide({
            'premises': [premise],
            'hypotheses': [
                vector_to_text(
                    perturbed_vector,
                    tokenizer
                )
            ]
        }).squeeze()

        delta_p = (
            orig_proc[true_y]
            - perturbed_prob[true_y]
        )

        return delta_p

    # -------------------------------------------------------------
    # Original prediction
    # 1 target-model query
    # -------------------------------------------------------------
    orig_proc = counted_grad_guide({
        'premises': [premise],
        'hypotheses': [input_text]
    }).squeeze()

    orig_label = torch.argmax(orig_proc)

    # Same token handling as original implementation.
    doc = nlp(" ".join(input_text))

    # -------------------------------------------------------------
    # Word saliency
    #
    # Critical:
    # counted_grad_guide MUST be passed here.
    # Otherwise saliency queries are not counted.
    # -------------------------------------------------------------
    position_word_list, word_saliency_list = (
        evaluate_word_saliency_nli(
            doc,
            premise,
            input_text,
            counted_grad_guide,
            tokenizer,
            true_y,
            dataset
        )
    )

    # -------------------------------------------------------------
    # PWWS synonym replacement
    # -------------------------------------------------------------
    perturbed_text, sub_rate, NE_rate, change_tuple_list = PWWS_nli(
        doc,
        premise,
        true_y,
        dataset,
        dataset_path,
        word_saliency_list=word_saliency_list,
        heuristic_fn=heuristic_fn,
        halt_condition_fn=halt_condition_fn,
        verbose=verbose
    )

    # -------------------------------------------------------------
    # Final adversarial prediction
    # -------------------------------------------------------------
    perturbed_vector = text_to_vector(
        perturbed_text,
        tokenizer
    )

    perturbed_o = counted_grad_guide({
        'premises': [premise],
        'hypotheses': [
            vector_to_text(
                perturbed_vector,
                tokenizer
            )
        ]
    }).squeeze()

    perturbed_y = torch.argmax(perturbed_o)

    # -------------------------------------------------------------
    # Paper-ready statistics
    # -------------------------------------------------------------

    # sub_rate is a ratio, not changed-token count.
    num_changed = len(change_tuple_list)

    # Actual number of target-model predictions.
    num_queries = int(
        counted_grad_guide.num_queries
    )

    # Actual wall-clock attack runtime.
    runtime_sec = float(
        time.perf_counter() - started
    )

    return (
        perturbed_text,
        num_changed,
        orig_label,
        perturbed_y,
        num_queries,
        runtime_sec
    )