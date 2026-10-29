import torch
import numpy as np

from deepWordBug.scoring import replaceone, replaceone_nli, random, replacerandom
from deepWordBug.transformer import homoglyph


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ModelQueryCounter:
    """Count target-model queries.

    Definition used here: one model prediction for one example = one query.
    A prediction on a batch of B examples therefore adds B queries.
    Random scoring in replacerandom() does not call the target model and adds
    zero queries.
    """

    def __init__(self, model):
        self.model = model
        self.num_queries = 0
        self.num_model_calls = 0

    def reset(self):
        self.num_queries = 0
        self.num_model_calls = 0

    @staticmethod
    def _infer_batch_size(inputs, outputs):
        # Model outputs are normally [batch_size, num_classes].
        if torch.is_tensor(outputs):
            if outputs.dim() >= 2:
                return int(outputs.size(0))
            return 1

        if isinstance(outputs, np.ndarray):
            if outputs.ndim >= 2:
                return int(outputs.shape[0])
            return 1

        # Infer batch size from structured inputs when output shape is unclear.
        if isinstance(inputs, dict):
            for key in ("premises", "hypotheses"):
                value = inputs.get(key)
                if isinstance(value, (list, tuple)):
                    return max(1, len(value))
                if torch.is_tensor(value):
                    return int(value.size(0)) if value.dim() >= 2 else 1

            for value in inputs.values():
                if isinstance(value, (list, tuple)):
                    return max(1, len(value))
                if torch.is_tensor(value):
                    return int(value.size(0)) if value.dim() >= 2 else 1

        if torch.is_tensor(inputs):
            return int(inputs.size(0)) if inputs.dim() >= 2 else 1

        if isinstance(inputs, (list, tuple)):
            return max(1, len(inputs))

        return 1

    def __call__(self, inputs, *args, **kwargs):
        outputs = self.model(inputs, *args, **kwargs)
        self.num_model_calls += 1
        self.num_queries += self._infer_batch_size(inputs, outputs)
        return outputs


def recoveradv(rawsequence, index2word, inputs, advwords):
    """Recover the perturbed text from token ids and perturbed token strings."""

    if isinstance(rawsequence, list):
        rawsequence = ' '.join(rawsequence)

    advsequence = rawsequence[:]

    # Search from left to right so repeated words are not repeatedly matched
    # to the same location.
    search_start = 0

    for i in range(inputs.size()[0]):
        token_id = inputs[i].item()

        if token_id < 3:
            continue

        if token_id not in index2word:
            continue

        wordi = index2word[token_id]

        if not wordi:
            continue

        found_idx = advsequence.find(wordi, search_start)

        if found_idx == -1:
            continue

        replacement = advwords[i]
        if replacement is None or replacement == '':
            replacement = wordi

        advsequence = (
            advsequence[:found_idx]
            + replacement
            + advsequence[found_idx + len(wordi):]
        )

        search_start = found_idx + len(replacement)

    return advsequence


def _to_int_label(label):
    if torch.is_tensor(label):
        if label.numel() == 1:
            return int(label.item())
        return int(label.view(-1)[0].item())
    return int(label)


def deepWordBug(
        numclass,
        dataid,
        data,
        origsample,
        origsampleidx,
        tgt,
        adv,
        wordinput,
        model,
        testword,
        index2word,
        word_index,
        dwbtokenizer):
    """
    DeepWordBug attack for text classification.

    Returns
    -------
    wordinputi : str
        Generated adversarial text.
    num_changed : int
        Number of changed token positions.
    orig_label : torch.Tensor/int
        Prediction on original input.
    new_label : torch.Tensor/int
        Prediction on adversarial input.
    true_label : int
        Ground-truth label.
    raw_text : str/list
        Original raw input.
    num_queries : int
        Number of target-model queries used by this attack.
    """

    num_changed = 0

    inputs, target, idx, raw = data
    inputs = inputs.to(device)
    target = target.to(device)

    origsample.append(inputs)
    origsampleidx.append(idx)
    tgt.append(target)

    counted_model = ModelQueryCounter(model)

    # ------------------------------------------------------------------
    # Original prediction
    # ------------------------------------------------------------------
    output = counted_model(raw).squeeze()
    orig_label = torch.argmax(output)

    # ------------------------------------------------------------------
    # Leave-one-out scoring stage.
    # replaceone() evaluates masked-token candidates in batches. The query
    # counter still counts each candidate example as one model query.
    # ------------------------------------------------------------------
    losses = replaceone(
        counted_model,
        inputs,
        orig_label.reshape(-1),
        numclass,
        index2word
    )

    _, indices = torch.sort(losses, dim=1, descending=True)

    # Keep an untouched copy so num_changed reflects actual token changes
    # rather than the number of homoglyph() calls.
    original_inputs = inputs.clone()
    advinputs = inputs.clone()
    wtmp = []

    for k in range(inputs.size()[0]):
        wtmp.append([])

        for i in range(inputs.size()[1]):
            token_id = advinputs[k, i].item()

            if token_id > 3 and token_id in index2word:
                wtmp[-1].append(index2word[token_id])
            else:
                wtmp[-1].append('')

    # ------------------------------------------------------------------
    # DeepWordBug perturbation stage.
    # Keep power=100 to preserve the original experimental behavior.
    # On short datasets such as MR this may perturb most eligible words.
    # ------------------------------------------------------------------
    for k in range(inputs.size(0)):
        j = 0
        t = 0
        # At most 100 changes, but never more than the number of positions
        # that were actually scored/queryable by replaceone().
        attackable_count = int(torch.isfinite(losses[k]).sum().item())
        power = min(100, attackable_count)

        while j < power and t < inputs.size(1):
            pos = indices[k][t]
            pos_int = int(pos.item()) if torch.is_tensor(pos) else int(pos)

            token_id = advinputs[k, pos_int].item()

            if token_id > 3 and token_id in index2word:
                # Keep homoglyph() API unchanged, but do not use its cumulative
                # counter as the final perturbation count.
                word, new_token_id, _ = homoglyph(
                    token_id,
                    word_index,
                    index2word,
                    0,
                    top_words=20000
                )

                advinputs[k, pos_int] = new_token_id
                wtmp[k][pos_int] = word
                j += 1

            t += 1

    # Count only positions whose token id actually changed.  This prevents
    # repeated/cumulative bookkeeping inside homoglyph() from making the
    # perturbation count larger than the number of attackable positions.
    changed_mask = (original_inputs != advinputs) & (original_inputs > 3)
    num_changed = int(changed_mask.sum().item())

    adv.append(advinputs)

    # ------------------------------------------------------------------
    # Recover adversarial raw text.
    # ------------------------------------------------------------------
    wordinputi = ''

    for i in range(len(wtmp)):
        wordinputi = recoveradv(
            raw[i],
            index2word,
            inputs[i],
            wtmp[i]
        )

    # ------------------------------------------------------------------
    # Final adversarial prediction.
    # ------------------------------------------------------------------
    output2 = counted_model(advinputs.tolist()).squeeze()
    new_label = torch.argmax(output2)

    # Query count includes the original prediction, every leave-one-out
    # candidate evaluated by replaceone(), and the final adversarial prediction.
    num_queries = int(counted_model.num_queries)

    return (
        wordinputi,
        num_changed,
        orig_label,
        new_label,
        _to_int_label(target),
        raw[0],
        num_queries
    )


def sequences_to_texts(sequences, word_index):
    """Convert integer token-id sequences back to whitespace-separated text."""

    index_word = {
        index: word
        for word, index in word_index.items()
    }

    texts = []

    for sequence in sequences:
        words = [
            index_word.get(index, '')
            for index in sequence
            if index > 0
        ]
        texts.append(' '.join(words))

    return texts


def deepWordBug_nli(
        premise,
        hypothese,
        numclass,
        dataid,
        data,
        origsample,
        origsampleidx,
        tgt,
        adv,
        wordinput,
        model,
        testword,
        index2word,
        word_index,
        dwbtokenizer):
    """
    DeepWordBug attack for NLI.

    NLI is kept compatible with the existing pipeline.  It also returns a
    query count as the seventh return value so the calling code can adopt the
    same logging interface later if desired.
    """

    num_changed = 0

    inputs, target, idx, raw = data
    inputs = inputs.to(device)
    target = target.to(device)

    origsample.append(inputs)
    origsampleidx.append(idx)
    tgt.append(target)

    counted_model = ModelQueryCounter(model)

    # Original prediction.
    output = counted_model({
        'premises': [premise],
        'hypotheses': [hypothese]
    }).squeeze()

    orig_label = torch.argmax(output)

    # Leave-one-out scoring for the hypothesis. Candidate hypotheses are
    # evaluated in batches, while each candidate still counts as one query.

    losses = replaceone_nli(
        premise,
        counted_model,
        inputs,
        orig_label.reshape(-1),
        numclass,
        index2word
    )

    _, indices = torch.sort(losses, dim=1, descending=True)

    # Keep an untouched copy so num_changed reflects actual token changes
    # rather than the number of homoglyph() calls.
    original_inputs = inputs.clone()
    advinputs = inputs.clone()
    wtmp = []

    for k in range(inputs.size()[0]):
        wtmp.append([])

        for i in range(inputs.size()[1]):
            token_id = advinputs[k, i].item()

            if token_id > 3 and token_id in index2word:
                wtmp[-1].append(index2word[token_id])
            else:
                wtmp[-1].append('')

    for k in range(inputs.size(0)):
        j = 0
        t = 0
        # At most 100 changes, but never more than the number of positions
        # that were actually scored/queryable by replaceone().
        attackable_count = int(torch.isfinite(losses[k]).sum().item())
        power = min(100, attackable_count)

        while j < power and t < inputs.size(1):
            pos = indices[k][t]
            pos_int = int(pos.item()) if torch.is_tensor(pos) else int(pos)

            token_id = advinputs[k, pos_int].item()

            if token_id > 3 and token_id in index2word:
                # Keep homoglyph() API unchanged, but do not use its cumulative
                # counter as the final perturbation count.
                word, new_token_id, _ = homoglyph(
                    token_id,
                    word_index,
                    index2word,
                    0,
                    top_words=20000
                )

                advinputs[k, pos_int] = new_token_id
                wtmp[k][pos_int] = word
                j += 1

            t += 1

    changed_mask = (original_inputs != advinputs) & (original_inputs > 3)
    num_changed = int(changed_mask.sum().item())

    adv.append(advinputs)

    wordinputi = ''

    for i in range(len(wtmp)):
        wordinputi = recoveradv(
            raw[i],
            index2word,
            inputs[i],
            wtmp[i]
        )

    # Keep the recovered hypothesis exactly as a space-separated string.
    # Do not remove spaces or strip punctuation here: that merges all words.
    sentence = wordinputi
    hypothesis_tokens = sentence.split()

    output2 = counted_model({
        'premises': [premise],
        'hypotheses': [hypothesis_tokens]
    }).squeeze()

    new_label = torch.argmax(output2)
    # Query count includes the original prediction, all leave-one-out
    # hypothesis candidates, and the final adversarial prediction.
    num_queries = int(counted_model.num_queries)

    return (
        sentence,
        num_changed,
        orig_label,
        new_label,
        _to_int_label(target),
        raw[0],
        num_queries
    )
