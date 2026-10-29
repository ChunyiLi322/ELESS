import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
# from torch.autograd import Variable
import argparse

import math

import numpy as np


import sys
# Input: 
# model: the torch model
# input: the input at current stage 
#        Torch tensor with size (Batchsize,length)
# Output: score with size (batchsize, length)
def replacerandom(model, inputs, pred, classes):
    losses = torch.rand(inputs.size()[0],inputs.size()[1])
    return losses




def random(model, inputs, pred, classes):
    losses = torch.rand(inputs.size()[0],inputs.size()[1])
    return losses    
    
    
    


def replaceone(model, inputs, pred, classes, index2word, candidate_batch_size=128):
    """Leave-one-out scoring for text classification.

    Only attackable positions (token id > 3) are queried. Candidate examples are
    evaluated in mini-batches so BERT/GPU inference is much faster while the
    query counter still counts every candidate example as one target-model query.
    """
    batch_size, seq_len = inputs.size()
    losses = torch.full(
        (batch_size, seq_len),
        float('-inf'),
        device=inputs.device
    )

    pred = pred.reshape(-1).long()

    for k in range(batch_size):
        # Only score positions that are truly attackable by DeepWordBug.
        # This keeps query counting consistent with the actual perturbation stage.
        valid_positions = [
            i for i in range(seq_len)
            if int(inputs[k, i].item()) > 3
            and int(inputs[k, i].item()) in index2word
        ]
        if len(valid_positions) == 0:
            continue
        valid_positions = torch.tensor(valid_positions, dtype=torch.long, device=inputs.device)

        candidates = inputs[k].unsqueeze(0).repeat(valid_positions.numel(), 1)
        row_ids = torch.arange(valid_positions.numel(), device=inputs.device)
        candidates[row_ids, valid_positions] = 2

        candidate_losses = []
        for begin in range(0, candidates.size(0), candidate_batch_size):
            chunk = candidates[begin:begin + candidate_batch_size]

            with torch.no_grad():
                tempoutput = model(chunk.tolist())
                if tempoutput.dim() == 1:
                    tempoutput = tempoutput.unsqueeze(0)

            target = pred[k].repeat(chunk.size(0)).to(tempoutput.device)
            chunk_losses = F.nll_loss(
                tempoutput,
                target,
                reduction='none'
            )
            candidate_losses.append(chunk_losses.to(inputs.device))

        candidate_losses = torch.cat(candidate_losses, dim=0)
        losses[k, valid_positions] = candidate_losses

    return losses



def replaceone_nli(
        premise,
        model,
        inputs,
        pred,
        classes,
        index2word,
        candidate_batch_size=128):

    batch_size, seq_len = inputs.size()

    losses = torch.full(
        (batch_size, seq_len),
        float('-inf'),
        device=inputs.device
    )

    pred = pred.reshape(-1).long()

    for k in range(batch_size):

        # 只处理能够映射回词表的有效 token
        valid_positions = [
            i for i in range(seq_len)
            if int(inputs[k, i].item()) > 3
            and int(inputs[k, i].item()) in index2word
        ]

        if not valid_positions:
            continue

        # 为每个有效位置生成一个替换候选
        candidates = inputs[k].unsqueeze(0).repeat(
            len(valid_positions), 1
        )

        row_ids = torch.arange(
            len(valid_positions),
            device=inputs.device
        )
        col_ids = torch.tensor(
            valid_positions,
            device=inputs.device
        )

        candidates[row_ids, col_ids] = 2

        candidate_losses = []

        for begin in range(
                0, candidates.size(0), candidate_batch_size):

            chunk = candidates[
                begin:begin + candidate_batch_size
            ]

            # 将整数 ID 转换为字符串 token。
            # 与原来的 recoveradv() 逻辑保持一致：
            # ID <= 3 的特殊 token 不放入文本词序列。
            chunk_hypotheses = []

            for row in chunk.tolist():
                tokens = [
                    index2word[int(token_id)]
                    for token_id in row
                    if int(token_id) > 3
                    and int(token_id) in index2word
                ]
                chunk_hypotheses.append(tokens)

            with torch.no_grad():
                tempoutput = model({
                    'premises': [premise] * len(chunk_hypotheses),
                    'hypotheses': chunk_hypotheses
                })

                if tempoutput.dim() == 1:
                    tempoutput = tempoutput.unsqueeze(0)

            target = pred[k].repeat(
                len(chunk_hypotheses)
            ).to(tempoutput.device)

            chunk_losses = F.nll_loss(
                tempoutput,
                target,
                reduction='none'
            )

            candidate_losses.append(
                chunk_losses.to(inputs.device)
            )

        candidate_losses = torch.cat(
            candidate_losses, dim=0
        )

        losses[k, valid_positions] = candidate_losses

    return losses


def temporal(model, inputs, pred, classes):
    losses1 = torch.zeros(inputs.size()[0],inputs.size()[1])
    dloss = torch.zeros(inputs.size()[0],inputs.size()[1])
    for i in range(inputs.size()[1]):
        tempinputs = inputs[:,:i+1]
        with torch.no_grad():
            tempoutput = torch.exp(model(tempinputs))
        losses1[:,i] = tempoutput.gather(1,pred.view(-1,1)).view(-1) 
    dloss[:,0] = losses1[:,0] - 1.0/classes
    for i in range(1,inputs.size()[1]):
        dloss[:,i] = losses1[:,i] - losses1[:,i-1]
    return dloss
    
def temporaltail(model, inputs, pred, classes):
    losses1 = torch.zeros(inputs.size()[0],inputs.size()[1])
    dloss = torch.zeros(inputs.size()[0],inputs.size()[1])
    for i in range(inputs.size()[1]):
        tempinputs = inputs[:,i:]
        with torch.no_grad():
            tempoutput = torch.exp(model(tempinputs))
        losses1[:,i] = tempoutput.gather(1,pred.view(-1,1)).view(-1)
    dloss[:,-1] = losses1[:,-1] - 1.0/classes
    for i in range(inputs.size()[1]-1):
        dloss[:,i] = losses1[:,i] - losses1[:,i+1]
    return dloss
    
def combined(model, inputs, pred, classes):
    temp = temporal(model, inputs, pred, classes)
    temptail = temporaltail(model, inputs, pred, classes)
    return (temp+temptail)/2
    
def grad(model, inputs, pred, classes):
    losses1 = torch.zeros(inputs.size()[0],inputs.size()[1])
    dloss = torch.zeros(inputs.size()[0],inputs.size()[1])
    if isinstance(model,torch.nn.DataParallel):
        model = model.module
    model.train()
    embd,output = model(inputs, returnembd = True)
    # embd.retain_grad()
    loss = F.nll_loss(output,pred)

    loss.backward()
    score = (inputs<=2).float()
    score = -score
    score = embd.grad.norm(2,dim=2) + score * 1e9
    return score

def grad_unconstrained(model, inputs, pred, classes):
    losses1 = torch.zeros(inputs.size()[0],inputs.size()[1])
    dloss = torch.zeros(inputs.size()[0],inputs.size()[1])
    if isinstance(model,torch.nn.DataParallel):
        model = model.module
    model.train()
    embd,output = model(inputs, returnembd = True)
    loss = F.nll_loss(output,pred)

    loss.backward()
    score = embd.grad.norm(2,dim=2)
    return score
    
def scorefunc(name):
    if "temporal" in name:
        return temporal
    elif "tail" in name:
        return temporaltail
    elif "combined" in name:
        return combined
    elif "replaceone" in name:
        return replaceone
    elif "random" in name:
        return random
    elif 'ucgrad' in name:
        return grad_unconstrained
    elif "grad" in name:
        return grad
    else:
        print('No scoring function found')
        sys.exit(1)
