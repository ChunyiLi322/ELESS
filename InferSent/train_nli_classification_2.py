import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from models import NLINet, InferSent
from torch.nn.utils.rnn import pad_sequence
import string

# Download NLTK data for tokenization
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# 定义一个编码器字典
encoder_dict = {
    'InferSent': InferSent,
    # 可以在这里添加更多编码器类
}

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

    def text_pred(self, text_data, datasize = 0):
        # Switch the model to eval mode.
        self.model.eval()

        # transform text data into indices and create batches
        data_batches = self.dataset.transform_text2(text_data)

        # Deactivate autograd for evaluation.
        probs_all = []

        if datasize == 0:

            with torch.no_grad():
                for batch in data_batches:
                    # Move input and output data to the GPU if one is used.
                    (s1_batch, s1_len), (s2_batch, s2_len) = batch
                    s1_batch, s2_batch = s1_batch.cuda(), s2_batch.cuda()
                    logits = self.model((s1_batch, s1_len), (s2_batch, s2_len))
                    probs = nn.functional.log_softmax(logits, dim=-1)
                    # probs_all.append(probs)

            # return torch.cat(probs_all, dim=0)
            return probs


        else:

            with torch.no_grad():
                for batch in data_batches:
                    # Move input and output data to the GPU if one is used.
                    (s1_batch, s1_len), (s2_batch, s2_len) = batch
                    s1_batch, s2_batch = s1_batch.cuda(), s2_batch.cuda()
                    logits = self.model((s1_batch, s1_len), (s2_batch, s2_len))
                    probs = nn.functional.log_softmax(logits, dim=-1)
                    probs_all.append(probs)

            return torch.cat(probs_all, dim=0)


class NLIDataset_InferSent(Dataset):
    """
    Dataset class for Natural Language Inference datasets.
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
            embedding_path: Path to the embedding file.
            word_emb_dim: Dimension of word embeddings.
            batch_size: Size of each batch.
            bos: Beginning of sentence token.
            eos: End of sentence token.
        """
        self.bos = bos
        self.eos = eos
        self.word_emb_dim = word_emb_dim
        self.batch_size = batch_size

        # build word dict
        self.word_vec = self.build_vocab(data['premises'] + data['hypotheses'], embedding_path)

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
        try:
            with open(embedding_path, 'r', encoding='utf8') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < self.word_emb_dim + 1:
                        continue  # Skip invalid lines
                    word = parts[0]
                    vec = parts[1:]
                    if word in word_dict:
                        word_vec[word] = np.array(list(map(float, vec)))
        except FileNotFoundError:
            print(f"Embedding file not found at {embedding_path}")
            raise

        print('Found {0}(/{1}) words with embedding vectors'.format(
            len(word_vec), len(word_dict)))
        return word_vec

    def get_batch(self, batch, word_vec, emb_dim=200):
        # sent in batch in decreasing order of lengths (bsize, max_len, word_dim)
        lengths = np.array([len(x) for x in batch])
        max_len = np.max(lengths)
        embed = np.zeros((max_len, len(batch), emb_dim))

        for i in range(len(batch)):
            for j in range(len(batch[i])):
                if batch[i][j] in word_vec:
                    embed[j, i, :] = word_vec[batch[i][j]]
                else:
                    embed[j, i, :] = word_vec['<oov>']

        return torch.from_numpy(embed).float(), lengths

    def transform_text(self, data):
        # transform data into seq of embeddings
        premises = data['premises']
        hypotheses = data['hypotheses']
        labels = data['labels']

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
            labels_batch = labels[stidx:stidx + self.batch_size]
            labels_batch = torch.tensor(labels_batch, dtype=torch.long)
            batches.append(((s1_batch, s1_len), (s2_batch, s2_len), labels_batch))

        return batches

    def transform_text2(self, data):
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

# Function to save the model
def save_model(model, filepath):
    torch.save(model.state_dict(), filepath)
    print(f"Model saved to {filepath}")

def read_data(filepath, data_size=20000, target_model='InferSent', lowercase=False, ignore_punctuation=False, stopwords=[]):
    """
    Read the premises, hypotheses and labels from some NLI dataset's
    file and return them in a dictionary. The file should be in the same
    form as SNLI's .txt files.

    Args:
        filepath: The path to a file containing some premises, hypotheses
            and labels that must be read. The file should be formatted in
            the same way as the SNLI (and MultiNLI) dataset.
        data_size: Maximum number of data points to read. Set to None to read all.
        target_model: The target model type ('bert', 'InferSent', etc.).
        lowercase: Whether to convert text to lowercase.
        ignore_punctuation: Whether to remove punctuation from text.
        stopwords: List of stopwords to remove.
    Returns:
        A dictionary containing three lists: premises, hypotheses, and labels.
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
            # if data_size is not None and idx >= data_size:
            #     break

            line = line.strip().split('\t')

            # Ensure that the line has enough columns
            if len(line) < 3:
                continue

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

    #         # Each premise and hypothesis is split into a list of words.
    #         premises.append([w for w in premise.rstrip().split()
    #                          if w not in stopwords])
    #         hypotheses.append([w for w in hypothesis.rstrip().split()
    #                            if w not in stopwords])
    #         labels.append(labeldict.get(line[0], -1))  # 默认-1表示未知标签
    #
    #
    # # 过滤掉标签为-1的数据
    # filtered_data = {"premises": [], "hypotheses": [], "labels": []}
    # for p, h, l in zip(premises, hypotheses, labels):
    #     if l != -1:
    #         filtered_data["premises"].append(p)
    #         filtered_data["hypotheses"].append(h)
    #         filtered_data["labels"].append(l)

    return {"premises": premises,
            "hypotheses": hypotheses,
            "labels": labels}





# Main function to execute the training and evaluation
def main():
    # Load the dataset
    train_data_path = '/home/lcy/ELESS/data/snli'
    # train_data_path = '/home/lcy/ELESS/data/mnli_mismatched'

    data = read_data(train_data_path, target_model='InferSent')


    print("Data import finished!")

    # 拆分训练集和验证集
    premises = data['premises']
    hypotheses = data['hypotheses']
    labels = data['labels']


    train_data = {
        "premises": premises,
        "hypotheses": hypotheses,
        "labels": labels
    }

    val_data = {
        "premises": premises,
        "hypotheses": hypotheses,
        "labels": labels
    }

    # train_premises, val_premises, train_hypotheses, val_hypotheses, train_labels, val_labels = train_test_split(
    #     premises, hypotheses, labels, test_size=0.01)

    # train_data = {
    #     "premises": train_premises,
    #     "hypotheses": train_hypotheses,
    #     "labels": train_labels
    # }
    # val_data = {
    #     "premises": val_premises,
    #     "hypotheses": val_hypotheses,
    #     "labels": val_labels
    # }

    # print("-------------------------------------------------------",len(train_data['hypotheses']))
    # print("-------------------------------------------------------",len(val_data['hypotheses']))

    # 加载 GloVe embeddings
    embedding_path = '/home/lcy/ELESS/embedding/glove.6B.200d.txt'

    # 初始化数据集
    train_dataset = NLIDataset_InferSent(embedding_path, data=train_data, batch_size=32)
    # val_dataset = NLIDataset_InferSent(embedding_path, data=val_data, batch_size=32)
    val_dataset = NLIDataset_InferSent(embedding_path, data=train_data, batch_size=1)

    # 将数据转换为批次
    train_batches = train_dataset.transform_text(train_data)
    # print("-------------------------------------------------------",train_batches)
    # print("-------------------------------------------------------",train_batches2)

    val_batches = val_dataset.transform_text(val_data)
    print("Get Embeddings!")
    


    # 定义模型配置
    config = {
        'word_emb_dim': 200,
        'enc_lstm_dim': 2048,
        'n_enc_layers': 1,
        'dpout_model': 0.,
        'dpout_fc': 0.,
        'fc_dim': 512,
        'bsize': 32,
        'n_classes': 3,
        'pool_type': 'max',
        'nonlinear_fc': 1,
        'encoder_type': 'InferSent',
        'use_cuda': True,
        'use_target': False,
        'version': 1,
    }

    # 初始化模型
    model = NLINet(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print("Model built!")

    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-7)
    # # criterion = nn.CrossEntropyLoss()
    criterion = nn.NLLLoss()




    # 训练循环
    for epoch in range(150):
        model.train()
        total_loss = 0



        for batch in train_batches:
            prod_all = []
            (s1_batch, s1_len), (s2_batch, s2_len), labels = batch
            optimizer.zero_grad()

            s1_batch = s1_batch.to(device)
            s2_batch = s2_batch.to(device)
            labels = labels.to(device)

            outputs = model((s1_batch, s1_len), (s2_batch, s2_len))

            # 计算 log_softmax
            log_probs = nn.functional.log_softmax(outputs, dim=-1)

            # 使用 NLLLoss 计算损失
            loss = criterion(log_probs, labels)*2
            
            total_loss += loss.item()

            loss.backward()
            optimizer.step()
            # print("---------------------",labels)
            # print("---------------------",preds_label)

        avg_loss = total_loss / len(train_batches)
        print(f'Epoch {epoch+1}, Loss: {avg_loss:.4f}')



    # save_model(model, f'nli_model_epoch_100.pt')
    save_model(model, f'snli_mis_model_epoch_150.pt')


    # def evaluate(model, val_batches):
    #     model.eval()
    #     total_correct = 0
    #     total_samples = 0
    #
    #     with torch.no_grad():
    #         for batch in val_batches:
    #             (s1_batch, s1_len), (s2_batch, s2_len), labels = batch
    #
    #             s1_batch = s1_batch.to(device)
    #             s2_batch = s2_batch.to(device)
    #             labels = labels.to(device)
    #
    #             logits = model((s1_batch, s1_len), (s2_batch, s2_len))
    #             probs = nn.functional.log_softmax(logits, dim=-1)
    #
    #             # 找到每个样本的预测标签
    #             pred_labels = torch.argmax(probs, dim=-1)
    #
    #             # 计算正确预测的数量
    #             total_correct += (pred_labels == labels).sum().item()
    #             total_samples += labels.size(0)
    #
    #     accuracy = total_correct / total_samples
    #     return accuracy
    #
    #
    #
    # val_accuracy = evaluate(model, val_batches)
    # print(f'Epoch {epoch+1}, Validation Accuracy: {val_accuracy:.4f}')
    orig_failures = 0
    data = read_data(train_data_path, data_size=32, target_model='infersent')
    # model = NLI_infer_InferSent('/home/lcy/ELESS/InferSent/mnli_m_model_epoch_100.pt',
    #                                 '/home/lcy/ELESS/embedding/glove.6B.200d.txt',
    #                                 data=data,
    #                                 batch_size=32)
    predictor = model.text_pred

    for idx, premise in enumerate(data['premises']):

        hypothese, true_label = data['hypotheses'][idx], data['labels'][idx]

        #print("-----------------premises-------------------",premise)
        #print("-----------------hypothese-------------------",hypothese)
        #print("-----------------true_label-------------------",true_label)

        orig_probs = predictor({'premises': [premise], 'hypotheses': [hypothese]})
        orig_label = torch.argmax(orig_probs, dim=-1).item()


        #print("-----------------orig_label-------------------",orig_label)

        # if true_label != orig_label:
            orig_failures += 1

    print("----------------准确度-------------------",(1-orig_failures/len(data['premises']))*100)



if __name__ == "__main__":
    main()
