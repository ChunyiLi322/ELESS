import difflib
import Levenshtein
import numpy as np
import csv
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import jaccard_score

def similar_lvst_jaro(str1, str2):
    return Levenshtein.jaro(str1, str2)


def similar_lvst_hamming(str1, str2):
    return Levenshtein.hamming(str1, str2)


def similar_lvst_distance(str1, str2):
    return Levenshtein.distance(str1, str2)


def get_file_textlist(file_path):
    file_textlist = []
    with open(file_path, "r", encoding='utf-8') as file:
        for line in file:
            line = line.strip('\n')
            file_textlist.append(line)
        file.close()
    return " ".join(file_textlist)


def record_sim(result_list, file_path):
    with open(file_path, 'a') as f:
        for i in result_list:
            f.write(i + ',')
        f.write('\n')
        f.close()

def read_sentences_from_file(file_path):
    original_sentences = []
    advanced_sentences = []
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        for i in range(0, len(lines), 3):  # 每三行一个段落，其中有一个空行
            original_sentences.append(lines[i].split(':')[1].strip())
            advanced_sentences.append(lines[i + 1].split(':')[1].strip())
    return original_sentences, advanced_sentences
#

#adversaries_mr_bert_ourmodel

# list1, list2 = read_sentences_from_file('ourmodel/adversaries_ag_bert_ourmodel.txt')
# list1, list2 = read_sentences_from_file('ourmodel/adversaries_ag_cnn_ourmodel.txt')
# list1, list2 = read_sentences_from_file('ourmodel/adversaries_ag_lstm_ourmodel.txt')
# list1, list2 = read_sentences_from_file('ourmodel/adversaries_fake_bert_ourmodel.txt')
# list1, list2 = read_sentences_from_file('ourmodel/adversaries_fake_cnn_150_ourmodel.txt')
# list1, list2 = read_sentences_from_file('ourmodel/adversaries_fake_lstm_ourmodel.txt')
# list1, list2 = read_sentences_from_file('ourmodel/adversaries_imdb_bert_ourmodel.txt')
# list1, list2 = read_sentences_from_file('ourmodel/adversaries_imdb_cnn_orgmodel.txt')
# list1, list2 = read_sentences_from_file('ourmodel/adversaries_imdb_lstm_ourmodel.txt')
# list1, list2 = read_sentences_from_file('ourmodel/adversaries_mr_bert_ourmodel.txt')
# list1, list2 = read_sentences_from_file('ourmodel/adversaries_mr_cnn_ourmodel.txt')
# list1, list2 = read_sentences_from_file('ourmodel/adversaries_mr_lstm_ourmodel.txt')
# list1, list2 = read_sentences_from_file('ourmodel/adversaries_yelp_bert_ourmodel.txt')
# list1, list2 = read_sentences_from_file('ourmodel/adversaries_yelp_cnn_ourmodel.txt')
# list1, list2 = read_sentences_from_file('ourmodel/adversaries_yelp_lstm_ourmodel.txt')


# list1, list2 = read_sentences_from_file('duibi/adversaries_ag_bert_pww_model.txt')
# list1, list2 = read_sentences_from_file('duibi/adversaries_ag_bert_tfl_model.txt')
# list1, list2 = read_sentences_from_file('duibi/adversaries_ag_cnn_pww_model.txt')
# list1, list2 = read_sentences_from_file('duibi/adversaries_ag_cnn_tfl_model.txt')
# list1, list2 = read_sentences_from_file('duibi/adversaries_ag_lstm_tfl_model.txt')
# list1, list2 = read_sentences_from_file('duibi/adversaries_ag_lstm_pww_model.txt')

# list1, list2 = read_sentences_from_file('duibi/adversaries_imdb_bert_pww_model.txt')
# list1, list2 = read_sentences_from_file('duibi/adversaries_imdb_bert_tfl_model.txt')
# list1, list2 = read_sentences_from_file('duibi/adversaries_imdb_cnn_pww_model.txt')
# list1, list2 = read_sentences_from_file('duibi/adversaries_imdb_cnn_tfl_model.txt')
# list1, list2 = read_sentences_from_file('duibi/adversaries_imdb_lstm_pww_model.txt')
# list1, list2 = read_sentences_from_file('duibi/adversaries_imdb_lstm_tfl_model.txt')




# list1, list2 = read_sentences_from_file('duibi/adversaries_imdb_bert_dwb_model.txt')
# list1, list2 = read_sentences_from_file('duibi/adversaries_imdb_lstm_dwbmodel.txt')
# list1, list2 = read_sentences_from_file('duibi/adversaries_imdb_cnn_dwbmodel.txt')
# list1, list2 = read_sentences_from_file('duibi/adversaries_ag_lstm_dwbmodel.txt')
# list1, list2 = read_sentences_from_file('duibi/adversaries_ag_bert_dwb_model.txt')
# list1, list2 = read_sentences_from_file('duibi/adversaries_ag_cnn_dwbmodel.txt')



# list1, list2 = read_sentences_from_file('duibi/adversaries_mnli_mis_esmi_ourmodel.txt')
# list1, list2 = read_sentences_from_file('duibi/adversaries_mnli_m_esmi_ourmodel.txt')
# list1, list2 = read_sentences_from_file('duibi/adversaries_snli_esmi_ourmodel.txt')
# list1, list2 = read_sentences_from_file('duibi/adversaries_mnli_mis_infersent_ourmodel.txt')
# list1, list2 = read_sentences_from_file('duibi/adversaries_mnli_m_infersent_ourmodel.txt')
# list1, list2 = read_sentences_from_file('duibi/adversaries_snil_infersent_ourmodel.txt)
# list1, list2 = read_sentences_from_file('duibi/adversaries_mnli_mm_BERT_ourmodel.txt')
# list1, list2 = read_sentences_from_file('duibi/adversaries_mnli_m_BERT_ourmodel.txt')
# list1, list2 = read_sentences_from_file('duibi/adversaries_snli_BERT_ourmodel.txt')



# list1, list2 = read_sentences_from_file('duibi/adversaries_ag_bert_th_model.txt')
# list1, list2 = read_sentences_from_file('duibi/adversaries_ag_cnn_th_model.txt')
# list1, list2 = read_sentences_from_file('duibi/adversaries_ag_lstm_th_model.txt')
# list1, list2 = read_sentences_from_file('duibi/adversaries_imdb_bert_th_model.txt')
# list1, list2 = read_sentences_from_file('duibi/adversaries_imdb_cnn_th_model.txt')
# list1, list2 = read_sentences_from_file('duibi/adversaries_imdb_lstm_th_model.txt')



# list1, list2 = read_sentences_from_file('xiaorong/adversaries_ag_cnn_ourmodel_1000_0002_10.txt')
# list1, list2 = read_sentences_from_file('xiaorong/adversaries_ag_cnn_ourmodel_1000_0002_30.txt')
# list1, list2 = read_sentences_from_file('xiaorong/adversaries_ag_cnn_ourmodel_1000_0002_50.txt')
# list1, list2 = read_sentences_from_file('xiaorong/adversaries_ag_cnn_ourmodel_1000_0002_100.txt')
# list1, list2 = read_sentences_from_file('xiaorong/adversaries_ag_cnn_ourmodel_1_0002_10.txt')
# list1, list2 = read_sentences_from_file('xiaorong/adversaries_ag_cnn_ourmodel_10_0002_10.txt')
# list1, list2 = read_sentences_from_file('xiaorong/adversaries_ag_cnn_ourmodel_100_0002_10.txt')
# list1, list2 = read_sentences_from_file('xiaorong/adversaries_imdb_cnn_ourmodel_1000_0002_10.txt')
# list1, list2 = read_sentences_from_file('xiaorong/adversaries_imdb_cnn_ourmodel_1000_0002_30.txt')
# list1, list2 = read_sentences_from_file('xiaorong/adversaries_imdb_cnn_ourmodel_1000_0002_50.txt')
# list1, list2 = read_sentences_from_file('xiaorong/adversaries_imdb_cnn_ourmodel_1000_0002_100.txt')
# list1, list2 = read_sentences_from_file('xiaorong/adversaries_imdb_cnn_ourmodel_1_0002_10.txt')
# list1, list2 = read_sentences_from_file('xiaorong/adversaries_imdb_cnn_ourmodel_10_0002_10.txt')
# list1, list2 = read_sentences_from_file('xiaorong/adversaries_imdb_cnn_ourmodel_100_0002_10.txt')


# list1, list2 = read_sentences_from_file('duibi/adversaries_ag_bert_li.txt')
# list1, list2 = read_sentences_from_file('duibi/adversaries_ag_cnn_li.txt')
# list1, list2 = read_sentences_from_file('duibi/adversaries_ag_lstm_li.txt')
#
# list1, list2 = read_sentences_from_file('duibi/adversaries_imdb_bert_li.txt')
# list1, list2 = read_sentences_from_file('duibi/adversaries_imdb_cnn_li.txt')
# list1, list2 = read_sentences_from_file('duibi/adversaries_imdb_lstm_li.txt')
#
# list1, list2 = read_sentences_from_file('duibi/adversaries_mr_bert_li.txt')
# list1, list2 = read_sentences_from_file('duibi/adversaries_mr_cnn_li.txt')
# list1, list2 = read_sentences_from_file('duibi/adversaries_mr_lstm_li.txt')
#
# list1, list2 = read_sentences_from_file('duibi/adversaries_yelp_bert_li.txt')
# list1, list2 = read_sentences_from_file('duibi/adversaries_yelp_cnn_li.txt')
# list1, list2 = read_sentences_from_file('duibi/adversaries_yelp_lstm_li.txt')






def compute_sim(file):
    list1, list2 = read_sentences_from_file(file)

    # 计算每个字符串的长度（以单词数量为单位）
    word_lengths = [len(s.split()) for s in list1]

    # 计算平均长度
    average_length = sum(word_lengths) / len(list1)


    # 创建TF-IDF向量化器
    vectorizer = TfidfVectorizer()

    # 将两个列表的字符串合并，以便计算向量
    combined_list = list1 + list2
    tfidf_matrix = vectorizer.fit_transform(combined_list)

    # 分别提取两个列表的TF-IDF向量
    tfidf_list1 = tfidf_matrix[:len(list1)]
    tfidf_list2 = tfidf_matrix[len(list1):]

    # 计算每个索引下字符串的余弦相似度
    cos_similarities = []
    for i in range(len(list1)):
        sim = cosine_similarity(tfidf_list1[i], tfidf_list2[i])[0][0]
        cos_similarities.append(sim)

    # 计算相似度的平均值
    average_similarity = np.mean(cos_similarities)
    return average_similarity,average_length


    # print(f"平均余弦相似度: {average_similarity}")
    # print(f"平均单词数量: {average_length}")






