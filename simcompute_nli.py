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
    original_hypothesis = []
    advanced_hypothesis = []

    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        for i in range(0, len(lines), 4):  # 每四行一组
            # 提取 orig hypothesis 和 adv hypothesis 的内容
            if "orig hypothesis" in lines[i + 1]:
                orig_hypo = lines[i + 1].split(':', 1)[1].strip()
            else:
                orig_hypo = " "

            if "adv hypothesis" in lines[i + 2]:
                adv_hypo = lines[i + 2].split(':', 1)[1].strip()
            else:
                adv_hypo = " "

            original_hypothesis.append(orig_hypo)
            advanced_hypothesis.append(adv_hypo)

    return original_hypothesis, advanced_hypothesis


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
