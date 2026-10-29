import os
import re
import math
import matplotlib.pyplot as plt
from collections import Counter, OrderedDict

# =============================
# Utils
# =============================
def _nanmean(xs):
    vals = []
    for x in xs:
        if isinstance(x, (int, float)) and not math.isnan(x):
            vals.append(float(x))
    return float(sum(vals) / len(vals)) if vals else float("nan")

# =============================
# Read data (严格按你的格式)
# =============================
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

# =============================
# 1) Perplexity (Offline n-gram LM)
# =============================
_TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|[0-9]+|[^\sA-Za-z0-9]")

def simple_tokenize(text):
    text = text.strip()
    if not text:
        return []
    return _TOKEN_RE.findall(text.lower())

class NgramLM(object):
    def __init__(self, n=3, k=0.1):
        self.n = n
        self.k = k
        self.ngram_counts = Counter()
        self.context_counts = Counter()
        self.vocab = set()
        self.vocab_size = 0

    def fit(self, sentences):
        n = self.n
        for s in sentences:
            toks = simple_tokenize(s)
            if not toks:
                continue
            self.vocab.update(toks)
            padded = ["<s>"] * (n - 1) + toks + ["</s>"]
            for i in range(n - 1, len(padded)):
                context = tuple(padded[i - (n - 1): i])
                word = padded[i]
                self.ngram_counts[context + (word,)] += 1
                self.context_counts[context] += 1
        self.vocab.update(["<s>", "</s>"])
        self.vocab_size = len(self.vocab)

    def logprob(self, sentence):
        n = self.n
        k = self.k
        V = self.vocab_size
        toks = simple_tokenize(sentence)
        if not toks:
            return 0.0, 0
        padded = ["<s>"] * (n - 1) + toks + ["</s>"]
        logp = 0.0
        count_tokens = 0
        for i in range(n - 1, len(padded)):
            context = tuple(padded[i - (n - 1): i])
            word = padded[i]
            c_ng = self.ngram_counts.get(context + (word,), 0)
            c_ctx = self.context_counts.get(context, 0)
            p = (c_ng + k) / (c_ctx + k * V)
            logp += math.log(p)
            count_tokens += 1
        return logp, count_tokens

def sentence_perplexities_ngram(sentences, lm):
    total_logp = 0.0
    total_tokens = 0
    for s in sentences:
        if not s.strip():
            continue
        logp, n_tok = lm.logprob(s)
        if n_tok == 0:
            continue
        total_logp += logp
        total_tokens += n_tok
    corpus_ppl = math.exp(-total_logp / max(total_tokens, 1))
    return corpus_ppl

# =============================
# 2) Grammar error rate (LanguageTool) - init once
# =============================
def grammar_error_rate_mean(sentences, tool):
    if tool is None:
        return float("nan")
    rates = []
    for sent in sentences:
        if not sent.strip():
            continue
        try:
            matches = tool.check(sent)
            count = len(matches)
        except Exception:
            count = float("nan")
        word_count = max(len(sent.split()), 1)
        rate = (count / word_count) if (count == count) else float("nan")
        rates.append(rate)
    return _nanmean(rates)

# =============================
# 3) Syntax anomaly (spaCy) - load once
# =============================
def syntax_anomaly_mean(sentences, nlp):
    if nlp is None:
        return float("nan")
    scores = []
    for sent in sentences:
        if not sent.strip():
            continue
        doc = nlp(sent)
        tokens = [t for t in doc if not t.is_space]
        n = max(len(tokens), 1)

        root_count = sum(1 for t in tokens if t.dep_ == "ROOT")
        unknown_dep = sum(1 for t in tokens if t.dep_ == "dep")
        unknown_dep_ratio = unknown_dep / n

        dep_dists = [abs(t.i - t.head.i) for t in tokens if t.head is not None]
        avg_dep_distance = sum(dep_dists) / max(len(dep_dists), 1)

        anomaly_score = abs(root_count - 1) + unknown_dep_ratio + max(0.0, (avg_dep_distance - 4.0) / 10.0)
        scores.append(anomaly_score)

    return _nanmean(scores)

# =============================
# Evaluate one file -> 3 metrics (adv only)
# =============================
def evaluate_file_summary(file_path, tool=None, nlp=None, ngram_n=3, ngram_k=0.1, lm_train_on="orig"):
    orig, adv = read_sentences_from_file(file_path)

    lm = NgramLM(n=ngram_n, k=ngram_k)
    if lm_train_on == "orig":
        lm.fit(orig)
    elif lm_train_on == "both":
        lm.fit(orig + adv)
    else:
        raise ValueError("lm_train_on must be 'orig' or 'both'")

    adv_ppl = sentence_perplexities_ngram(adv, lm)
    adv_ger = grammar_error_rate_mean(adv, tool)
    adv_syn = syntax_anomaly_mean(adv, nlp)

    return adv_ppl, adv_ger, adv_syn

# =============================
# Main
# =============================
if __name__ == "__main__":
    BASE = "/root/autodl-fs/adv_results"

    # —— 严格使用你给出的文件名（逐条列出）——
    FILES = {
        "SNLI": {
            "InferSent": {
                "PWWS":        f"{BASE}/adversaries_snli_infersent_pww_model.txt",
                "DeepWordBug": f"{BASE}/adversaries_snli_infersent_dwb_model.txt",
                "TextFooler":  f"{BASE}/adversaries_snli_infersent_tf_model.txt",
                "HardLabel":   f"{BASE}/adversaries_snli_infersent_hl_model.txt",
                "Limeattack":  f"{BASE}/adversaries_snli_infersent_li_model.txt",
                "TextRSS":     f"{BASE}/adversaries_snli_infersent_our_model.txt",
            },
            "ESIM": {
                "PWWS":        f"{BASE}/adversaries_snli_esmi_pww_model.txt",
                "DeepWordBug": f"{BASE}/adversaries_snli_esmi_dwb_model.txt",
                "TextFooler":  f"{BASE}/adversaries_snli_esmi_tf_model.txt",
                "HardLabel":   f"{BASE}/adversaries_snli_esmi_hl_model.txt",
                "Limeattack":  f"{BASE}/adversaries_snli_esmi_li_model.txt",
                "TextRSS":     f"{BASE}/adversaries_snli_esmi_our_model.txt",
            },
            "Bert": {
                "PWWS":        f"{BASE}/adversaries_snli_BERT_pww_model.txt",
                "DeepWordBug": f"{BASE}/adversaries_snli_BERT_dwb_model.txt",
                "TextFooler":  f"{BASE}/adversaries_snli_BERT_tf_model.txt",
                "HardLabel":   f"{BASE}/adversaries_snli_BERT_hl_model.txt",
                "Limeattack":  f"{BASE}/adversaries_snli_BERT_li_model.txt",
                "TextRSS":     f"{BASE}/adversaries_snli_BERT_our_model.txt",
            },
        },

        "MNLI+": {
            "InferSent": {
                "PWWS":        f"{BASE}/adversaries_mnli_m_infersent_pww_model.txt",
                "DeepWordBug": f"{BASE}/adversaries_mnli_m_infersent_dwb_model.txt",
                "TextFooler":  f"{BASE}/adversaries_mnli_m_infersent_tf_model.txt",
                "HardLabel":   f"{BASE}/adversaries_mnli_m_infersent_hl_model.txt",
                "Limeattack":  f"{BASE}/adversaries_mnli_m_infersent_li_model.txt",
                "TextRSS":     f"{BASE}/adversaries_mnli_m_infersent_our_model.txt",
            },
            "ESIM": {
                "PWWS":        f"{BASE}/adversaries_mnli_m_esmi_pww_model.txt",
                "DeepWordBug": f"{BASE}/adversaries_mnli_m_esmi_dwb_model.txt",
                "TextFooler":  f"{BASE}/adversaries_mnli_m_esmi_tf_model.txt",
                "HardLabel":   f"{BASE}/adversaries_mnli_m_esmi_hl_model.txt",
                "Limeattack":  f"{BASE}/adversaries_mnli_m_esmi_li_model.txt",
                "TextRSS":     f"{BASE}/adversaries_mnli_m_esmi_our_model.txt",
            },
            "Bert": {
                "PWWS":        f"{BASE}/adversaries_mnli_m_BERT_pww_model.txt",
                "DeepWordBug": f"{BASE}/adversaries_mnli_m_BERT_dwb_model.txt",
                "TextFooler":  f"{BASE}/adversaries_mnli_m_BERT_tf_model.txt",
                "HardLabel":   f"{BASE}/adversaries_mnli_m_BERT_hl_model.txt",
                "Limeattack":  f"{BASE}/adversaries_mnli_m_BERT_li_model.txt",
                "TextRSS":     f"{BASE}/adversaries_mnli_m_BERT_our_model.txt",
            },
        },

        "MNLI-": {
            "InferSent": {
                "PWWS":        f"{BASE}/adversaries_mnli_mis_infersent_pww_model.txt",
                "DeepWordBug": f"{BASE}/adversaries_mnli_mis_infersent_dwb_model.txt",
                "TextFooler":  f"{BASE}/adversaries_mnli_mis_infersent_tf_model.txt",
                "HardLabel":   f"{BASE}/adversaries_mnli_mis_infersent_hl_model.txt",
                "Limeattack":  f"{BASE}/adversaries_mnli_mis_infersent_li_model.txt",
                "TextRSS":     f"{BASE}/adversaries_mnli_mis_infersent_our_model.txt",
            },
            "ESIM": {
                "PWWS":        f"{BASE}/adversaries_mnli_mis_esmi_pww_model.txt",
                "DeepWordBug": f"{BASE}/adversaries_mnli_mis_esmi_dwb_model.txt",
                "TextFooler":  f"{BASE}/adversaries_mnli_mis_esmi_tf_model.txt",
                "HardLabel":   f"{BASE}/adversaries_mnli_mis_esmi_hl_model.txt",
                "Limeattack":  f"{BASE}/adversaries_mnli_mis_esmi_li_model.txt",
                "TextRSS":     f"{BASE}/adversaries_mnli_mis_esmi_our_model.txt",
            },
            "Bert": {
                "PWWS":        f"{BASE}/adversaries_mnli_mis_BERT_pww_model.txt",
                "DeepWordBug": f"{BASE}/adversaries_mnli_mis_BERT_dwb_model.txt",
                "TextFooler":  f"{BASE}/adversaries_mnli_mis_BERT_tf_model.txt",
                "HardLabel":   f"{BASE}/adversaries_mnli_mis_BERT_hl_model.txt",
                "Limeattack":  f"{BASE}/adversaries_mnli_mis_BERT_li_model.txt",
                "TextRSS":     f"{BASE}/adversaries_mnli_mis_BERT_our_model.txt",
            },
        },
    }

    DATASETS = ["SNLI", "MNLI+", "MNLI-"]
    MODELS = ["InferSent", "ESIM", "Bert"]
    ATTACKS = ["PWWS", "DeepWordBug", "TextFooler", "HardLabel", "Limeattack", "TextRSS"]
    
    
    print("============---======================0==================================================")
    # init LanguageTool / spaCy（失败则对应指标为 NaN，不中断）
    tool = None
    try:
        import language_tool_python
        tool = language_tool_python.LanguageTool("en-US")
    except Exception as e:
        print("[WARN] LanguageTool init failed -> Grammar will be NaN. Error:", str(e))
    print("============---======================1==================================================")
    nlp = None
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
    except Exception as e:
        print("[WARN] spaCy load failed -> Syntax anomaly will be NaN. Error:", str(e))
        print("       If needed: python -m spacy download en_core_web_sm")

    # x-axis：每个数据集下依次 WordCNN, WordLSTM, Bert
    xlabels = []
    xtriples = []
    for d in DATASETS:
        for m in MODELS:
            xlabels.append(f"{m}-{d}")
            xtriples.append((d, m))
            
    print("============---======================2==================================================")
    # 结果容器
    metric_ppl = {atk: [] for atk in ATTACKS}
    metric_ger = {atk: [] for atk in ATTACKS}
    metric_syn = {atk: [] for atk in ATTACKS}

    for atk in ATTACKS:
        for d, m in xtriples:
            fp = FILES[d][m][atk]  # 严格来自上面字典
            if not os.path.exists(fp):
                print(f"[WARN] Missing file: {fp}")
                metric_ppl[atk].append(float("nan"))
                metric_ger[atk].append(float("nan"))
                metric_syn[atk].append(float("nan"))
                continue

            adv_ppl, adv_ger, adv_syn = evaluate_file_summary(
                fp, tool=tool, nlp=nlp, ngram_n=3, ngram_k=0.1, lm_train_on="orig"
            )
            metric_ppl[atk].append(adv_ppl)
            metric_ger[atk].append(adv_ger)
            metric_syn[atk].append(adv_syn)

    # def plot_one(metric_dict, ylabel, title, save_path):
    #     plt.figure(figsize=(14, 4))
    #     xs = list(range(len(xlabels)))
    #     for atk in ATTACKS:
    #         plt.plot(xs, metric_dict[atk], marker="o", linewidth=2, label=atk)
    #     plt.xticks(xs, xlabels, rotation=35, ha="right")
    #     plt.ylabel(ylabel)
    #     plt.title(title)
    #     plt.grid(True, linestyle="--", alpha=0.4)
    #     plt.legend()
    #     plt.tight_layout()
    #     os.makedirs(os.path.dirname(save_path), exist_ok=True)
    #     plt.savefig(save_path, dpi=200)
    #     plt.show()
    
    def plot_one(metric_dict, ylabel, title, save_path):
        plt.figure(figsize=(14, 4))
        xs = list(range(len(xlabels)))

        # 为每种攻防方法指定“线型 + 端点形状”
        style_map = {
            "PWWS":        {"linestyle": "-",  "marker": "o"},  # 实线 + 圆点
            "DeepWordBug": {"linestyle": "--", "marker": "x"},  # 虚线 + 叉
            "TextFooler":  {"linestyle": "-.", "marker": "s"},  # 点划线 + 方块
            "HardLabel":   {"linestyle": ":",  "marker": "D"},  # 点线 + 菱形
            "Limeattack":  {"linestyle": "-",  "marker": "P"},  # 实线 + 五边形（新增）
            "TextRSS":     {"linestyle": "--", "marker": "^"},  # 虚线 + 三角
        }

        for atk in ATTACKS:
            style = style_map.get(atk, {})
            plt.plot(
                xs,
                metric_dict[atk],
                linewidth=2,
                markersize=6,
                label=atk,
                **style
            )

        plt.xticks(xs, xlabels, rotation=35, ha="right")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.legend()
        plt.tight_layout()

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200)
        plt.show()

    
    def plot_one_bar(metric_dict, ylabel, title, save_path):
        plt.figure(figsize=(14, 4))
        xs = list(range(len(xlabels)))

        bar_width = 0.13
        offsets = {
            "PWWS":        -2.5 * bar_width,
            "DeepWordBug": -1.5 * bar_width,
            "TextFooler":  -0.5 * bar_width,
            "HardLabel":    0.5 * bar_width,
            "Limeattack":   1.5 * bar_width,
            "TextRSS":      2.5 * bar_width,
        }

        for atk in ATTACKS:
            plt.bar(
                [x + offsets[atk] for x in xs],
                metric_dict[atk],
                width=bar_width,
                label=atk
            )

        plt.xticks(xs, xlabels, rotation=35, ha="right")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True, axis="y", linestyle="--", alpha=0.4)
        plt.legend(ncol=3)  # legend 更紧凑
        plt.tight_layout()

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200)
        plt.show()

        

    out_dir = "./quality_plots"

    plot_one(
        metric_ppl,
        "Perplexity (adv, corpus token-weighted)",
        "Experiment Ⅱ. Perplexity Comparison",
        f"{out_dir}/perplexity_nli.pdf"
    )

    plot_one(
        metric_ger,
        "Grammar error rate (adv, errors/word)",
        "Experiment Ⅱ. Grammar Error Rate Comparison",
        f"{out_dir}/grammar_error_rate_nli.pdf"
    )

    plot_one_bar(
        metric_syn,
        "Syntax anomaly score (adv, avg)",
        "Experiment Ⅱ. Syntax Anomaly Score Comparison",
        f"{out_dir}/syntax_anomaly_nli.pdf"
    )


    print(f"\nSaved figures to: {os.path.abspath(out_dir)}")
