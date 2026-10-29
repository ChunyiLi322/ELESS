import os

# ============================================================
# ELESSAttack NLI experiments - paper-oriented runs
# Style: common parameters first, then several commands sharing them.
# Uncomment the command/os.system pair you want to run.
# ============================================================

PYTHON = 'python ./attack_nli.py '

# Shared resources. ESIM uses a word-dictionary pickle instead of GloVe
# for --word_embeddings_path, so it has a separate resource string.
RESOURCE = \
    '--word_embeddings_path ./embedding/glove.6B.200d.txt ' \
    '--counter_fitting_embeddings_path ./embedding/counter-fitted-vectors.txt ' \
    '--counter_fitting_cos_sim_path ./cos_sim_counter_fitting.npy ' \
    '--USE_cache_path ./tf_cache '

RESOURCE_ESIM_SNLI = \
    '--word_embeddings_path ./ESIM/data/SNLI/worddict.pkl ' \
    '--counter_fitting_embeddings_path ./embedding/counter-fitted-vectors.txt ' \
    '--counter_fitting_cos_sim_path ./cos_sim_counter_fitting.npy ' \
    '--USE_cache_path ./tf_cache '

RESOURCE_ESIM_MNLI = RESOURCE_ESIM_SNLI.replace('./ESIM/data/SNLI/worddict.pkl', './ESIM/data/MNLI/worddict.pkl')

# Main-comparison common settings. Attack-specific defaults remain inside attack_nli.py.
MAIN_COMMON = '--data_size 500 --random_seed 1234 '

# ============================================================
# 1. Main NLI comparison
# SNLI / MNLI-matched / MNLI-mismatched x InferSent / ESIM / BERT
# Baselines: pww, deepWordBug, TextFooler, hardlabel, limeattack, plus ELESS.
# ============================================================

# ==================== SNLI ====================

# ---------------- SNLI / INFERSENT ----------------
# command = PYTHON + '--dataset_path data/snli --target_model infersent --target_model_name pww --result_model_path_file adversaries_snli_infersent_pww.txt --target_model_path ./InferSent/snli_model_epoch_150.pt ' + MAIN_COMMON + RESOURCE
# os.system(command)
# command = PYTHON + '--dataset_path data/snli --target_model infersent --target_model_name deepWordBug --result_model_path_file adversaries_snli_infersent_deepWordBug.txt --target_model_path ./InferSent/snli_model_epoch_150.pt ' + MAIN_COMMON + RESOURCE
# os.system(command)
# command = PYTHON + '--dataset_path data/snli --target_model infersent --target_model_name TextFooler --result_model_path_file adversaries_snli_infersent_TextFooler.txt --target_model_path ./InferSent/snli_model_epoch_150.pt ' + MAIN_COMMON + RESOURCE
# os.system(command)
# command = PYTHON + '--dataset_path data/snli --target_model infersent --target_model_name hardlabel --result_model_path_file adversaries_snli_infersent_hardlabel.txt --target_model_path ./InferSent/snli_model_epoch_150.pt ' + MAIN_COMMON + RESOURCE
# os.system(command)
# command = PYTHON + '--dataset_path data/snli --target_model infersent --target_model_name limeattack --result_model_path_file adversaries_snli_infersent_limeattack.txt --target_model_path ./InferSent/snli_model_epoch_150.pt ' + MAIN_COMMON + RESOURCE
# os.system(command)
# command = PYTHON + '--dataset_path data/snli --target_model infersent --target_model_name ELESS --result_model_path_file adversaries_snli_infersent_ELESS.txt --target_model_path ./InferSent/snli_model_epoch_150.pt ' + MAIN_COMMON + RESOURCE
# os.system(command)

# ---------------- SNLI / ESIM ----------------
# command = PYTHON + '--dataset_path data/snli --target_model esim --target_model_name pww --result_model_path_file adversaries_snli_esim_pww.txt --target_model_path ./ESIM/results/SNLI/best.pth.tar ' + MAIN_COMMON + RESOURCE_ESIM_SNLI
# os.system(command)
# command = PYTHON + '--dataset_path data/snli --target_model esim --target_model_name deepWordBug --result_model_path_file adversaries_snli_esim_deepWordBug.txt --target_model_path ./ESIM/results/SNLI/best.pth.tar ' + MAIN_COMMON + RESOURCE_ESIM_SNLI
# os.system(command)
# command = PYTHON + '--dataset_path data/snli --target_model esim --target_model_name TextFooler --result_model_path_file adversaries_snli_esim_TextFooler.txt --target_model_path ./ESIM/results/SNLI/best.pth.tar ' + MAIN_COMMON + RESOURCE_ESIM_SNLI
# os.system(command)
# command = PYTHON + '--dataset_path data/snli --target_model esim --target_model_name hardlabel --result_model_path_file adversaries_snli_esim_hardlabel.txt --target_model_path ./ESIM/results/SNLI/best.pth.tar ' + MAIN_COMMON + RESOURCE_ESIM_SNLI
# os.system(command)
# command = PYTHON + '--dataset_path data/snli --target_model esim --target_model_name limeattack --result_model_path_file adversaries_snli_esim_limeattack.txt --target_model_path ./ESIM/results/SNLI/best.pth.tar ' + MAIN_COMMON + RESOURCE_ESIM_SNLI
# os.system(command)
# command = PYTHON + '--dataset_path data/snli --target_model esim --target_model_name ELESS --result_model_path_file adversaries_snli_esim_ELESS.txt --target_model_path ./ESIM/results/SNLI/best.pth.tar ' + MAIN_COMMON + RESOURCE_ESIM_SNLI
# os.system(command)

# ---------------- SNLI / BERT ----------------
# command = PYTHON + '--dataset_path data/snli --target_model bert --target_model_name pww --result_model_path_file adversaries_snli_bert_pww.txt --target_model_path ./BERT/results/snli ' + MAIN_COMMON + RESOURCE
# os.system(command)
# command = PYTHON + '--dataset_path data/snli --target_model bert --target_model_name deepWordBug --result_model_path_file adversaries_snli_bert_deepWordBug.txt --target_model_path ./BERT/results/snli ' + MAIN_COMMON + RESOURCE
# os.system(command)
# command = PYTHON + '--dataset_path data/snli --target_model bert --target_model_name TextFooler --result_model_path_file adversaries_snli_bert_TextFooler.txt --target_model_path ./BERT/results/snli ' + MAIN_COMMON + RESOURCE
# os.system(command)
# command = PYTHON + '--dataset_path data/snli --target_model bert --target_model_name hardlabel --result_model_path_file adversaries_snli_bert_hardlabel.txt --target_model_path ./BERT/results/snli ' + MAIN_COMMON + RESOURCE
# os.system(command)
# command = PYTHON + '--dataset_path data/snli --target_model bert --target_model_name limeattack --result_model_path_file adversaries_snli_bert_limeattack.txt --target_model_path ./BERT/results/snli ' + MAIN_COMMON + RESOURCE
# os.system(command)
# command = PYTHON + '--dataset_path data/snli --target_model bert --target_model_name ELESS --result_model_path_file adversaries_snli_bert_ELESS.txt --target_model_path ./BERT/results/snli ' + MAIN_COMMON + RESOURCE
# os.system(command)

# ==================== MNLI matched ====================

# ---------------- MNLI matched / INFERSENT ----------------
# command = PYTHON + '--dataset_path data/mnli_matched --target_model infersent --target_model_name pww --result_model_path_file adversaries_mnli_m_infersent_pww.txt --target_model_path ./InferSent/mnli_m_model_epoch_500.pt ' + MAIN_COMMON + RESOURCE
# os.system(command)
# command = PYTHON + '--dataset_path data/mnli_matched --target_model infersent --target_model_name deepWordBug --result_model_path_file adversaries_mnli_m_infersent_deepWordBug.txt --target_model_path ./InferSent/mnli_m_model_epoch_500.pt ' + MAIN_COMMON + RESOURCE
# os.system(command)
# command = PYTHON + '--dataset_path data/mnli_matched --target_model infersent --target_model_name TextFooler --result_model_path_file adversaries_mnli_m_infersent_TextFooler.txt --target_model_path ./InferSent/mnli_m_model_epoch_500.pt ' + MAIN_COMMON + RESOURCE
# os.system(command)
# command = PYTHON + '--dataset_path data/mnli_matched --target_model infersent --target_model_name hardlabel --result_model_path_file adversaries_mnli_m_infersent_hardlabel.txt --target_model_path ./InferSent/mnli_m_model_epoch_500.pt ' + MAIN_COMMON + RESOURCE
# os.system(command)
# command = PYTHON + '--dataset_path data/mnli_matched --target_model infersent --target_model_name limeattack --result_model_path_file adversaries_mnli_m_infersent_limeattack.txt --target_model_path ./InferSent/mnli_m_model_epoch_500.pt ' + MAIN_COMMON + RESOURCE
# os.system(command)
# command = PYTHON + '--dataset_path data/mnli_matched --target_model infersent --target_model_name ELESS --result_model_path_file adversaries_mnli_m_infersent_ELESS.txt --target_model_path ./InferSent/mnli_m_model_epoch_500.pt ' + MAIN_COMMON + RESOURCE
# os.system(command)

# ---------------- MNLI matched / ESIM ----------------
# command = PYTHON + '--dataset_path data/mnli_matched --target_model esim --target_model_name pww --result_model_path_file adversaries_mnli_m_esim_pww.txt --target_model_path ./ESIM/results/MNLI/best.pth.tar ' + MAIN_COMMON + RESOURCE_ESIM_MNLI
# os.system(command)
# command = PYTHON + '--dataset_path data/mnli_matched --target_model esim --target_model_name deepWordBug --result_model_path_file adversaries_mnli_m_esim_deepWordBug.txt --target_model_path ./ESIM/results/MNLI/best.pth.tar ' + MAIN_COMMON + RESOURCE_ESIM_MNLI
# os.system(command)
# command = PYTHON + '--dataset_path data/mnli_matched --target_model esim --target_model_name TextFooler --result_model_path_file adversaries_mnli_m_esim_TextFooler.txt --target_model_path ./ESIM/results/MNLI/best.pth.tar ' + MAIN_COMMON + RESOURCE_ESIM_MNLI
# os.system(command)
# command = PYTHON + '--dataset_path data/mnli_matched --target_model esim --target_model_name hardlabel --result_model_path_file adversaries_mnli_m_esim_hardlabel.txt --target_model_path ./ESIM/results/MNLI/best.pth.tar ' + MAIN_COMMON + RESOURCE_ESIM_MNLI
# os.system(command)
# command = PYTHON + '--dataset_path data/mnli_matched --target_model esim --target_model_name limeattack --result_model_path_file adversaries_mnli_m_esim_limeattack.txt --target_model_path ./ESIM/results/MNLI/best.pth.tar ' + MAIN_COMMON + RESOURCE_ESIM_MNLI
# os.system(command)
# command = PYTHON + '--dataset_path data/mnli_matched --target_model esim --target_model_name ELESS --result_model_path_file adversaries_mnli_m_esim_ELESS.txt --target_model_path ./ESIM/results/MNLI/best.pth.tar ' + MAIN_COMMON + RESOURCE_ESIM_MNLI
# os.system(command)

# ---------------- MNLI matched / BERT ----------------
# command = PYTHON + '--dataset_path data/mnli_matched --target_model bert --target_model_name pww --result_model_path_file adversaries_mnli_m_bert_pww.txt --target_model_path ./BERT/results/mnli ' + MAIN_COMMON + RESOURCE
# os.system(command)
# command = PYTHON + '--dataset_path data/mnli_matched --target_model bert --target_model_name deepWordBug --result_model_path_file adversaries_mnli_m_bert_deepWordBug.txt --target_model_path ./BERT/results/mnli ' + MAIN_COMMON + RESOURCE
# os.system(command)
# command = PYTHON + '--dataset_path data/mnli_matched --target_model bert --target_model_name TextFooler --result_model_path_file adversaries_mnli_m_bert_TextFooler.txt --target_model_path ./BERT/results/mnli ' + MAIN_COMMON + RESOURCE
# os.system(command)
# command = PYTHON + '--dataset_path data/mnli_matched --target_model bert --target_model_name hardlabel --result_model_path_file adversaries_mnli_m_bert_hardlabel.txt --target_model_path ./BERT/results/mnli ' + MAIN_COMMON + RESOURCE
# os.system(command)
# command = PYTHON + '--dataset_path data/mnli_matched --target_model bert --target_model_name limeattack --result_model_path_file adversaries_mnli_m_bert_limeattack.txt --target_model_path ./BERT/results/mnli ' + MAIN_COMMON + RESOURCE
# os.system(command)
# command = PYTHON + '--dataset_path data/mnli_matched --target_model bert --target_model_name ELESS --result_model_path_file adversaries_mnli_m_bert_ELESS.txt --target_model_path ./BERT/results/mnli ' + MAIN_COMMON + RESOURCE
# os.system(command)

# ==================== MNLI mismatched ====================

# ---------------- MNLI mismatched / INFERSENT ----------------
# command = PYTHON + '--dataset_path data/mnli_mismatched --target_model infersent --target_model_name pww --result_model_path_file adversaries_mnli_mis_infersent_pww.txt --target_model_path ./InferSent/mnli_mis_model_epoch_500.pt ' + MAIN_COMMON + RESOURCE
# os.system(command)
# command = PYTHON + '--dataset_path data/mnli_mismatched --target_model infersent --target_model_name deepWordBug --result_model_path_file adversaries_mnli_mis_infersent_deepWordBug.txt --target_model_path ./InferSent/mnli_mis_model_epoch_500.pt ' + MAIN_COMMON + RESOURCE
# os.system(command)
# command = PYTHON + '--dataset_path data/mnli_mismatched --target_model infersent --target_model_name TextFooler --result_model_path_file adversaries_mnli_mis_infersent_TextFooler.txt --target_model_path ./InferSent/mnli_mis_model_epoch_500.pt ' + MAIN_COMMON + RESOURCE
# os.system(command)
# command = PYTHON + '--dataset_path data/mnli_mismatched --target_model infersent --target_model_name hardlabel --result_model_path_file adversaries_mnli_mis_infersent_hardlabel.txt --target_model_path ./InferSent/mnli_mis_model_epoch_500.pt ' + MAIN_COMMON + RESOURCE
# os.system(command)
# command = PYTHON + '--dataset_path data/mnli_mismatched --target_model infersent --target_model_name limeattack --result_model_path_file adversaries_mnli_mis_infersent_limeattack.txt --target_model_path ./InferSent/mnli_mis_model_epoch_500.pt ' + MAIN_COMMON + RESOURCE
# os.system(command)
# command = PYTHON + '--dataset_path data/mnli_mismatched --target_model infersent --target_model_name ELESS --result_model_path_file adversaries_mnli_mis_infersent_ELESS.txt --target_model_path ./InferSent/mnli_mis_model_epoch_500.pt ' + MAIN_COMMON + RESOURCE
# os.system(command)

# ---------------- MNLI mismatched / ESIM ----------------
# command = PYTHON + '--dataset_path data/mnli_mismatched --target_model esim --target_model_name pww --result_model_path_file adversaries_mnli_mis_esim_pww.txt --target_model_path ./ESIM/results/MNLI/best.pth.tar ' + MAIN_COMMON + RESOURCE_ESIM_MNLI
# os.system(command)
# command = PYTHON + '--dataset_path data/mnli_mismatched --target_model esim --target_model_name deepWordBug --result_model_path_file adversaries_mnli_mis_esim_deepWordBug.txt --target_model_path ./ESIM/results/MNLI/best.pth.tar ' + MAIN_COMMON + RESOURCE_ESIM_MNLI
# os.system(command)
# command = PYTHON + '--dataset_path data/mnli_mismatched --target_model esim --target_model_name TextFooler --result_model_path_file adversaries_mnli_mis_esim_TextFooler.txt --target_model_path ./ESIM/results/MNLI/best.pth.tar ' + MAIN_COMMON + RESOURCE_ESIM_MNLI
# os.system(command)
# command = PYTHON + '--dataset_path data/mnli_mismatched --target_model esim --target_model_name hardlabel --result_model_path_file adversaries_mnli_mis_esim_hardlabel.txt --target_model_path ./ESIM/results/MNLI/best.pth.tar ' + MAIN_COMMON + RESOURCE_ESIM_MNLI
# os.system(command)
# command = PYTHON + '--dataset_path data/mnli_mismatched --target_model esim --target_model_name limeattack --result_model_path_file adversaries_mnli_mis_esim_limeattack.txt --target_model_path ./ESIM/results/MNLI/best.pth.tar ' + MAIN_COMMON + RESOURCE_ESIM_MNLI
# os.system(command)
# command = PYTHON + '--dataset_path data/mnli_mismatched --target_model esim --target_model_name ELESS --result_model_path_file adversaries_mnli_mis_esim_ELESS.txt --target_model_path ./ESIM/results/MNLI/best.pth.tar ' + MAIN_COMMON + RESOURCE_ESIM_MNLI
# os.system(command)

# ---------------- MNLI mismatched / BERT ----------------
# command = PYTHON + '--dataset_path data/mnli_mismatched --target_model bert --target_model_name pww --result_model_path_file adversaries_mnli_mis_bert_pww.txt --target_model_path ./BERT/results/mnli ' + MAIN_COMMON + RESOURCE
# os.system(command)
command = PYTHON + '--dataset_path data/mnli_mismatched --target_model bert --target_model_name deepWordBug --result_model_path_file adversaries_mnli_mis_bert_deepWordBug.txt --target_model_path ./BERT/results/mnli ' + MAIN_COMMON + RESOURCE
os.system(command)
# command = PYTHON + '--dataset_path data/mnli_mismatched --target_model bert --target_model_name TextFooler --result_model_path_file adversaries_mnli_mis_bert_TextFooler.txt --target_model_path ./BERT/results/mnli ' + MAIN_COMMON + RESOURCE
# os.system(command)
# command = PYTHON + '--dataset_path data/mnli_mismatched --target_model bert --target_model_name hardlabel --result_model_path_file adversaries_mnli_mis_bert_hardlabel.txt --target_model_path ./BERT/results/mnli ' + MAIN_COMMON + RESOURCE
# os.system(command)
# command = PYTHON + '--dataset_path data/mnli_mismatched --target_model bert --target_model_name limeattack --result_model_path_file adversaries_mnli_mis_bert_limeattack.txt --target_model_path ./BERT/results/mnli ' + MAIN_COMMON + RESOURCE
# os.system(command)
# command = PYTHON + '--dataset_path data/mnli_mismatched --target_model bert --target_model_name ELESS --result_model_path_file adversaries_mnli_mis_bert_ELESS.txt --target_model_path ./BERT/results/mnli ' + MAIN_COMMON + RESOURCE
# os.system(command)

# ============================================================
# 2. Primary ablation study
# Same dataset/model/K/rho/encoder/seed; only switch one component.
# NLI paper-oriented setting: SNLI+BERT and MNLI-matched+BERT.
# ============================================================

ABLATION_COMMON = \
    '--target_model bert --target_model_name ELESS ' \
    '--data_size 500 --synonym_num 50 --sim_score_window 15 ' \
    '--sim_score_threshold 0.5 --perturb_budget 0.30 ' \
    '--initial_temperature 0.10 --cooling_factor 0.95 ' \
    '--energy_zeta 0 --max_query_budget 0 --random_seed 1234 ' + RESOURCE

# snli + BERT
# command = PYTHON + '--dataset_path data/snli --target_model_path ./BERT/results/snli --result_model_path_file ablation_snli_bert_full.txt --use_ranking 1 --use_local_semantics 1 --use_energy_search 1 ' + ABLATION_COMMON
# os.system(command)
# command = PYTHON + '--dataset_path data/snli --target_model_path ./BERT/results/snli --result_model_path_file ablation_snli_bert_noRank.txt --use_ranking 0 --use_local_semantics 1 --use_energy_search 1 ' + ABLATION_COMMON
# os.system(command)
# command = PYTHON + '--dataset_path data/snli --target_model_path ./BERT/results/snli --result_model_path_file ablation_snli_bert_noLocal.txt --use_ranking 1 --use_local_semantics 0 --use_energy_search 1 ' + ABLATION_COMMON
# os.system(command)
# command = PYTHON + '--dataset_path data/snli --target_model_path ./BERT/results/snli --result_model_path_file ablation_snli_bert_noEnergy.txt --use_ranking 1 --use_local_semantics 1 --use_energy_search 0 ' + ABLATION_COMMON
# os.system(command)

# mnli_matched + BERT
# command = PYTHON + '--dataset_path data/mnli_matched --target_model_path ./BERT/results/mnli --result_model_path_file ablation_mnli_m_bert_full.txt --use_ranking 1 --use_local_semantics 1 --use_energy_search 1 ' + ABLATION_COMMON
# os.system(command)
# command = PYTHON + '--dataset_path data/mnli_matched --target_model_path ./BERT/results/mnli --result_model_path_file ablation_mnli_m_bert_noRank.txt --use_ranking 0 --use_local_semantics 1 --use_energy_search 1 ' + ABLATION_COMMON
# os.system(command)
# command = PYTHON + '--dataset_path data/mnli_matched --target_model_path ./BERT/results/mnli --result_model_path_file ablation_mnli_m_bert_noLocal.txt --use_ranking 1 --use_local_semantics 0 --use_energy_search 1 ' + ABLATION_COMMON
# os.system(command)
# command = PYTHON + '--dataset_path data/mnli_matched --target_model_path ./BERT/results/mnli --result_model_path_file ablation_mnli_m_bert_noEnergy.txt --use_ranking 1 --use_local_semantics 1 --use_energy_search 0 ' + ABLATION_COMMON
# os.system(command)

# ============================================================
# 3. Factorial ablation
# SNLI + BERT, all 2^3 combinations of ranking/local/energy.
# ============================================================

FACTORIAL_COMMON = \
    '--dataset_path data/snli --target_model bert --target_model_name ELESS ' \
    '--target_model_path ./BERT/results/snli --data_size 500 --synonym_num 50 ' \
    '--sim_score_window 15 --sim_score_threshold 0.5 --perturb_budget 0.30 ' \
    '--initial_temperature 0.10 --cooling_factor 0.95 ' \
    '--energy_zeta 0 --max_query_budget 0 --random_seed 1234 ' + RESOURCE

# command = PYTHON + '--result_model_path_file factorial_nli_000.txt --use_ranking 0 --use_local_semantics 0 --use_energy_search 0 ' + FACTORIAL_COMMON
# os.system(command)
# command = PYTHON + '--result_model_path_file factorial_nli_100.txt --use_ranking 1 --use_local_semantics 0 --use_energy_search 0 ' + FACTORIAL_COMMON
# os.system(command)
# command = PYTHON + '--result_model_path_file factorial_nli_010.txt --use_ranking 0 --use_local_semantics 1 --use_energy_search 0 ' + FACTORIAL_COMMON
# os.system(command)
# command = PYTHON + '--result_model_path_file factorial_nli_001.txt --use_ranking 0 --use_local_semantics 0 --use_energy_search 1 ' + FACTORIAL_COMMON
# os.system(command)
# command = PYTHON + '--result_model_path_file factorial_nli_110.txt --use_ranking 1 --use_local_semantics 1 --use_energy_search 0 ' + FACTORIAL_COMMON
# os.system(command)
# command = PYTHON + '--result_model_path_file factorial_nli_101.txt --use_ranking 1 --use_local_semantics 0 --use_energy_search 1 ' + FACTORIAL_COMMON
# os.system(command)
# command = PYTHON + '--result_model_path_file factorial_nli_011.txt --use_ranking 0 --use_local_semantics 1 --use_energy_search 1 ' + FACTORIAL_COMMON
# os.system(command)
# command = PYTHON + '--result_model_path_file factorial_nli_111.txt --use_ranking 1 --use_local_semantics 1 --use_energy_search 1 ' + FACTORIAL_COMMON
# os.system(command)

# ============================================================
# 4. Search behavior / local optimum analysis
# Same samples + same seed. Compare greedy vs ELESS stochastic search.
# ============================================================

TRAJECTORY_COMMON = \
    '--dataset_path data/snli --target_model bert --target_model_name ELESS ' \
    '--target_model_path ./BERT/results/snli --data_size 500 --synonym_num 50 ' \
    '--sim_score_window 15 --sim_score_threshold 0.5 --perturb_budget 0.30 ' \
    '--initial_temperature 0.10 --cooling_factor 0.95 --random_seed 1234 ' \
    '--use_ranking 1 --use_local_semantics 1 ' + RESOURCE
# command = PYTHON + '--result_model_path_file trajectory_nli_greedy.txt --use_energy_search 0 ' + TRAJECTORY_COMMON
# os.system(command)
# command = PYTHON + '--result_model_path_file trajectory_nli_eless.txt --use_energy_search 1 ' + TRAJECTORY_COMMON
# os.system(command)

# ============================================================
# 5. zeta sensitivity
# SNLI+BERT and MNLI-matched+BERT; vary only the named factor.
# ============================================================

ZETA_COMMON_SNLI = \
    '--dataset_path data/snli --target_model_path ./BERT/results/snli ' \
    '--target_model bert --target_model_name ELESS --data_size 500 --sim_score_window 15 --sim_score_threshold 0.5 --random_seed 1234 --use_ranking 1 --use_local_semantics 1 --use_energy_search 1 --synonym_num 50 --perturb_budget 0.30 --initial_temperature 0.10 --cooling_factor 0.95 --max_query_budget 0 ' + RESOURCE
ZETA_COMMON_MNLI_M = ZETA_COMMON_SNLI.replace('data/snli', 'data/mnli_matched').replace('./BERT/results/snli', './BERT/results/mnli')

# snli
# command = PYTHON + '--energy_zeta 1 --result_model_path_file zeta_snli_1.txt ' + ZETA_COMMON_SNLI
# os.system(command)
# command = PYTHON + '--energy_zeta 10 --result_model_path_file zeta_snli_10.txt ' + ZETA_COMMON_SNLI
# os.system(command)
# command = PYTHON + '--energy_zeta 100 --result_model_path_file zeta_snli_100.txt ' + ZETA_COMMON_SNLI
# os.system(command)
# command = PYTHON + '--energy_zeta 1000 --result_model_path_file zeta_snli_1000.txt ' + ZETA_COMMON_SNLI
# os.system(command)

# mnli_m
# command = PYTHON + '--energy_zeta 1 --result_model_path_file zeta_mnli_m_1.txt ' + ZETA_COMMON_MNLI_M
# os.system(command)
# command = PYTHON + '--energy_zeta 10 --result_model_path_file zeta_mnli_m_10.txt ' + ZETA_COMMON_MNLI_M
# os.system(command)
# command = PYTHON + '--energy_zeta 100 --result_model_path_file zeta_mnli_m_100.txt ' + ZETA_COMMON_MNLI_M
# os.system(command)
# command = PYTHON + '--energy_zeta 1000 --result_model_path_file zeta_mnli_m_1000.txt ' + ZETA_COMMON_MNLI_M
# os.system(command)


# ============================================================
# 6. Synonym candidate size K sensitivity
# SNLI+BERT and MNLI-matched+BERT; vary only the named factor.
# ============================================================

K_COMMON_SNLI = \
    '--dataset_path data/snli --target_model_path ./BERT/results/snli ' \
    '--target_model bert --target_model_name ELESS --data_size 500 --sim_score_window 15 --sim_score_threshold 0.5 --random_seed 1234 --use_ranking 1 --use_local_semantics 1 --use_energy_search 1 --perturb_budget 0.30 --initial_temperature 0.10 --cooling_factor 0.95 --energy_zeta 0 --max_query_budget 0 ' + RESOURCE
K_COMMON_MNLI_M = K_COMMON_SNLI.replace('data/snli', 'data/mnli_matched').replace('./BERT/results/snli', './BERT/results/mnli')

# snli
# command = PYTHON + '--synonym_num 10 --result_model_path_file k_snli_10.txt ' + K_COMMON_SNLI
# os.system(command)
# command = PYTHON + '--synonym_num 20 --result_model_path_file k_snli_20.txt ' + K_COMMON_SNLI
# os.system(command)
# command = PYTHON + '--synonym_num 50 --result_model_path_file k_snli_50.txt ' + K_COMMON_SNLI
# os.system(command)
# command = PYTHON + '--synonym_num 100 --result_model_path_file k_snli_100.txt ' + K_COMMON_SNLI
# os.system(command)
# command = PYTHON + '--synonym_num 200 --result_model_path_file k_snli_200.txt ' + K_COMMON_SNLI
# os.system(command)

# mnli_m
# command = PYTHON + '--synonym_num 10 --result_model_path_file k_mnli_m_10.txt ' + K_COMMON_MNLI_M
# os.system(command)
# command = PYTHON + '--synonym_num 20 --result_model_path_file k_mnli_m_20.txt ' + K_COMMON_MNLI_M
# os.system(command)
# command = PYTHON + '--synonym_num 50 --result_model_path_file k_mnli_m_50.txt ' + K_COMMON_MNLI_M
# os.system(command)
# command = PYTHON + '--synonym_num 100 --result_model_path_file k_mnli_m_100.txt ' + K_COMMON_MNLI_M
# os.system(command)
# command = PYTHON + '--synonym_num 200 --result_model_path_file k_mnli_m_200.txt ' + K_COMMON_MNLI_M
# os.system(command)


# ============================================================
# 7. Initial temperature T0 sensitivity
# SNLI+BERT and MNLI-matched+BERT; vary only the named factor.
# ============================================================

T0_COMMON_SNLI = \
    '--dataset_path data/snli --target_model_path ./BERT/results/snli ' \
    '--target_model bert --target_model_name ELESS --data_size 500 --sim_score_window 15 --sim_score_threshold 0.5 --random_seed 1234 --use_ranking 1 --use_local_semantics 1 --use_energy_search 1 --synonym_num 50 --perturb_budget 0.30 --cooling_factor 0.95 --energy_zeta 0 --max_query_budget 0 ' + RESOURCE
T0_COMMON_MNLI_M = T0_COMMON_SNLI.replace('data/snli', 'data/mnli_matched').replace('./BERT/results/snli', './BERT/results/mnli')

# snli
# command = PYTHON + '--initial_temperature 0.01 --result_model_path_file t0_snli_001.txt ' + T0_COMMON_SNLI
# os.system(command)
# command = PYTHON + '--initial_temperature 0.03 --result_model_path_file t0_snli_003.txt ' + T0_COMMON_SNLI
# os.system(command)
# command = PYTHON + '--initial_temperature 0.05 --result_model_path_file t0_snli_005.txt ' + T0_COMMON_SNLI
# os.system(command)
# command = PYTHON + '--initial_temperature 0.10 --result_model_path_file t0_snli_010.txt ' + T0_COMMON_SNLI
# os.system(command)
# command = PYTHON + '--initial_temperature 0.20 --result_model_path_file t0_snli_020.txt ' + T0_COMMON_SNLI
# os.system(command)

# mnli_m
# command = PYTHON + '--initial_temperature 0.01 --result_model_path_file t0_mnli_m_001.txt ' + T0_COMMON_MNLI_M
# os.system(command)
# command = PYTHON + '--initial_temperature 0.03 --result_model_path_file t0_mnli_m_003.txt ' + T0_COMMON_MNLI_M
# os.system(command)
# command = PYTHON + '--initial_temperature 0.05 --result_model_path_file t0_mnli_m_005.txt ' + T0_COMMON_MNLI_M
# os.system(command)
# command = PYTHON + '--initial_temperature 0.10 --result_model_path_file t0_mnli_m_010.txt ' + T0_COMMON_MNLI_M
# os.system(command)
# command = PYTHON + '--initial_temperature 0.20 --result_model_path_file t0_mnli_m_020.txt ' + T0_COMMON_MNLI_M
# os.system(command)


# ============================================================
# 8. Cooling factor alpha sensitivity
# SNLI+BERT and MNLI-matched+BERT; vary only the named factor.
# ============================================================

ALPHA_COMMON_SNLI = \
    '--dataset_path data/snli --target_model_path ./BERT/results/snli ' \
    '--target_model bert --target_model_name ELESS --data_size 500 --sim_score_window 15 --sim_score_threshold 0.5 --random_seed 1234 --use_ranking 1 --use_local_semantics 1 --use_energy_search 1 --synonym_num 50 --perturb_budget 0.30 --initial_temperature 0.10 --energy_zeta 0 --max_query_budget 0 ' + RESOURCE
ALPHA_COMMON_MNLI_M = ALPHA_COMMON_SNLI.replace('data/snli', 'data/mnli_matched').replace('./BERT/results/snli', './BERT/results/mnli')

# snli
# command = PYTHON + '--cooling_factor 0.80 --result_model_path_file alpha_snli_080.txt ' + ALPHA_COMMON_SNLI
# os.system(command)
# command = PYTHON + '--cooling_factor 0.90 --result_model_path_file alpha_snli_090.txt ' + ALPHA_COMMON_SNLI
# os.system(command)
# command = PYTHON + '--cooling_factor 0.95 --result_model_path_file alpha_snli_095.txt ' + ALPHA_COMMON_SNLI
# os.system(command)
# command = PYTHON + '--cooling_factor 0.99 --result_model_path_file alpha_snli_099.txt ' + ALPHA_COMMON_SNLI
# os.system(command)

# mnli_m
# command = PYTHON + '--cooling_factor 0.80 --result_model_path_file alpha_mnli_m_080.txt ' + ALPHA_COMMON_MNLI_M
# os.system(command)
# command = PYTHON + '--cooling_factor 0.90 --result_model_path_file alpha_mnli_m_090.txt ' + ALPHA_COMMON_MNLI_M
# os.system(command)
# command = PYTHON + '--cooling_factor 0.95 --result_model_path_file alpha_mnli_m_095.txt ' + ALPHA_COMMON_MNLI_M
# os.system(command)
# command = PYTHON + '--cooling_factor 0.99 --result_model_path_file alpha_mnli_m_099.txt ' + ALPHA_COMMON_MNLI_M
# os.system(command)


# ============================================================
# 9. Perturbation budget rho
# SNLI+BERT and MNLI-matched+BERT; vary only the named factor.
# ============================================================

RHO_COMMON_SNLI = \
    '--dataset_path data/snli --target_model_path ./BERT/results/snli ' \
    '--target_model bert --target_model_name ELESS --data_size 500 --sim_score_window 15 --sim_score_threshold 0.5 --random_seed 1234 --use_ranking 1 --use_local_semantics 1 --use_energy_search 1 --synonym_num 50 --initial_temperature 0.10 --cooling_factor 0.95 --energy_zeta 0 --max_query_budget 0 ' + RESOURCE
RHO_COMMON_MNLI_M = RHO_COMMON_SNLI.replace('data/snli', 'data/mnli_matched').replace('./BERT/results/snli', './BERT/results/mnli')

# snli
# command = PYTHON + '--perturb_budget 0.05 --result_model_path_file rho_snli_005.txt ' + RHO_COMMON_SNLI
# os.system(command)
# command = PYTHON + '--perturb_budget 0.10 --result_model_path_file rho_snli_010.txt ' + RHO_COMMON_SNLI
# os.system(command)
# command = PYTHON + '--perturb_budget 0.15 --result_model_path_file rho_snli_015.txt ' + RHO_COMMON_SNLI
# os.system(command)
# command = PYTHON + '--perturb_budget 0.20 --result_model_path_file rho_snli_020.txt ' + RHO_COMMON_SNLI
# os.system(command)
# command = PYTHON + '--perturb_budget 0.30 --result_model_path_file rho_snli_030.txt ' + RHO_COMMON_SNLI
# os.system(command)

# mnli_m
# command = PYTHON + '--perturb_budget 0.05 --result_model_path_file rho_mnli_m_005.txt ' + RHO_COMMON_MNLI_M
# os.system(command)
# command = PYTHON + '--perturb_budget 0.10 --result_model_path_file rho_mnli_m_010.txt ' + RHO_COMMON_MNLI_M
# os.system(command)
# command = PYTHON + '--perturb_budget 0.15 --result_model_path_file rho_mnli_m_015.txt ' + RHO_COMMON_MNLI_M
# os.system(command)
# command = PYTHON + '--perturb_budget 0.20 --result_model_path_file rho_mnli_m_020.txt ' + RHO_COMMON_MNLI_M
# os.system(command)
# command = PYTHON + '--perturb_budget 0.30 --result_model_path_file rho_mnli_m_030.txt ' + RHO_COMMON_MNLI_M
# os.system(command)


# ============================================================
# 10. Query budget
# SNLI+BERT and MNLI-matched+BERT; vary only the named factor.
# ============================================================

QUERY_COMMON_SNLI = \
    '--dataset_path data/snli --target_model_path ./BERT/results/snli ' \
    '--target_model bert --target_model_name ELESS --data_size 500 --sim_score_window 15 --sim_score_threshold 0.5 --random_seed 1234 --use_ranking 1 --use_local_semantics 1 --use_energy_search 1 --synonym_num 50 --perturb_budget 0.30 --initial_temperature 0.10 --cooling_factor 0.95 --energy_zeta 0 ' + RESOURCE
QUERY_COMMON_MNLI_M = QUERY_COMMON_SNLI.replace('data/snli', 'data/mnli_matched').replace('./BERT/results/snli', './BERT/results/mnli')

# snli
# command = PYTHON + '--max_query_budget 50 --result_model_path_file query_snli_50.txt ' + QUERY_COMMON_SNLI
# os.system(command)
# command = PYTHON + '--max_query_budget 100 --result_model_path_file query_snli_100.txt ' + QUERY_COMMON_SNLI
# os.system(command)
# command = PYTHON + '--max_query_budget 200 --result_model_path_file query_snli_200.txt ' + QUERY_COMMON_SNLI
# os.system(command)
# command = PYTHON + '--max_query_budget 500 --result_model_path_file query_snli_500.txt ' + QUERY_COMMON_SNLI
# os.system(command)
# command = PYTHON + '--max_query_budget 1000 --result_model_path_file query_snli_1000.txt ' + QUERY_COMMON_SNLI
# os.system(command)

# mnli_m
# command = PYTHON + '--max_query_budget 50 --result_model_path_file query_mnli_m_50.txt ' + QUERY_COMMON_MNLI_M
# os.system(command)
# command = PYTHON + '--max_query_budget 100 --result_model_path_file query_mnli_m_100.txt ' + QUERY_COMMON_MNLI_M
# os.system(command)
# command = PYTHON + '--max_query_budget 200 --result_model_path_file query_mnli_m_200.txt ' + QUERY_COMMON_MNLI_M
# os.system(command)
# command = PYTHON + '--max_query_budget 500 --result_model_path_file query_mnli_m_500.txt ' + QUERY_COMMON_MNLI_M
# os.system(command)
# command = PYTHON + '--max_query_budget 1000 --result_model_path_file query_mnli_m_1000.txt ' + QUERY_COMMON_MNLI_M
# os.system(command)

# ============================================================
# 11. Repeated-run stability
# Same parameters, only random_seed changes.
# ============================================================

STABILITY_COMMON_SNLI = \
    '--dataset_path data/snli --target_model bert --target_model_name ELESS ' \
    '--target_model_path ./BERT/results/snli --data_size 500 --synonym_num 50 ' \
    '--sim_score_window 15 --sim_score_threshold 0.5 --perturb_budget 0.30 ' \
    '--initial_temperature 0.10 --cooling_factor 0.95 --energy_zeta 0 ' \
    '--max_query_budget 0 --use_ranking 1 --use_local_semantics 1 --use_energy_search 1 ' + RESOURCE
STABILITY_COMMON_MNLI_M = STABILITY_COMMON_SNLI.replace('data/snli', 'data/mnli_matched').replace('./BERT/results/snli', './BERT/results/mnli')

# snli
# command = PYTHON + '--random_seed 1234 --result_model_path_file stability_snli_bert_1234.txt ' + STABILITY_COMMON_SNLI
# os.system(command)
# command = PYTHON + '--random_seed 2024 --result_model_path_file stability_snli_bert_2024.txt ' + STABILITY_COMMON_SNLI
# os.system(command)
# command = PYTHON + '--random_seed 3407 --result_model_path_file stability_snli_bert_3407.txt ' + STABILITY_COMMON_SNLI
# os.system(command)
# command = PYTHON + '--random_seed 5678 --result_model_path_file stability_snli_bert_5678.txt ' + STABILITY_COMMON_SNLI
# os.system(command)
# command = PYTHON + '--random_seed 9999 --result_model_path_file stability_snli_bert_9999.txt ' + STABILITY_COMMON_SNLI
# os.system(command)

# mnli_m
# command = PYTHON + '--random_seed 1234 --result_model_path_file stability_mnli_m_bert_1234.txt ' + STABILITY_COMMON_MNLI_M
# os.system(command)
# command = PYTHON + '--random_seed 2024 --result_model_path_file stability_mnli_m_bert_2024.txt ' + STABILITY_COMMON_MNLI_M
# os.system(command)
# command = PYTHON + '--random_seed 3407 --result_model_path_file stability_mnli_m_bert_3407.txt ' + STABILITY_COMMON_MNLI_M
# os.system(command)
# command = PYTHON + '--random_seed 5678 --result_model_path_file stability_mnli_m_bert_5678.txt ' + STABILITY_COMMON_MNLI_M
# os.system(command)
# command = PYTHON + '--random_seed 9999 --result_model_path_file stability_mnli_m_bert_9999.txt ' + STABILITY_COMMON_MNLI_M
# os.system(command)

# ============================================================
# 12. Experiments that do NOT need a new attack parameter sweep
# ============================================================
# Input-length analysis:
#   Reuse normal ELESS sample-level logs and group by hypothesis length offline.
#
# Automatic quality / efficiency / significance:
#   Reuse outputs from the main comparison and evaluate them offline.
#
# Transferability:
#   Generate ELESS adversarial hypotheses on one NLI target model and evaluate
#   the saved premise-hypothesis pairs on the other target models separately.
#
# Additional pretrained encoders / defenses:
#   These require extra integration code and are not merely parameter changes
#   to attack_nli.py.

# NOTE: All commands are commented out by default. Uncomment exactly the runs
# you want. This avoids accidentally launching the full experiment matrix.
