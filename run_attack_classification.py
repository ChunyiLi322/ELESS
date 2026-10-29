import os

# ============================================================
# ELESSAttack classification experiments - paper-oriented runs
# Style: common parameters first, then several commands sharing them.
# Uncomment the command/os.system pair you want to run.
# ============================================================

PYTHON = 'python ./attack_classification.py '

# Shared resource paths
RESOURCE = \
    '--word_embeddings_path ./embedding/glove.6B.200d.txt ' \
    '--counter_fitting_embeddings_path ./embedding/counter-fitted-vectors.txt ' \
    '--counter_fitting_cos_sim_path ./cos_sim_counter_fitting.npy ' \
    '--USE_cache_path ./tf_cache '

# Shared default ELESS parameters used unless one controlled experiment overrides one of them.
MAIN_COMMON = \
    '--data_size 500 ' \
    '--synonym_num 50 ' \
    '--sim_score_window 15 ' \
    '--sim_score_threshold 0.5 ' \
    '--import_score_threshold -1.0 ' \
    '--perturb_budget 0.30 ' \
    '--initial_temperature 0.10 ' \
    '--cooling_factor 0.95 ' \
    '--energy_zeta 0 ' \
    '--max_query_budget 0 ' \
    '--random_seed 1234 ' \
    '--use_ranking 1 ' \
    '--use_local_semantics 1 ' \
    '--use_energy_search 1 '

# ============================================================
# 1. Main classification comparison
# Paper: MR / IMDb / Yelp / AG x WordCNN / WordLSTM / BERT
# Baselines: pww, deepWordBug, TextFooler, hardlabel, limeattack,
#            texthoaxer, plus ELESS.
# These commands use the same data_size/resources. Attack-specific
# parameters are handled inside attack_classification.py.
# ============================================================

MAIN_COMMON = '--data_size 500 --random_seed 1234 '

# ---------------- MR / WordCNN ----------------
# command = PYTHON + '--dataset_path data/mr --target_model wordCNN --nclasses 2 --perturb_model pww --result_model_path_file adversaries_mr_cnn_pww.txt --target_model_path ./targetmodelpathcnn/mr ' + MAIN_COMMON + RESOURCE
# os.system(command)
# command = PYTHON + '--dataset_path data/mr --target_model wordCNN --nclasses 2 --perturb_model deepWordBug --result_model_path_file adversaries_mr_cnn_deepWordBug.txt --target_model_path ./targetmodelpathcnn/mr ' + MAIN_COMMON + RESOURCE
# os.system(command)
# command = PYTHON + '--dataset_path data/mr --target_model wordCNN --nclasses 2 --perturb_model TextFooler --result_model_path_file adversaries_mr_cnn_TextFooler.txt --target_model_path ./targetmodelpathcnn/mr ' + MAIN_COMMON + RESOURCE
# os.system(command)
# command = PYTHON + '--dataset_path data/mr --target_model wordCNN --nclasses 2 --perturb_model hardlabel --result_model_path_file adversaries_mr_cnn_hardlabel.txt --target_model_path ./targetmodelpathcnn/mr ' + MAIN_COMMON + RESOURCE
# os.system(command)
# command = PYTHON + '--dataset_path data/mr --target_model wordCNN --nclasses 2 --perturb_model limeattack --result_model_path_file adversaries_mr_cnn_limeattack.txt --target_model_path ./targetmodelpathcnn/mr ' + MAIN_COMMON + RESOURCE
# os.system(command)
# command = PYTHON + '--dataset_path data/mr --target_model wordCNN --nclasses 2 --perturb_model ELESS --result_model_path_file adversaries_mr_cnn_ELESS.txt --target_model_path ./targetmodelpathcnn/mr ' + MAIN_COMMON + RESOURCE
# os.system(command)

# ---------------- MR / WordLSTM ----------------
# command = PYTHON + '--dataset_path data/mr --target_model wordLSTM --nclasses 2 --perturb_model pww --result_model_path_file adversaries_mr_lstm_pww.txt --target_model_path ./targetmodelpathlstm/mr ' + MAIN_COMMON + RESOURCE
# os.system(command)
# command = PYTHON + '--dataset_path data/mr --target_model wordLSTM --nclasses 2 --perturb_model deepWordBug --result_model_path_file adversaries_mr_lstm_deepWordBug.txt --target_model_path ./targetmodelpathlstm/mr ' + MAIN_COMMON + RESOURCE
# os.system(command)
# command = PYTHON + '--dataset_path data/mr --target_model wordLSTM --nclasses 2 --perturb_model TextFooler --result_model_path_file adversaries_mr_lstm_TextFooler.txt --target_model_path ./targetmodelpathlstm/mr ' + MAIN_COMMON + RESOURCE
# os.system(command)
# command = PYTHON + '--dataset_path data/mr --target_model wordLSTM --nclasses 2 --perturb_model hardlabel --result_model_path_file adversaries_mr_lstm_hardlabel.txt --target_model_path ./targetmodelpathlstm/mr ' + MAIN_COMMON + RESOURCE
# os.system(command)
# command = PYTHON + '--dataset_path data/mr --target_model wordLSTM --nclasses 2 --perturb_model limeattack --result_model_path_file adversaries_mr_lstm_limeattack.txt --target_model_path ./targetmodelpathlstm/mr ' + MAIN_COMMON + RESOURCE
# os.system(command)
# command = PYTHON + '--dataset_path data/mr --target_model wordLSTM --nclasses 2 --perturb_model ELESS --result_model_path_file adversaries_mr_lstm_ELESS.txt --target_model_path ./targetmodelpathlstm/mr ' + MAIN_COMMON + RESOURCE
# os.system(command)

# ---------------- MR / BERT ----------------
# command = PYTHON + '--dataset_path data/mr --target_model bert --nclasses 2 --perturb_model pww --result_model_path_file adversaries_mr_bert_pww.txt --target_model_path ./BERT/results/mr ' + MAIN_COMMON + RESOURCE
# os.system(command)
# command = PYTHON + '--dataset_path data/mr --target_model bert --nclasses 2 --perturb_model deepWordBug --result_model_path_file adversaries_mr_bert_deepWordBug.txt --target_model_path ./BERT/results/mr ' + MAIN_COMMON + RESOURCE
# os.system(command)
# command = PYTHON + '--dataset_path data/mr --target_model bert --nclasses 2 --perturb_model TextFooler --result_model_path_file adversaries_mr_bert_TextFooler.txt --target_model_path ./BERT/results/mr ' + MAIN_COMMON + RESOURCE
# os.system(command)
# command = PYTHON + '--dataset_path data/mr --target_model bert --nclasses 2 --perturb_model hardlabel --result_model_path_file adversaries_mr_bert_hardlabel.txt --target_model_path ./BERT/results/mr ' + MAIN_COMMON + RESOURCE
# os.system(command)
# command = PYTHON + '--dataset_path data/mr --target_model bert --nclasses 2 --perturb_model limeattack --result_model_path_file adversaries_mr_bert_limeattack.txt --target_model_path ./BERT/results/mr ' + MAIN_COMMON + RESOURCE
# os.system(command)
# command = PYTHON + '--dataset_path data/mr --target_model bert --nclasses 2 --perturb_model ELESS --result_model_path_file adversaries_mr_bert_ELESS.txt --target_model_path ./BERT/results/mr ' + MAIN_COMMON + RESOURCE
# os.system(command)



# ==================== IMDb ====================

# ---------------- IMDb / WordCNN ----------------
# command = PYTHON + '--dataset_path data/imdb --target_model wordCNN --nclasses 2 --perturb_model pww --result_model_path_file adversaries_imdb_cnn_pww.txt --target_model_path ./targetmodelpathcnn/imdb ' + MAIN_COMMON + RESOURCE
# os.system(command)
# command = PYTHON + '--dataset_path data/imdb --target_model wordCNN --nclasses 2 --perturb_model deepWordBug --result_model_path_file adversaries_imdb_cnn_deepWordBug.txt --target_model_path ./targetmodelpathcnn/imdb ' + MAIN_COMMON + RESOURCE
# os.system(command)
# command = PYTHON + '--dataset_path data/imdb --target_model wordCNN --nclasses 2 --perturb_model TextFooler --result_model_path_file adversaries_imdb_cnn_TextFooler.txt --target_model_path ./targetmodelpathcnn/imdb ' + MAIN_COMMON + RESOURCE
# os.system(command)
# command = PYTHON + '--dataset_path data/imdb --target_model wordCNN --nclasses 2 --perturb_model hardlabel --result_model_path_file adversaries_imdb_cnn_hardlabel.txt --target_model_path ./targetmodelpathcnn/imdb ' + MAIN_COMMON + RESOURCE
# os.system(command)
# command = PYTHON + '--dataset_path data/imdb --target_model wordCNN --nclasses 2 --perturb_model limeattack --result_model_path_file adversaries_imdb_cnn_limeattack.txt --target_model_path ./targetmodelpathcnn/imdb ' + MAIN_COMMON + RESOURCE
# os.system(command)
# command = PYTHON + '--dataset_path data/imdb --target_model wordCNN --nclasses 2 --perturb_model ELESS --result_model_path_file adversaries_imdb_cnn_ELESS.txt --target_model_path ./targetmodelpathcnn/imdb ' + MAIN_COMMON + RESOURCE
# os.system(command)

# # ---------------- IMDb / WordLSTM ----------------
# command = PYTHON + '--dataset_path data/imdb --target_model wordLSTM --nclasses 2 --perturb_model pww --result_model_path_file adversaries_imdb_lstm_pww.txt --target_model_path ./targetmodelpathlstm/imdb ' + MAIN_COMMON + RESOURCE
# os.system(command)
# command = PYTHON + '--dataset_path data/imdb --target_model wordLSTM --nclasses 2 --perturb_model deepWordBug --result_model_path_file adversaries_imdb_lstm_deepWordBug.txt --target_model_path ./targetmodelpathlstm/imdb ' + MAIN_COMMON + RESOURCE
# os.system(command)
# command = PYTHON + '--dataset_path data/imdb --target_model wordLSTM --nclasses 2 --perturb_model TextFooler --result_model_path_file adversaries_imdb_lstm_TextFooler.txt --target_model_path ./targetmodelpathlstm/imdb ' + MAIN_COMMON + RESOURCE
# os.system(command)
# command = PYTHON + '--dataset_path data/imdb --target_model wordLSTM --nclasses 2 --perturb_model hardlabel --result_model_path_file adversaries_imdb_lstm_hardlabel.txt --target_model_path ./targetmodelpathlstm/imdb ' + MAIN_COMMON + RESOURCE
# os.system(command)
# command = PYTHON + '--dataset_path data/imdb --target_model wordLSTM --nclasses 2 --perturb_model limeattack --result_model_path_file adversaries_imdb_lstm_limeattack.txt --target_model_path ./targetmodelpathlstm/imdb ' + MAIN_COMMON + RESOURCE
# os.system(command)
# command = PYTHON + '--dataset_path data/imdb --target_model wordLSTM --nclasses 2 --perturb_model ELESS --result_model_path_file adversaries_imdb_lstm_ELESS.txt --target_model_path ./targetmodelpathlstm/imdb ' + MAIN_COMMON + RESOURCE
# os.system(command)

# # ---------------- IMDb / BERT ----------------
# command = PYTHON + '--dataset_path data/imdb --target_model bert --nclasses 2 --perturb_model pww --result_model_path_file adversaries_imdb_bert_pww.txt --target_model_path ./BERT/results/imdb ' + MAIN_COMMON + RESOURCE
# os.system(command)
# command = PYTHON + '--dataset_path data/imdb --target_model bert --nclasses 2 --perturb_model deepWordBug --result_model_path_file adversaries_imdb_bert_deepWordBug.txt --target_model_path ./BERT/results/imdb ' + MAIN_COMMON + RESOURCE
# os.system(command)
# command = PYTHON + '--dataset_path data/imdb --target_model bert --nclasses 2 --perturb_model TextFooler --result_model_path_file adversaries_imdb_bert_TextFooler.txt --target_model_path ./BERT/results/imdb ' + MAIN_COMMON + RESOURCE
# os.system(command)
# command = PYTHON + '--dataset_path data/imdb --target_model bert --nclasses 2 --perturb_model hardlabel --result_model_path_file adversaries_imdb_bert_hardlabel.txt --target_model_path ./BERT/results/imdb ' + MAIN_COMMON + RESOURCE
# os.system(command)
# command = PYTHON + '--dataset_path data/imdb --target_model bert --nclasses 2 --perturb_model limeattack --result_model_path_file adversaries_imdb_bert_limeattack.txt --target_model_path ./BERT/results/imdb ' + MAIN_COMMON + RESOURCE
# os.system(command)
# command = PYTHON + '--dataset_path data/imdb --target_model bert --nclasses 2 --perturb_model ELESS --result_model_path_file adversaries_imdb_bert_ELESS.txt --target_model_path ./BERT/results/imdb ' + MAIN_COMMON + RESOURCE
# os.system(command)


# # ==================== Yelp ====================

# # ---------------- Yelp / WordCNN ----------------
# command = PYTHON + '--dataset_path data/yelp --target_model wordCNN --nclasses 2 --perturb_model pww --result_model_path_file adversaries_yelp_cnn_pww.txt --target_model_path ./targetmodelpathcnn/yelp ' + MAIN_COMMON + RESOURCE
# os.system(command)
# command = PYTHON + '--dataset_path data/yelp --target_model wordCNN --nclasses 2 --perturb_model deepWordBug --result_model_path_file adversaries_yelp_cnn_deepWordBug.txt --target_model_path ./targetmodelpathcnn/yelp ' + MAIN_COMMON + RESOURCE
# os.system(command)
# command = PYTHON + '--dataset_path data/yelp --target_model wordCNN --nclasses 2 --perturb_model TextFooler --result_model_path_file adversaries_yelp_cnn_TextFooler.txt --target_model_path ./targetmodelpathcnn/yelp ' + MAIN_COMMON + RESOURCE
# os.system(command)
# command = PYTHON + '--dataset_path data/yelp --target_model wordCNN --nclasses 2 --perturb_model hardlabel --result_model_path_file adversaries_yelp_cnn_hardlabel.txt --target_model_path ./targetmodelpathcnn/yelp ' + MAIN_COMMON + RESOURCE
# os.system(command)
# command = PYTHON + '--dataset_path data/yelp --target_model wordCNN --nclasses 2 --perturb_model limeattack --result_model_path_file adversaries_yelp_cnn_limeattack.txt --target_model_path ./targetmodelpathcnn/yelp ' + MAIN_COMMON + RESOURCE
# os.system(command)
# command = PYTHON + '--dataset_path data/yelp --target_model wordCNN --nclasses 2 --perturb_model ELESS --result_model_path_file adversaries_yelp_cnn_ELESS.txt --target_model_path ./targetmodelpathcnn/yelp ' + MAIN_COMMON + RESOURCE
# os.system(command)

# # ---------------- Yelp / WordLSTM ----------------
# command = PYTHON + '--dataset_path data/yelp --target_model wordLSTM --nclasses 2 --perturb_model pww --result_model_path_file adversaries_yelp_lstm_pww.txt --target_model_path ./targetmodelpathlstm/yelp ' + MAIN_COMMON + RESOURCE
# os.system(command)
# command = PYTHON + '--dataset_path data/yelp --target_model wordLSTM --nclasses 2 --perturb_model deepWordBug --result_model_path_file adversaries_yelp_lstm_deepWordBug.txt --target_model_path ./targetmodelpathlstm/yelp ' + MAIN_COMMON + RESOURCE
# os.system(command)
# command = PYTHON + '--dataset_path data/yelp --target_model wordLSTM --nclasses 2 --perturb_model TextFooler --result_model_path_file adversaries_yelp_lstm_TextFooler.txt --target_model_path ./targetmodelpathlstm/yelp ' + MAIN_COMMON + RESOURCE
# os.system(command)
# command = PYTHON + '--dataset_path data/yelp --target_model wordLSTM --nclasses 2 --perturb_model hardlabel --result_model_path_file adversaries_yelp_lstm_hardlabel.txt --target_model_path ./targetmodelpathlstm/yelp ' + MAIN_COMMON + RESOURCE
# os.system(command)
# command = PYTHON + '--dataset_path data/yelp --target_model wordLSTM --nclasses 2 --perturb_model limeattack --result_model_path_file adversaries_yelp_lstm_limeattack.txt --target_model_path ./targetmodelpathlstm/yelp ' + MAIN_COMMON + RESOURCE
# os.system(command)
# command = PYTHON + '--dataset_path data/yelp --target_model wordLSTM --nclasses 2 --perturb_model ELESS --result_model_path_file adversaries_yelp_lstm_ELESS.txt --target_model_path ./targetmodelpathlstm/yelp ' + MAIN_COMMON + RESOURCE
# os.system(command)

# # ---------------- Yelp / BERT ----------------
# command = PYTHON + '--dataset_path data/yelp --target_model bert --nclasses 2 --perturb_model pww --result_model_path_file adversaries_yelp_bert_pww.txt --target_model_path ./BERT/results/yelp ' + MAIN_COMMON + RESOURCE
# os.system(command)
# command = PYTHON + '--dataset_path data/yelp --target_model bert --nclasses 2 --perturb_model deepWordBug --result_model_path_file adversaries_yelp_bert_deepWordBug.txt --target_model_path ./BERT/results/yelp ' + MAIN_COMMON + RESOURCE
# os.system(command)
# command = PYTHON + '--dataset_path data/yelp --target_model bert --nclasses 2 --perturb_model TextFooler --result_model_path_file adversaries_yelp_bert_TextFooler.txt --target_model_path ./BERT/results/yelp ' + MAIN_COMMON + RESOURCE
# os.system(command)
# command = PYTHON + '--dataset_path data/yelp --target_model bert --nclasses 2 --perturb_model hardlabel --result_model_path_file adversaries_yelp_bert_hardlabel.txt --target_model_path ./BERT/results/yelp ' + MAIN_COMMON + RESOURCE
# os.system(command)
# command = PYTHON + '--dataset_path data/yelp --target_model bert --nclasses 2 --perturb_model limeattack --result_model_path_file adversaries_yelp_bert_limeattack.txt --target_model_path ./BERT/results/yelp ' + MAIN_COMMON + RESOURCE
# os.system(command)
# command = PYTHON + '--dataset_path data/yelp --target_model bert --nclasses 2 --perturb_model ELESS --result_model_path_file adversaries_yelp_bert_ELESS.txt --target_model_path ./BERT/results/yelp ' + MAIN_COMMON + RESOURCE
# os.system(command)


# # ==================== AG ====================

# # ---------------- AG / WordCNN ----------------
# command = PYTHON + '--dataset_path data/ag --target_model wordCNN --nclasses 4 --perturb_model pww --result_model_path_file adversaries_ag_cnn_pww.txt --target_model_path ./targetmodelpathcnn/ag ' + MAIN_COMMON + RESOURCE
# os.system(command)
# command = PYTHON + '--dataset_path data/ag --target_model wordCNN --nclasses 4 --perturb_model deepWordBug --result_model_path_file adversaries_ag_cnn_deepWordBug.txt --target_model_path ./targetmodelpathcnn/ag ' + MAIN_COMMON + RESOURCE
# os.system(command)
# command = PYTHON + '--dataset_path data/ag --target_model wordCNN --nclasses 4 --perturb_model TextFooler --result_model_path_file adversaries_ag_cnn_TextFooler.txt --target_model_path ./targetmodelpathcnn/ag ' + MAIN_COMMON + RESOURCE
# os.system(command)
# command = PYTHON + '--dataset_path data/ag --target_model wordCNN --nclasses 4 --perturb_model hardlabel --result_model_path_file adversaries_ag_cnn_hardlabel.txt --target_model_path ./targetmodelpathcnn/ag ' + MAIN_COMMON + RESOURCE
# os.system(command)
# command = PYTHON + '--dataset_path data/ag --target_model wordCNN --nclasses 4 --perturb_model limeattack --result_model_path_file adversaries_ag_cnn_limeattack.txt --target_model_path ./targetmodelpathcnn/ag ' + MAIN_COMMON + RESOURCE
# os.system(command)
# command = PYTHON + '--dataset_path data/ag --target_model wordCNN --nclasses 4 --perturb_model ELESS --result_model_path_file adversaries_ag_cnn_ELESS.txt --target_model_path ./targetmodelpathcnn/ag ' + MAIN_COMMON + RESOURCE
# os.system(command)

# # ---------------- AG / WordLSTM ----------------
# command = PYTHON + '--dataset_path data/ag --target_model wordLSTM --nclasses 4 --perturb_model pww --result_model_path_file adversaries_ag_lstm_pww.txt --target_model_path ./targetmodelpathlstm/ag ' + MAIN_COMMON + RESOURCE
# os.system(command)
# command = PYTHON + '--dataset_path data/ag --target_model wordLSTM --nclasses 4 --perturb_model deepWordBug --result_model_path_file adversaries_ag_lstm_deepWordBug.txt --target_model_path ./targetmodelpathlstm/ag ' + MAIN_COMMON + RESOURCE
# os.system(command)
# command = PYTHON + '--dataset_path data/ag --target_model wordLSTM --nclasses 4 --perturb_model TextFooler --result_model_path_file adversaries_ag_lstm_TextFooler.txt --target_model_path ./targetmodelpathlstm/ag ' + MAIN_COMMON + RESOURCE
# os.system(command)
# command = PYTHON + '--dataset_path data/ag --target_model wordLSTM --nclasses 4 --perturb_model hardlabel --result_model_path_file adversaries_ag_lstm_hardlabel.txt --target_model_path ./targetmodelpathlstm/ag ' + MAIN_COMMON + RESOURCE
# os.system(command)
# command = PYTHON + '--dataset_path data/ag --target_model wordLSTM --nclasses 4 --perturb_model limeattack --result_model_path_file adversaries_ag_lstm_limeattack.txt --target_model_path ./targetmodelpathlstm/ag ' + MAIN_COMMON + RESOURCE
# os.system(command)
# command = PYTHON + '--dataset_path data/ag --target_model wordLSTM --nclasses 4 --perturb_model ELESS --result_model_path_file adversaries_ag_lstm_ELESS.txt --target_model_path ./targetmodelpathlstm/ag ' + MAIN_COMMON + RESOURCE
# os.system(command)

# # ---------------- AG / BERT ----------------
# command = PYTHON + '--dataset_path data/ag --target_model bert --nclasses 4 --perturb_model pww --result_model_path_file adversaries_ag_bert_pww.txt --target_model_path ./BERT/results/ag ' + MAIN_COMMON + RESOURCE
# os.system(command)
# command = PYTHON + '--dataset_path data/ag --target_model bert --nclasses 4 --perturb_model deepWordBug --result_model_path_file adversaries_ag_bert_deepWordBug.txt --target_model_path ./BERT/results/ag ' + MAIN_COMMON + RESOURCE
# os.system(command)
# command = PYTHON + '--dataset_path data/ag --target_model bert --nclasses 4 --perturb_model TextFooler --result_model_path_file adversaries_ag_bert_TextFooler.txt --target_model_path ./BERT/results/ag ' + MAIN_COMMON + RESOURCE
# os.system(command)
# command = PYTHON + '--dataset_path data/ag --target_model bert --nclasses 4 --perturb_model hardlabel --result_model_path_file adversaries_ag_bert_hardlabel.txt --target_model_path ./BERT/results/ag ' + MAIN_COMMON + RESOURCE
# os.system(command)
# command = PYTHON + '--dataset_path data/ag --target_model bert --nclasses 4 --perturb_model limeattack --result_model_path_file adversaries_ag_bert_limeattack.txt --target_model_path ./BERT/results/ag ' + MAIN_COMMON + RESOURCE
# os.system(command)
# command = PYTHON + '--dataset_path data/ag --target_model bert --nclasses 4 --perturb_model ELESS --result_model_path_file adversaries_ag_bert_ELESS.txt --target_model_path ./BERT/results/ag ' + MAIN_COMMON + RESOURCE
# os.system(command)











# ============================================================
# 2. Primary ablation study
# Use MR + WordCNN for efficient controlled ablation.
# Same dataset/model/K/rho/seed; only switch one component.
# ============================================================

ABLATION_COMMON = \
    '--nclasses 2 --perturb_model ELESS ' \
    '--data_size 500 --synonym_num 50 --sim_score_window 15 ' \
    '--sim_score_threshold 0.5 --perturb_budget 0.30 ' \
    '--initial_temperature 0.10 --cooling_factor 0.95 ' \
    '--energy_zeta 0 --max_query_budget 0 --random_seed 1234 --output_dir /root/autodl-fs/adv_results_duibi ' + RESOURCE

# # MR + WordCNN
# command = PYTHON + '--target_model wordCNN --dataset_path data/mr --target_model_path ./targetmodelpathcnn/mr --result_model_path_file ablation_mr_cnn_full.txt --use_ranking 1 --use_local_semantics 1 --use_energy_search 1 ' + ABLATION_COMMON
# os.system(command)

# command = PYTHON + '--target_model wordCNN --dataset_path data/mr --target_model_path ./targetmodelpathcnn/mr --result_model_path_file ablation_mr_cnn_noRank.txt --use_ranking 0 --use_local_semantics 1 --use_energy_search 1 ' + ABLATION_COMMON
# os.system(command)

# command = PYTHON + '--target_model wordCNN --dataset_path data/mr --target_model_path ./targetmodelpathcnn/mr --result_model_path_file ablation_mr_cnn_noLocal.txt --use_ranking 1 --use_local_semantics 0 --use_energy_search 1 ' + ABLATION_COMMON
# os.system(command)

# command = PYTHON + '--target_model wordCNN --dataset_path data/mr --target_model_path ./targetmodelpathcnn/mr --result_model_path_file ablation_mr_cnn_noEnergy.txt --use_ranking 1 --use_local_semantics 1 --use_energy_search 0 ' + ABLATION_COMMON
# os.system(command)

# ---------------- MR / WordLSTM ----------------
# command = PYTHON + '--dataset_path data/mr --target_model wordLSTM --target_model_path ./targetmodelpathlstm/mr --result_model_path_file ablation_mr_lstm_full.txt --use_ranking 1 --use_local_semantics 1 --use_energy_search 1 ' + ABLATION_COMMON
# os.system(command)

# command = PYTHON + '--dataset_path data/mr --target_model wordLSTM --target_model_path ./targetmodelpathlstm/mr --result_model_path_file ablation_mr_lstm_noRank.txt --use_ranking 0 --use_local_semantics 1 --use_energy_search 1 ' + ABLATION_COMMON
# os.system(command)

# command = PYTHON + '--dataset_path data/mr --target_model wordLSTM --target_model_path ./targetmodelpathlstm/mr --result_model_path_file ablation_mr_lstm_noLocal.txt --use_ranking 1 --use_local_semantics 0 --use_energy_search 1 ' + ABLATION_COMMON
# os.system(command)

# command = PYTHON + '--dataset_path data/mr --target_model wordLSTM --target_model_path ./targetmodelpathlstm/mr --result_model_path_file ablation_mr_lstm_noEnergy.txt --use_ranking 1 --use_local_semantics 1 --use_energy_search 0 ' + ABLATION_COMMON
# os.system(command)


# ---------------- MR / BERT ----------------
# command = PYTHON + '--dataset_path data/mr --target_model bert --target_model_path ./BERT/results/mr --result_model_path_file ablation_mr_bert_full.txt --use_ranking 1 --use_local_semantics 1 --use_energy_search 1 ' + ABLATION_COMMON
# os.system(command)

# command = PYTHON + '--dataset_path data/mr --target_model bert --target_model_path ./BERT/results/mr --result_model_path_file ablation_mr_bert_noRank.txt --use_ranking 0 --use_local_semantics 1 --use_energy_search 1 ' + ABLATION_COMMON
# os.system(command)

# command = PYTHON + '--dataset_path data/mr --target_model bert --target_model_path ./BERT/results/mr --result_model_path_file ablation_mr_bert_noLocal.txt --use_ranking 1 --use_local_semantics 0 --use_energy_search 1 ' + ABLATION_COMMON
# os.system(command)

command = PYTHON + '--dataset_path data/mr --target_model bert --target_model_path ./BERT/results/mr --result_model_path_file ablation_mr_bert_noEnergy.txt --use_ranking 1 --use_local_semantics 1 --use_energy_search 0 ' + ABLATION_COMMON
os.system(command)



# ============================================================
# 5. zeta sensitivity
# AG + WordCNN
# zeta = 1, 10, 100, 1000
# ============================================================

ZETA_COMMON_AG = \
    '--dataset_path data/ag --target_model wordCNN --nclasses 4 --perturb_model ELESS ' \
    '--target_model_path ./targetmodelpathcnn/ag --data_size 500 --synonym_num 50 ' \
    '--sim_score_window 15 --sim_score_threshold 0.5 --perturb_budget 0.30 ' \
    '--initial_temperature 0.10 --cooling_factor 0.95 --random_seed 1234 ' \
    '--use_ranking 1 --use_local_semantics 1 --use_energy_search 1 --output_dir /root/autodl-fs/adv_results_duibi' + RESOURCE

# command = PYTHON + '--energy_zeta 1 --result_model_path_file zeta_ag_cnn_1.txt ' + ZETA_COMMON_AG
# os.system(command)
# command = PYTHON + '--energy_zeta 10 --result_model_path_file zeta_ag_cnn_10.txt ' + ZETA_COMMON_AG
# os.system(command)
# command = PYTHON + '--energy_zeta 100 --result_model_path_file zeta_ag_cnn_100.txt ' + ZETA_COMMON_AG
# os.system(command)
# command = PYTHON + '--energy_zeta 1000 --result_model_path_file zeta_ag_cnn_1000.txt ' + ZETA_COMMON_AG
# os.system(command)


# ============================================================
# 6. zeta sensitivity
# IMDb + WordCNN
# zeta = 1, 10, 100, 1000
# ============================================================

ZETA_COMMON_IMDB_CNN = \
    '--dataset_path data/imdb --target_model wordCNN --nclasses 2 --perturb_model ELESS ' \
    '--target_model_path ./targetmodelpathcnn/imdb --data_size 500 --synonym_num 50 ' \
    '--sim_score_window 15 --sim_score_threshold 0.5 --perturb_budget 0.30 ' \
    '--initial_temperature 0.10 --cooling_factor 0.95 --random_seed 1234 ' \
    '--use_ranking 1 --use_local_semantics 1 --use_energy_search 1 ' + RESOURCE

# command = PYTHON + '--energy_zeta 1 --result_model_path_file zeta_imdb_cnn_1.txt ' + ZETA_COMMON_IMDB_CNN
# os.system(command)

# command = PYTHON + '--energy_zeta 10 --result_model_path_file zeta_imdb_cnn_10.txt ' + ZETA_COMMON_IMDB_CNN
# os.system(command)

# command = PYTHON + '--energy_zeta 100 --result_model_path_file zeta_imdb_cnn_100.txt ' + ZETA_COMMON_IMDB_CNN
# os.system(command)

# command = PYTHON + '--energy_zeta 1000 --result_model_path_file zeta_imdb_cnn_1000.txt ' + ZETA_COMMON_IMDB_CNN
# os.system(command)


# ============================================================
# 6. Synonym candidate size K sensitivity
# AG + WordCNN
# K = 10, 20, 50, 100, 200
# ============================================================

K_COMMON_AG = \
    '--dataset_path data/ag --target_model wordCNN --nclasses 4 --perturb_model ELESS ' \
    '--target_model_path ./targetmodelpathcnn/ag --data_size 500 ' \
    '--sim_score_window 15 --sim_score_threshold 0.5 --perturb_budget 0.30 ' \
    '--initial_temperature 0.10 --cooling_factor 0.95 --energy_zeta 0 ' \
    '--random_seed 1234 --use_ranking 1 --use_local_semantics 1 --use_energy_search 1 --output_dir /root/autodl-fs/adv_results_duibi' + RESOURCE

# command = PYTHON + '--synonym_num 10 --result_model_path_file k_ag_cnn_10.txt ' + K_COMMON_AG
# os.system(command)
# command = PYTHON + '--synonym_num 20 --result_model_path_file k_ag_cnn_20.txt ' + K_COMMON_AG
# os.system(command)
# command = PYTHON + '--synonym_num 50 --result_model_path_file k_ag_cnn_50.txt ' + K_COMMON_AG
# os.system(command)
# command = PYTHON + '--synonym_num 100 --result_model_path_file k_ag_cnn_100.txt ' + K_COMMON_AG
# os.system(command)
# command = PYTHON + '--synonym_num 200 --result_model_path_file k_ag_cnn_200.txt ' + K_COMMON_AG
# os.system(command)


# ============================================================
# 7. Local semantic window sensitivity
# MR + WordCNN
# window = 5, 9, 15, 21, 31
# ============================================================

WINDOW_COMMON = \
    '--dataset_path data/mr --target_model wordCNN --nclasses 2 --perturb_model ELESS ' \
    '--target_model_path ./targetmodelpathcnn/mr --data_size 500 --synonym_num 50 ' \
    '--sim_score_threshold 0.5 --perturb_budget 0.30 --initial_temperature 0.10 ' \
    '--cooling_factor 0.95 --energy_zeta 0 --random_seed 1234 ' \
    '--use_ranking 1 --use_local_semantics 1 --use_energy_search 1 --output_dir /root/autodl-fs/adv_results_duibi' + RESOURCE

# command = PYTHON + '--sim_score_window 5 --result_model_path_file window_mr_cnn_5.txt ' + WINDOW_COMMON
# os.system(command)
# command = PYTHON + '--sim_score_window 9 --result_model_path_file window_mr_cnn_9.txt ' + WINDOW_COMMON
# os.system(command)
# command = PYTHON + '--sim_score_window 15 --result_model_path_file window_mr_cnn_15.txt ' + WINDOW_COMMON
# os.system(command)
# command = PYTHON + '--sim_score_window 21 --result_model_path_file window_mr_cnn_21.txt ' + WINDOW_COMMON
# os.system(command)
# command = PYTHON + '--sim_score_window 31 --result_model_path_file window_mr_cnn_31.txt ' + WINDOW_COMMON
# os.system(command)


# ============================================================
# 8. Cooling factor alpha sensitivity
# MR + WordCNN
# alpha = 0.70, 0.80, 0.90, 0.95, 0.99
# ============================================================

ALPHA_COMMON = \
    '--dataset_path data/mr --target_model wordCNN --nclasses 2 --perturb_model ELESS ' \
    '--target_model_path ./targetmodelpathcnn/mr --data_size 500 --synonym_num 50 ' \
    '--sim_score_window 15 --sim_score_threshold 0.5 --perturb_budget 0.30 ' \
    '--initial_temperature 0.10 --energy_zeta 0 --random_seed 1234 ' \
    '--use_ranking 1 --use_local_semantics 1 --use_energy_search 1 --output_dir /root/autodl-fs/adv_results_duibi' + RESOURCE

# command = PYTHON + '--cooling_factor 0.70 --result_model_path_file alpha_mr_cnn_070.txt ' + ALPHA_COMMON
# os.system(command)
# command = PYTHON + '--cooling_factor 0.80 --result_model_path_file alpha_mr_cnn_080.txt ' + ALPHA_COMMON
# os.system(command)
# command = PYTHON + '--cooling_factor 0.90 --result_model_path_file alpha_mr_cnn_090.txt ' + ALPHA_COMMON
# os.system(command)
# command = PYTHON + '--cooling_factor 0.95 --result_model_path_file alpha_mr_cnn_095.txt ' + ALPHA_COMMON
# os.system(command)
# command = PYTHON + '--cooling_factor 0.99 --result_model_path_file alpha_mr_cnn_099.txt ' + ALPHA_COMMON
# os.system(command)


# ============================================================
# 9. Perturbation budget rho
# MR + WordCNN
# rho = 0.05, 0.10, 0.15, 0.20, 0.30
# ============================================================

RHO_COMMON_MR = \
    '--dataset_path data/mr --target_model wordCNN --nclasses 2 --perturb_model ELESS ' \
    '--target_model_path ./targetmodelpathcnn/mr --data_size 500 --synonym_num 50 ' \
    '--sim_score_window 15 --sim_score_threshold 0.5 --initial_temperature 0.10 ' \
    '--cooling_factor 0.95 --energy_zeta 0 --random_seed 1234 ' \
    '--use_ranking 1 --use_local_semantics 1 --use_energy_search 1 --output_dir /root/autodl-fs/adv_results_duibi' + RESOURCE

# command = PYTHON + '--perturb_budget 0.05 --result_model_path_file rho_mr_cnn_005.txt ' + RHO_COMMON_MR
# os.system(command)
# command = PYTHON + '--perturb_budget 0.10 --result_model_path_file rho_mr_cnn_010.txt ' + RHO_COMMON_MR
# os.system(command)
# command = PYTHON + '--perturb_budget 0.15 --result_model_path_file rho_mr_cnn_015.txt ' + RHO_COMMON_MR
# os.system(command)
# command = PYTHON + '--perturb_budget 0.20 --result_model_path_file rho_mr_cnn_020.txt ' + RHO_COMMON_MR
# os.system(command)
# command = PYTHON + '--perturb_budget 0.30 --result_model_path_file rho_mr_cnn_030.txt ' + RHO_COMMON_MR
# os.system(command)


# ============================================================
# 10. Query budget
# MR + WordCNN and AG + WordCNN
# Budgets: 50, 100, 200, 500, 1000
# ============================================================

QUERY_COMMON_MR_CNN = \
    '--dataset_path data/mr --target_model wordCNN --nclasses 2 --perturb_model ELESS ' \
    '--target_model_path ./targetmodelpathcnn/mr --data_size 500 --synonym_num 50 ' \
    '--sim_score_window 15 --sim_score_threshold 0.5 --perturb_budget 0.30 ' \
    '--initial_temperature 0.10 --cooling_factor 0.95 --energy_zeta 0 ' \
    '--random_seed 1234 --use_ranking 1 --use_local_semantics 1 --use_energy_search 1 --output_dir /root/autodl-fs/adv_results_duibi' + RESOURCE

QUERY_COMMON_AG_CNN = \
    '--dataset_path data/ag --target_model wordCNN --nclasses 4 --perturb_model ELESS ' \
    '--target_model_path ./targetmodelpathcnn/ag --data_size 500 --synonym_num 50 ' \
    '--sim_score_window 15 --sim_score_threshold 0.5 --perturb_budget 0.30 ' \
    '--initial_temperature 0.10 --cooling_factor 0.95 --energy_zeta 0 ' \
    '--random_seed 1234 --use_ranking 1 --use_local_semantics 1 --use_energy_search 1 --output_dir /root/autodl-fs/adv_results_duibi' + RESOURCE

# MR + WordCNN
# command = PYTHON + '--max_query_budget 50 --result_model_path_file query_mr_cnn_50.txt ' + QUERY_COMMON_MR_CNN
# os.system(command)
# command = PYTHON + '--max_query_budget 100 --result_model_path_file query_mr_cnn_100.txt ' + QUERY_COMMON_MR_CNN
# os.system(command)
# command = PYTHON + '--max_query_budget 200 --result_model_path_file query_mr_cnn_200.txt ' + QUERY_COMMON_MR_CNN
# os.system(command)
# command = PYTHON + '--max_query_budget 500 --result_model_path_file query_mr_cnn_500.txt ' + QUERY_COMMON_MR_CNN
# os.system(command)
# command = PYTHON + '--max_query_budget 1000 --result_model_path_file query_mr_cnn_1000.txt ' + QUERY_COMMON_MR_CNN
# os.system(command)

# AG + WordCNN
# command = PYTHON + '--max_query_budget 50 --result_model_path_file query_ag_cnn_50.txt ' + QUERY_COMMON_AG_CNN
# os.system(command)
# command = PYTHON + '--max_query_budget 100 --result_model_path_file query_ag_cnn_100.txt ' + QUERY_COMMON_AG_CNN
# os.system(command)
# command = PYTHON + '--max_query_budget 200 --result_model_path_file query_ag_cnn_200.txt ' + QUERY_COMMON_AG_CNN
# os.system(command)
# command = PYTHON + '--max_query_budget 500 --result_model_path_file query_ag_cnn_500.txt ' + QUERY_COMMON_AG_CNN
# os.system(command)
# command = PYTHON + '--max_query_budget 1000 --result_model_path_file query_ag_cnn_1000.txt ' + QUERY_COMMON_AG_CNN
# os.system(command)


# ============================================================
# 11. Repeated-run stability
# MR + WordCNN
# Same parameters; only random_seed changes.
# ============================================================

STABILITY_COMMON_MR_CNN = \
    '--dataset_path data/mr --target_model wordCNN --nclasses 2 --perturb_model ELESS ' \
    '--target_model_path ./targetmodelpathcnn/mr --data_size 500 --synonym_num 50 ' \
    '--sim_score_window 15 --sim_score_threshold 0.5 --perturb_budget 0.30 ' \
    '--initial_temperature 0.10 --cooling_factor 0.95 --energy_zeta 0 ' \
    '--use_ranking 1 --use_local_semantics 1 --use_energy_search 1 --output_dir /root/autodl-fs/adv_results_duibi' + RESOURCE

# command = PYTHON + '--random_seed 1234 --result_model_path_file stability_mr_cnn_1234.txt ' + STABILITY_COMMON_MR_CNN
# os.system(command)
# command = PYTHON + '--random_seed 2024 --result_model_path_file stability_mr_cnn_2024.txt ' + STABILITY_COMMON_MR_CNN
# os.system(command)
# command = PYTHON + '--random_seed 3407 --result_model_path_file stability_mr_cnn_3407.txt ' + STABILITY_COMMON_MR_CNN
# os.system(command)
# command = PYTHON + '--random_seed 5678 --result_model_path_file stability_mr_cnn_5678.txt ' + STABILITY_COMMON_MR_CNN
# os.system(command)
# command = PYTHON + '--random_seed 9999 --result_model_path_file stability_mr_cnn_9999.txt ' + STABILITY_COMMON_MR_CNN
# os.system(command)


# ============================================================
# 12. Experiments that do NOT need a new attack parameter sweep
# ============================================================
# Input-length analysis:
#   Reuse normal ELESS sample-level logs and split successful/eligible
#   samples into Short/Medium/Long/Very long groups offline.
#
# Automatic quality / efficiency / significance:
#   Reuse outputs from the main comparison and evaluate
#   grammar/perplexity/syntax/time/statistical significance offline.
#
# Transferability:
#   Generate ELESS adversarial examples from WordCNN/WordLSTM/BERT
#   in the main comparison and evaluate the saved adversarial texts
#   on other target models with a separate evaluator.
#
# Defense / component-generality / additional pretrained encoders:
#   These require extra model/integration code and are not simple
#   parameter sweeps of attack_classification.py.