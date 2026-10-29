import os

# command = 'python run_classifier.py --data_dir /data/medg/misc/jindi/nlp/datasets/mr ' \
#           '--task_name mr --output_dir results/mr_retrain --cache_dir pytorch_cache --do_train --do_eval ' \
#           '--do_lower_case '


command = 'python run_classifier_1.py --data_dir /root/autodl-tmp/TextRSS/targetmodelpathbert/mr ' \
          '--bert_model /root/autodl-tmp/TextRSS/BERT/bert_small --max_seq_length 128 --train_batch_size 128 ' \
          '--task_name mr --output_dir /root/autodl-tmp/TextRSS/BERT/results/mr --cache_dir pytorch_cache --do_train  --do_eval --do_lower_case '


os.system(command)