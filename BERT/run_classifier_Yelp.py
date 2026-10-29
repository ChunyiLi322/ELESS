import os


command = 'python run_classifier_1.py --data_dir /root/autodl-tmp/ELESS/targetmodelpathbert/yelp ' \
          '--bert_model /root/autodl-tmp/ELESS/BERT/bert_small --max_seq_length 128 --train_batch_size 32 ' \
          '--task_name yelp --output_dir /root/autodl-tmp/ELESS/BERT/results/yelp --cache_dir pytorch_cache --do_train  --do_eval --do_lower_case '

os.system(command)