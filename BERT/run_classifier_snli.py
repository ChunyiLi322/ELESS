import os


command = 'python run_classifier_1.py --data_dir /root/autodl-tmp/ELESS/targetmodelpathbert/snli_1.0 ' \
          '--bert_model /root/autodl-tmp/ELESS/BERT/bert_small --max_seq_length 128 --train_batch_size 128 ' \
          '--task_name snli --output_dir /root/autodl-tmp/ELESS/BERT/results/snli --cache_dir pytorch_cache --do_train  --do_eval --do_lower_case '

# '--do_resume'

os.system(command)