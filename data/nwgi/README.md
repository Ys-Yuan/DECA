---
dataset_info:
  features:
  - name: question
    dtype: string
  - name: context
    dtype: string
  - name: answer
    dtype: string
  splits:
  - name: train
    num_bytes: 6170396
    num_examples: 16184
  - name: test
    num_bytes: 1548452
    num_examples: 4047
  download_size: 2349327
  dataset_size: 7718848
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  - split: test
    path: data/test-*
---
