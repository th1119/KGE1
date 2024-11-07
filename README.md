This is the code for 'Restricted Neighborhood Knowledge Graph Embedding'. This is not the final version of the code. The current part of the code is built upon the contributions made in the article 'Geometry Interaction Knowledge Graph Embeddings'.


Before running the code:

cd code

python process_datasets.py 

On WN18RR:

CUDA_VISIBLE_DEVICES=0 python3 ../run.py --dataset WN18RR --model GIE --rank 300 --regularizer N3 --reg 0.0 --optimizer Adam --max_epochs 150 --patience 15 --valid 5 --batch_size 500 --neg_sample_size 50 --init_size 0.001 --learning_rate 0.001 --gamma 0.0 --bias learn --dtype double --double_neg --multi_c

on FB15k-237:

CUDA_VISIBLE_DEVICES=0 python3 learn.py --dataset FB237 --model GIE --rank 800 --optimizer Adam --learning_rate 0.01 --batch_size 2000 --regularizer N3 --reg 5e-2 --max_epochs 150 --valid 5 -train -id 0 -save
