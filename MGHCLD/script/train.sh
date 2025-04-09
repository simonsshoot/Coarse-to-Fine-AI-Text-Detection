#some training commands for reference

#HC3 first
 python train_classifier.py --device_num 1 --per_gpu_batch_size 10 --total_epoch 30 --lr 2e-5 --warmup_steps 2000\
 --dataset HC3_origin --path MGHCLD/detect_data/HC3_origin \
     --name onlyclass-first-HC3origin  --database_name train --test_dataset_name valid --only_classifier --save_csv MGHCLD/results/detect_data/middle/train.csv --per_gpu_batch_size 8 --per_gpu_eval_batch_size 8
    
#HC3 second:
python train_classifier.py --device_num 1 --per_gpu_batch_size 8 --total_epoch 15 --lr 2e-5 --warmup_steps 600\
 --dataset middle --path MGHCLD/detect_data/middle \
     --name onlyclass-second-HC3attack --freeze_embedding_layer --database_name train --test_dataset_name test  --save_csv MGHCLD/results/savetrain.csv

#seqxgpt first:
 python train_classifier.py --device_num 1  --total_epoch 30 --lr 2e-5 --warmup_steps 2000 --dataset seqxgpt_origin --path MGHCLD/detect_data/seqxgpt_origin \
     --name onlyclass-first-seqxgptorigin  --database_name train --test_dataset_name valid --only_classifier --save_csv MGHCLD/results/detect_data/middle/train.csv --per_gpu_eval_batch_size 32 
    
#seqxgpt second:
python train_classifier.py --device_num 1 --per_gpu_batch_size 8 --total_epoch 15 --lr 2e-5 --warmup_steps 600\
 --dataset middle --path MGHCLD/detect_data/middle \
     --name onlyclass-second-seqxgptattack --freeze_embedding_layer --database_name train --test_dataset_name test  --save_csv MGHCLD/results/savetrain.csv

#checkgpt first:
 python train_classifier.py --device_num 1  --total_epoch 30 --lr 2e-5 --warmup_steps 1500\
 --dataset checkgpt_origin --path MGHCLD/detect_data/checkgpt_origin \
     --name onlyclass-first-checkgptorigin  --database_name train --test_dataset_name valid --only_classifier --save_csv MGHCLD/results/detect_data/middle/train.csv --per_gpu_eval_batch_size 32

#checkgpt second:
python train_classifier.py --device_num 1 --per_gpu_batch_size 8 --total_epoch 15 --lr 2e-5 --warmup_steps 600\
 --dataset middle --path MGHCLD/detect_data/middle \
     --name onlyclass-second-checkgptattack --freeze_embedding_layer --database_name train --test_dataset_name test  --save_csv MGHCLD/results/savetrain.csv

#humanized attack:
#######################################################

#checkgpt first:
python train_classifier.py --device_num 1 --total_epoch 15 --lr 2e-5 --warmup_steps 1000 --dataset checkgpt --path MGHCLD/detect_data/checkgpt \
     --name oneclass-first-checkgpt --only_classifier --freeze_embedding_layer --database_name train --test_dataset_name test  --save_csv MGHCLD/results/savetrain.csv --per_gpu_eval_batch_size 32


python train_classifier.py --device_num 1 --total_epoch 20 --lr 3e-5 --warmup_steps 2000\
 --dataset middle --path MGHCLD/detect_data/middle \
     --name onlyclass-second-checkgpt --freeze_embedding_layer --database_name train --test_dataset_name test  --save_csv MGHCLD/results/savetrain.csv


#train deberta:
python train_classifier.py --device_num 1 --per_gpu_eval_batch_size 16 --total_epoch 20 --lr 1e-5 --warmup_steps 500\
 --dataset middle --path MGHCLD/detect_data/middle \
     --name checkgptorigindeberta --freeze_embedding_layer --database_name train --test_dataset_name test  --save_csv MGHCLD/results/savetrain.csv --model_name /models/deberta-v3-base

python train_classifier.py --device_num 1 --per_gpu_eval_batch_size 16 --total_epoch 25 --lr 3e-5 --warmup_steps 1500\
 --dataset middle --path MGHCLD/detect_data/middle \
     --name checkgptattackdeberta --freeze_embedding_layer --database_name train --test_dataset_name test  --save_csv MGHCLD/results/savetrain.csv --model_name /models/deberta-v3-base --per_gpu_batch_size 16




#train radar patch:
python train_classifier.py --device_num 1 --per_gpu_eval_batch_size 16 --total_epoch 20 --lr 1e-5 --warmup_steps 500\
 --dataset middle --path MGHCLD/detect_data/middle \
     --name checkgptoriginradar --freeze_embedding_layer --database_name train --test_dataset_name test  --save_csv MGHCLD/results/savetrain.csv 


python train_classifier.py --device_num 1 --per_gpu_eval_batch_size 16 --total_epoch 25 --lr 2e-5 --warmup_steps 1500\
 --dataset middle --path MGHCLD/detect_data/middle \
     --name checkgptattacksecondradar --freeze_embedding_layer --database_name train --test_dataset_name test  --save_csv MGHCLD/results/savetrain.csv --per_gpu_batch_size 16


#directly train the second layer based on the humanized datasets:
python train_classifier.py --device_num 3 --per_gpu_batch_size 10 --total_epoch 25 --lr 4e-5 --warmup_steps 2000\
 --dataset checkgpt --path MGHCLD/detect_data/checkgpt \
     --name directattacksecond --freeze_embedding_layer --database_name train --test_dataset_name test  --save_csv MGHCLD/results/savetrain.csv --per_gpu_batch_size 8 --per_gpu_eval_batch_size 8