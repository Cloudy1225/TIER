python train.py --dataset cora --n_layers 2 --hidden_dim 256 --dropout 0.7 --residual_conn 0 --batch_norm 0 --n_clusters_list "7, 64" --lamda 1
# python train.py --dataset cora --n_layers 2 --hidden_dim 64 --dropout 0.7 --residual_conn 0 --batch_norm 0 --n_clusters_list "6, 64" --lamda 1
# python train.py --dataset citeseer --n_layers 2 --hidden_dim 128 --dropout 0.7 --residual_conn 0 --batch_norm 0 --n_clusters_list "6, 64" --lamda 1
python train.py --dataset citeseer --n_layers 2 --hidden_dim 128 --dropout 0.7 --residual_conn 0 --batch_norm 0 --n_clusters_list "6, 64" --lamda 2
python train.py --dataset pubmed --n_layers 2 --hidden_dim 256 --dropout 0.5 --residual_conn 0 --batch_norm 0 --n_clusters_list "3, 16, 64" --lamda 1
python train.py --dataset wikics --n_layers 2 --hidden_dim 256 --dropout 0.7 --residual_conn 0 --batch_norm 0 --n_clusters_list "10, 32, 128" --lamda 1
python train.py --dataset photo --n_layers 2 --hidden_dim 64 --dropout 0.7 --residual_conn 0 --batch_norm 0 --n_clusters_list "12, 64, 256" --lamda 1
python train.py --dataset computer --n_layers 3 --hidden_dim 128 --dropout 0.5 --residual_conn 0 --batch_norm 0 --n_clusters_list "10, 128, 512" --lamda 1
python train.py --dataset history --n_layers 2 --hidden_dim 64 --dropout 0.7 --residual_conn 0 --batch_norm 0 --n_clusters_list "12, 64, 256" --lamda 1
python train.py --dataset arxiv --n_layers 2 --hidden_dim 128 --dropout 0.7 --residual_conn 1 --batch_norm 1 --n_clusters_list "40, 128, 512, 2048" --lamda 1
