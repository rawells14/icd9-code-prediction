python train.py --data_path caml-mimic/mimicdata/mimic3/train_50.csv --vocab caml-mimic//mimicdata/mimic3/vocab.csv --Y 50 --num_epochs 200 --num_layers 2 --heads 4 --dropout 0.2 --patience 10 --criterion prec_at_8 --lr 0.0001 --embed-file ../../mimicdata/mimic3/processed_full.embed --gpu
