# Curriculum Learning Meets Directed Acyclic Graph for Multimodal Emotion Recognition (Accepted by LREC-COLING 2024)

Emotion recognition in conversation (ERC) is a crucial task in natural language processing and affective computing. This paper proposes MultiDAG+CL, a novel approach for Multimodal Emotion Recognition in Conversation (ERC) that employs Directed Acyclic Graphs (DAG) to integrate textual, acoustic, and visual features within a unified framework. The model is enhanced by Curriculum Learning (CL) to address challenges related to emotional shifts and data imbalance. Curriculum learning facilitates the learning process by gradually presenting training samples in a meaningful order, thereby improving the model's performance in handling emotional variations and data imbalance. Experimental results on the IEMOCAP and MELD datasets demonstrate that the MultiDAG+CL models outperform baseline models.

## Requirements
* Python 3.6
* PyTorch 1.6.0
* [Transformers](https://github.com/huggingface/transformers) 3.5.1
* CUDA 10.1

### Datasets and Utterance Feature
You can download the dataset and extracted utterance feature from https://drive.google.com/drive/folders/1zCfjx-HhqEY2tQlxvg1X_6T7sB6hVA2T?usp=sharing. Multimodal datasets are only available for IEMOCAP and MELD, marking with "_mm" in their names.

#### Training
To train model with all three modals (not using curriculum learning): 
`!python run.py --dataset_name IEMOCAP --gnn_layers 4 --lr 0.0005 --batch_size 16 --epochs 30 --dropout 0.4 --emb_dim 2948`
  (2948 = 1024 + 1582 + 342)

In read function (dataset.py): 
`features.append(u['cls'][0]+u['cls'][1]+u['cls'][2])`

To run program with text + visual: 
`!python run.py --dataset_name IEMOCAP --gnn_layers 4 --lr 0.0005 --batch_size 16 --epochs 30 --dropout 0.4 --emb_dim 1366`
  (1366 = 1024 + 342)

In read function (dataset.py):
`features.append(u['cls'][0]+u['cls'][2])`

Evaluate (required saving model first):
`!python evaluate.py --dataset_name IEMOCAP --state_dict_file /link/to/saved/model --gnn_layers 4 --lr 0.0005 --batch_size 16 --dropout 0.4 --emb_dim 2948`

To train model with curriculum learning: adding --curriculum and --bucket_number parameter.

Train model with all three modals and curriculum learning: 
`!python run.py --dataset_name IEMOCAP --gnn_layers 4 --lr 0.0005 --batch_size 16 --epochs 30 --dropout 0.4 --emb_dim 2948 --curriculum --bucket_number 12`


