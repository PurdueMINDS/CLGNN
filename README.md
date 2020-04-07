## Implementation of Collective Learning-GNN in PyTorch

Author: Mengyue Hang

### Requirements:

     PyTorch 1.2.0
     Python 3.6

     networkx==2.4
     numpy==1.17.3
     scipy==1.3.1

### Usage:

     for unlabeled test data: cd clgnn; python train_unlabeled.py -h
     for partially-labeled test data: cd clgnn; python train_labeled.py -h

     A full list of parameters is shown in help message with -h.

     We provide Cora as example dataset. You can put your own dataset in the data/ for testing.

     e.g. to test GCN and CL-GCN (with our collective learning framework) performance on      unlabeled test data:
     python train_unlabeled.py --model_choice gcn_rand --iterations 2

     to test tk and CL-tk performance on partially-labeled test data:
     python train_labeled.py --model_choice tk_rand --baseline
     python train_labeled.py --model_choice tk_rand
