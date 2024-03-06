import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pickle, pandas as pd
import json
import numpy as np
import random
from pandas import DataFrame

import cl


class IEMOCAPDataset(Dataset):

    def __init__(self, dataset_name = 'IEMOCAP', split = 'train', speaker_vocab=None, label_vocab=None, args = None, tokenizer = None, babystep_index = None):
        self.speaker_vocab = speaker_vocab
        self.label_vocab = label_vocab
        self.args = args
        self.data = self.read(dataset_name, split, tokenizer)
        if args.curriculum and split == 'train':
            self.data = self.babystep(self.getbuckets(self.data, args.bucket_number), babystep_index)
        print(len(self.data))

        self.len = len(self.data)

    def read(self, dataset_name, split, tokenizer):
        with open('../data/%s/%s_data_roberta_mm.json.feature'%(dataset_name, split), encoding='utf-8') as f:
            raw_data = json.load(f)

        # process dialogue
        dialogs = []
        # raw_data = sorted(raw_data, key=lambda x:len(x))
        for d in raw_data:
            # if len(d) < 5 or len(d) > 6:
            #     continue
            utterances = []
            labels = []
            speakers = []
            features = []
            for i,u in enumerate(d):
                utterances.append(u['text'])
                labels.append(self.label_vocab['stoi'][u['label']] if 'label' in u.keys() else -1)
                speakers.append(self.speaker_vocab['stoi'][u['speaker']])
                features.append(u['cls'][0] + u['cls'][1]+u['cls'][2])
                #different modalities
                #features.append(u['cls'][0])
                #features.append(u['cls'][1])
                #features.append(u['cls'][2])
                #features.append(u['cls'][0] + u['cls'][1])
                #features.append(u['cls'][0] + u['cls'][2])
                #features.append(u['cls'][1]+u['cls'][2])
            dialog = cl.Dialog(utterances, labels, speakers, features)
            # dialogs.append({
            #     'utterances': utterances,
            #     'labels': labels,
            #     'speakers':speakers,
            #     'features': features
            # })
            dialogs.append(dialog)
        if self.args.curriculum and split == 'train':
          totalut = 0
          totalshift = 0
          totalspeaker = 0
          for i in range(0, len(dialogs)):
            totalut += dialogs[i].numberofutterances
            totalshift += dialogs[i].numberofemotionshifts
            totalspeaker += dialogs[i].numberofspeakers
        # random.shuffle(dialogs)
          dialogs.sort(key= lambda dialog: dialog.difficulty)
        # if (split == 'train'):
        #     num_buckets = 8
        #     bucket_length = (len(dialogs) + num_buckets - 1) // num_buckets
        #     buckets = [dialogs[i:i + bucket_length] for i in range(0, len(dialogs), bucket_length)]
        # print('')
        else:
          random.shuffle(dialogs)
        return dialogs

    def getbuckets(self, dialogs, num_buckets):
        buckets = []
        bucket_length = (len(dialogs) + num_buckets - 1) // num_buckets
        buckets = [dialogs[i:i + bucket_length] for i in range(0, len(dialogs), bucket_length)]
        print('bucket')
        print(len(buckets))
        return buckets
    #parameter for curriculum learning
    def babystep(self, buckets, index):
        data = []
        for i in range(0, index):
            data+= buckets[i];
        return data
    def __getitem__(self, index):
        '''
        :param index:
        :return:
            feature,
            label
            speaker
            length
            text
        '''
        return torch.FloatTensor(self.data[index]['features']), \
               torch.LongTensor(self.data[index]['labels']),\
               self.data[index]['speakers'], \
               len(self.data[index]['labels']), \
               self.data[index]['utterances']

    def __len__(self):
        return self.len

    def get_adj(self, speakers, max_dialog_len):
        '''
        get adj matrix
        :param speakers:  (B, N)
        :param max_dialog_len:
        :return:
            adj: (B, N, N) adj[:,i,:] means the direct predecessors of node i
        '''
        adj = []
        for speaker in speakers:
            a = torch.zeros(max_dialog_len, max_dialog_len)
            for i,s in enumerate(speaker):
                get_local_pred = False
                get_global_pred = False
                for j in range(i - 1, -1, -1):
                    if speaker[j] == s and not get_local_pred:
                        get_local_pred = True
                        a[i,j] = 1
                    elif speaker[j] != s and not get_global_pred:
                        get_global_pred = True
                        a[i,j] = 1
                    if get_global_pred and get_local_pred:
                        break
            adj.append(a)
        return torch.stack(adj)

    def get_adj_v1(self, speakers, max_dialog_len):
        '''
        get adj matrix
        :param speakers:  (B, N)
        :param max_dialog_len:
        :return:
            adj: (B, N, N) adj[:,i,:] means the direct predecessors of node i
        '''
        adj = []
        for speaker in speakers:
            a = torch.zeros(max_dialog_len, max_dialog_len)
            for i,s in enumerate(speaker):
                cnt = 0
                for j in range(i - 1, -1, -1):             
                    a[i,j] = 1
                    if speaker[j] == s:
                        cnt += 1
                        if cnt==self.args.windowp:
                            break
            adj.append(a)
        return torch.stack(adj)

    def get_s_mask(self, speakers, max_dialog_len):
        '''
        :param speakers:
        :param max_dialog_len:
        :return:
         s_mask: (B, N, N) s_mask[:,i,:] means the speaker informations for predecessors of node i, where 1 denotes the same speaker, 0 denotes the different speaker
         s_mask_onehot (B, N, N, 2) onehot emcoding of s_mask
        '''
        s_mask = []
        s_mask_onehot = []
        for speaker in speakers:
            s = torch.zeros(max_dialog_len, max_dialog_len, dtype = torch.long)
            s_onehot = torch.zeros(max_dialog_len, max_dialog_len, 2)
            for i in range(len(speaker)):
                for j in range(len(speaker)):
                    if speaker[i] == speaker[j]:
                        s[i,j] = 1
                        s_onehot[i,j,1] = 1
                    else:
                        s_onehot[i,j,0] = 1

            s_mask.append(s)
            s_mask_onehot.append(s_onehot)
        return torch.stack(s_mask), torch.stack(s_mask_onehot)

    def collate_fn(self, data):
        '''
        :param data:
            features, labels, speakers, length, utterances
        :return:
            features: (B, N, D) padded
            labels: (B, N) padded
            adj: (B, N, N) adj[:,i,:] means the direct predecessors of node i
            s_mask: (B, N, N) s_mask[:,i,:] means the speaker informations for predecessors of node i, where 1 denotes the same speaker, 0 denotes the different speaker
            lengths: (B, )
            utterances:  not a tensor
        '''
        max_dialog_len = max([d[3] for d in data])
        feaures = pad_sequence([d[0] for d in data], batch_first = True) # (B, N, D)
        labels = pad_sequence([d[1] for d in data], batch_first = True, padding_value = -1) # (B, N )
        adj = self.get_adj_v1([d[2] for d in data], max_dialog_len)
        s_mask, s_mask_onehot = self.get_s_mask([d[2] for d in data], max_dialog_len)
        lengths = torch.LongTensor([d[3] for d in data])
        speakers = pad_sequence([torch.LongTensor(d[2]) for d in data], batch_first = True, padding_value = -1)
        utterances = [d[4] for d in data]

        return feaures, labels, adj,s_mask, s_mask_onehot,lengths, speakers, utterances
