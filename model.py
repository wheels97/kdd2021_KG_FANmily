#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import numpy as np
import os
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from dataloader import TestDataset
from collections import defaultdict

from ogb.linkproppred import Evaluator
from ogb.lsc import WikiKG90MEvaluator


class KGEModel(nn.Module):
    def __init__(self, model_name, nentity, nrelation, hidden_dim, gamma, evaluator,
                 double_entity_embedding=False, double_relation_embedding=False):
        super(KGEModel, self).__init__()
        self.model_name = model_name
        self.nentity = nentity
        self.nrelation = nrelation
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0
        
        self.gamma = nn.Parameter(
            torch.Tensor([gamma]), 
            requires_grad=False
        )
        
        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]), 
            requires_grad=False
        )
        
        self.entity_dim = hidden_dim*2 if double_entity_embedding else hidden_dim
        self.relation_dim = hidden_dim*2 if double_relation_embedding else hidden_dim
        
        self.entity_embedding = nn.Parameter(torch.zeros(nentity, self.entity_dim))
        nn.init.uniform_(
            tensor=self.entity_embedding, 
            a=-self.embedding_range.item(), 
            b=self.embedding_range.item()
        )
        
        self.relation_embedding = nn.Parameter(torch.zeros(nrelation, self.relation_dim))
        nn.init.uniform_(
            tensor=self.relation_embedding, 
            a=-self.embedding_range.item(), 
            b=self.embedding_range.item()
        )
        
        # Do not forget to modify this line when you add a new model in the "forward" function
        if model_name not in ['TransE', 'DistMult', 'ComplEx', 'RotatE']:
            raise ValueError('model %s not supported' % model_name)
            
        if model_name == 'RotatE' and (not double_entity_embedding or double_relation_embedding):
            raise ValueError('RotatE should use --double_entity_embedding')

        if model_name == 'ComplEx' and (not double_entity_embedding or not double_relation_embedding):
            raise ValueError('ComplEx should use --double_entity_embedding and --double_relation_embedding')

        self.evaluator = evaluator
        
    def forward(self, sample, mode):  #, mode='single'
        '''
        Forward function that calculate the score of a batch of triples.
        In the 'single' mode, sample is a batch of triple.
        In the 'head-batch' or 'tail-batch' mode, sample consists two part.
        The first part is usually the positive sample.
        And the second part is the entities in the negative samples.
        Because negative samples and positive samples usually share two elements 
        in their triple ((head, relation) or (relation, tail)).
        '''
#         device = "cuda:0"
#         sample = sample.to(device)
        if mode == "test" or mode == "valid":
            head_part, tail_part = sample
            batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)
            
            head = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=head_part[:, 0]
            ).unsqueeze(1)
            
            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=head_part[:, 1]
            ).unsqueeze(1)
            
            tail = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=tail_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)

        elif mode == "train":
            head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=sample[:,0]
            ).unsqueeze(1)

            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=sample[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=sample[:, 2]
            ).unsqueeze(1)


        model_func = {
            'TransE': self.TransE,
            'DistMult': self.DistMult,
            'ComplEx': self.ComplEx,
            'RotatE': self.RotatE,
        }

        if self.model_name in model_func:
            score = model_func[self.model_name](head, relation, tail)
        else:
            raise ValueError('model %s not supported' % self.model_name)

        return score
    
    def TransE(self, head, relation, tail):
        score = (head + relation) - tail
        score = self.gamma.item() - torch.norm(score, p=1, dim=2)
        return score

    def DistMult(self, head, relation, tail):
        score = (head * relation) * tail
        score = score.sum(dim = 2)
        return score

    def ComplEx(self, head, relation, tail):
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_relation, im_relation = torch.chunk(relation, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        re_score = re_head * re_relation - im_head * im_relation
        im_score = re_head * im_relation + im_head * re_relation
        score = re_score * re_tail + im_score * im_tail

        score = score.sum(dim=2)
        return score

    def RotatE(self, head, relation, tail):
        pi = 3.14159265358979323846
        
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        #Make phases of relations uniformly distributed in [-pi, pi]

        phase_relation = relation/(self.embedding_range.item()/pi)

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        re_score = re_head * re_relation - im_head * im_relation
        im_score = re_head * im_relation + im_head * re_relation
        re_score = re_score - re_tail
        im_score = im_score - im_tail

        score = torch.stack([re_score, im_score], dim = 0)
        score = score.norm(dim = 0)

        score = self.gamma.item() - score.sum(dim = 2)
        return score

    @staticmethod
    def train_step(model, optimizer, train_iterator, args):
        '''
        A single train step. Apply back-propation and return the loss
        '''

        model.train()
        optimizer.zero_grad()
        positive_sample, negative_sample, subsampling_weight = next(train_iterator)

        if args.cuda:
            device = "cuda:0"
            positive_sample = positive_sample.to(device)#cuda()
            negative_sample = negative_sample.to(device)#cuda()
            subsampling_weight = subsampling_weight.to(device)#cuda()

#         negative_score = model(torch.cat((positive_sample[:, [0, 1]], negative_sample), 1), "train")
        negative_score = model((positive_sample, negative_sample), "valid")
        if args.negative_adversarial_sampling:
            #In self-adversarial sampling, we do not apply back-propagation on the sampling weight
            negative_score = (F.softmax(negative_score * args.adversarial_temperature, dim = 1).detach() 
                              * F.logsigmoid(-negative_score)).sum(dim = 1)
        else:
            negative_score = F.logsigmoid(-negative_score).mean(dim = 1)

        positive_score = model(positive_sample, "train")
        positive_score = F.logsigmoid(positive_score).squeeze(dim = 1)
        logging.info("这里这里！训练的positive score为：{}".format(positive_score))

        if args.uni_weight:
            positive_sample_loss = - positive_score.mean()
            negative_sample_loss = - negative_score.mean()
        else:
            positive_sample_loss = - (subsampling_weight * positive_score).sum()/subsampling_weight.sum()
            negative_sample_loss = - (subsampling_weight * negative_score).sum()/subsampling_weight.sum()

        loss = (positive_sample_loss + negative_sample_loss)/2
        logging.info("训练损失为：{}".format(loss))
        
        if args.regularization != 0.0:
            #Use L3 regularization for ComplEx and DistMult
            regularization = args.regularization * (
                model.entity_embedding.norm(p = 3)**3 + 
                model.relation_embedding.norm(p = 3).norm(p = 3)**3
            )
            loss = loss + regularization
            regularization_log = {'regularization': regularization.item()}
        else:
            regularization_log = {}
        
        loss.backward()
        optimizer.step()
        log = {
            **regularization_log,
            'positive_sample_loss': positive_sample_loss.item(),
            'negative_sample_loss': negative_sample_loss.item(),
            'loss': loss.item()
        }

        return log
 
      
    @staticmethod
    def valid_step(model, test_triples, args,random_sampling=False):
        '''
        Evaluate the model on test datasets
        '''

        model.eval()

        test_dataloader_tail = DataLoader(
            TestDataset(
                test_triples,
                args,
                'tail-batch',
                random_sampling
            ),
            batch_size=args.test_batch_size,  # 32
            num_workers=max(1, args.cpu_num//2),#待验证
            collate_fn=TestDataset.collate_fn
        )

        test_dataset = test_dataloader_tail  # test_dataloader_head,

        test_logs = defaultdict(list)

        step = 0
        total_steps = len(test_dataset)
        MRR = []
        with torch.no_grad():
#             device = "cuda:0"
            for positive_sample, negative_sample in test_dataset:
                if args.cuda:
                    positive_sample = positive_sample.to(device)#cuda()
                    negative_sample = negative_sample.to(device)#cuda()

                batch_size = positive_sample.size(0)

#                 logging.info("加载出来的正样本为：{}".format(positive_sample))
#                 logging.info("加载出来的负样本为：{}".format(negative_sample))
#                 logging.info("加载出来的负样本维度为：{}".format(len(negative_sample)))

                score = model((positive_sample, negative_sample), "valid")

                test_score_total = score.numpy().argsort()[:,::-1][:,:10]
                test_correct = positive_sample[:,-1]

                t_pred_top10 = test_score_total.copy()
                t_correct_index = np.array(test_correct).astype(float)
                input_dict = {}

#                 logging.info("pred：{},correct:{}".format(t_pred_top10,t_correct_index))
                
                input_dict['h,r->t'] = {'t_pred_top10': t_pred_top10, 't_correct_index': t_correct_index}
                batch_results = model.evaluator.eval(input_dict)
                logging.info("step_{},mrr is:{}".format(step,batch_results['mrr']))
#                 logging.info("batch_results is {}".format(batch_results))

                MRR.append(batch_results['mrr'])

                if step % args.test_log_steps == 0:
                    logging.info('Evaluating the model... (%d/%d)' % (step, total_steps))
                step += 1

            res = np.mean(MRR)
            logging.info("total MRR is {}".format(res)) 
            print("total MRR is {}".format(res))
            
        return True
    
    @staticmethod
    def test_step(model, test_triples, args,random_sampling=False):
        '''
        Evaluate the model on test datasets
        '''

        model.eval()

        test_dataloader_tail = DataLoader(
            TestDataset(
                test_triples,
                args,
                'tail-batch',
                random_sampling
            ),
            batch_size=args.test_batch_size,  # 32
            num_workers=max(1, args.cpu_num//2),#待验证
            collate_fn=TestDataset.collate_fn
        )

        test_dataset = test_dataloader_tail  # test_dataloader_head,

        test_logs = defaultdict(list)

        step = 0
        total_steps = len(test_dataset)
        t_pred_top10 = np.zeros((1359303,10))
        
        with torch.no_grad():
#             device = "cuda:0"
            for positive_sample, negative_sample in test_dataset:
                if args.cuda:
                    positive_sample = positive_sample.to(device)#cuda()
                    negative_sample = negative_sample.to(device)#cuda()

                batch_size = positive_sample.size(0)

#                 logging.info("加载出来的正样本为：{}".format(positive_sample))
#                 logging.info("加载出来的负样本为：{}".format(negative_sample))
#                 logging.info("加载出来的负样本维度为：{}".format(len(negative_sample)))

                score = model((positive_sample, negative_sample), "valid")

                test_score_total = score.numpy().argsort()[:,::-1][:,:10]
                t_pred_top = test_score_total.copy()
#                 logging.info("pred：{}".format(t_pred_top))
                if len(t_pred_top) == args.test_batch_size:
                    t_pred_top10[step*args.test_batch_size:(step+1)*args.test_batch_size] = t_pred_top
                else:
                    t_pred_top10[step*args.test_batch_size:] = t_pred_top
                step += 1
                
        input_dict = {}
        input_dict['h,r->t'] = {'t_pred_top10': t_pred_top10}

        model.evaluator.save_test_submission(input_dict = input_dict, dir_path = 'result/')

        return True