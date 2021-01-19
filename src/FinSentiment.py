import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl

from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split, RandomSampler, SequentialSampler
from torch.nn.functional import cross_entropy

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.metrics.functional import f1
from pytorch_lightning.metrics.functional.classification import accuracy, multiclass_roc, auc, confusion_matrix

from transformers import AutoModel, AutoTokenizer, BertTokenizer, BertModel, DistilBertTokenizer, DistilBertModel
from transformers import AdamW, get_linear_schedule_with_warmup

import matplotlib.pyplot as plt
import seaborn as sn

from read_data import *


_, _, _, slope_df = read_data('boundary', dir_path='data/', existing_company_only=False, sample=None)
_, _, _, test_slope_df = read_test_data('boundary', dir_path='data/', existing_company_only=False, sample=None)
slope_df = pd.concat([slope_df,test_slope_df])

company_embedding_df = pd.read_csv('data/company_embedding_centered.csv')


class BertPooler(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class FinSentiment(pl.LightningModule):

    def __init__(self, pretrained_bert_name, incorrect_type, company_embedding_method='null', company_weight=1, lr=1e-5, hidden_dropout_prob=0.1, discriminate=True, burn_in_epochs=10):

        assert company_embedding_method in ['sum_embedding', 'sum_output', 'concat', 'null']
        
        super().__init__()

        print('[PROGRESS] Saving hyperparameters')
        self.save_hyperparameters()

        print('[PROGRESS] Initializing BERT model')
        self.bert = AutoModel.from_pretrained(pretrained_bert_name, return_dict=True)
        
        print('[PROGRESS] Initializing Company Embedding Layer')
        self.company_layer = nn.Embedding(num_embeddings=company_embedding_df.shape[0]+1, embedding_dim=768, padding_idx=0)
        #company_embedding_weight = torch.tensor(company_embedding_df.drop(columns=['ticker']).values)
        company_embedding_weight = torch.cat([torch.zeros((1, 768)), torch.tensor(company_embedding_df.drop(columns=['ticker']).values)])
        company_layer_state_dict = {'weight': company_embedding_weight}
        self.company_layer.load_state_dict(company_layer_state_dict)

        print('[PROGRESS] Initializing classifier')
        if company_embedding_method == 'concat':
            self.classifier = nn.Sequential(
                nn.Linear(768*2, 768*2),
                nn.ReLU(),
                nn.Dropout(hidden_dropout_prob),
                nn.Linear(768*2, 5)
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(768, 768),
                nn.ReLU(),
                nn.Linear(768, 5)
            )

        self.pooler = BertPooler(self.bert.config)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        
        #####  Others  #####
        self.incorrect_type = incorrect_type
        if self.incorrect_type == 'correct':
            self.weights = torch.tensor([21.7055,  6.5835,  3.0551,  2.6391, 10.4392])
        elif self.incorrect_type == 'inverse':
            self.weights = torch.tensor([17.9589,  5.0731,  3.4133,  2.7635, 10.8273])
        elif self.incorrect_type == 'boundary':
            self.weights = torch.tensor([7.3299, 5.8003, 3.9076, 3.3302, 7.4085]) 
        print('[PROGRESS] Done initialization')

        self.model_name = '_'.join([pretrained_bert_name, incorrect_type, company_embedding_method])
        self.incorrect_type = incorrect_type
        self.metrics_result = {'inverse':{},'correct':{}, 'boundary':{}}


    def forward(self, batch):
        input_ids, input_mask, label_ids, doc_ids = batch
        
        if self.hparams.company_embedding_method != 'sum_embedding':

            try:
                bert_output = self.bert(input_ids=input_ids, attention_mask=input_mask)['pooler_output']
            except:
                bert_output = self.bert(input_ids=input_ids, attention_mask=input_mask)['last_hidden_state']
                bert_output = self.pooler(bert_output)
            bert_output = self.dropout(bert_output)

            if self.hparams.company_embedding_method == 'null':
                cls_input = bert_output

            elif self.hparams.company_embedding_method == 'sum_output':

                ticker_id = get_ticker_id(doc_ids, company_embedding_df, slope_df).to(self.device)
                company_embeddings = self.company_layer(ticker_id).squeeze()
                cls_input = bert_output + company_embeddings*self.hparams.company_weight

            else: # self.hparams.company_embedding_method == 'concat'
                ticker_id = get_ticker_id(doc_ids, company_embedding_df, slope_df).to(self.device)
                company_embeddings = self.company_layer(ticker_id).squeeze()
                cls_input = torch.cat([bert_output, company_embeddings], dim=1)

        else:
            ticker_id = get_ticker_id(doc_ids, company_embedding_df, slope_df).to(self.device)
            company_embeddings = self.company_layer(ticker_id).squeeze()

            bert_embedding = self.bert.embeddings(input_ids)
            company_embeddings = company_embeddings.unsqueeze(1)
            full_embedding = bert_embedding + company_embeddings*self.hparams.company_weight
            
            try:
                bert_output = self.bert.transformer(
                    x=full_embedding,
                    attn_mask=input_mask,
                    head_mask=self.bert.get_head_mask(None, self.bert.config.num_hidden_layers)
                )
                bert_output = self.pooler(bert_output[0])
            except:
                extended_attention_mask = self.bert.get_extended_attention_mask(input_mask, input_ids.size(), self.device)

                encoder_outputs = self.bert.encoder(
                    full_embedding,
                    attention_mask=extended_attention_mask,
                    head_mask=self.bert.get_head_mask(None, self.bert.config.num_hidden_layers)
                )
                bert_output = self.bert.pooler(encoder_outputs[0])
            
            cls_input = self.dropout(bert_output)
        
        logits = self.classifier(cls_input)

        return logits


    def training_step(self, batch, batch_idx):

        assert self.trainer.max_epochs > self.hparams.burn_in_epochs

        if self.trainer.current_epoch < self.hparams.burn_in_epochs:
            for param in self.bert.parameters():
                param.requires_grad_(False)
            for param in self.pooler.parameters():
                param.requires_grad_(False)
            for param in self.company_layer.parameters():
                param.requires_grad_(False)
        else:
            for param in self.parameters():
                param.requires_grad_(True)

        input_ids, input_mask, label_ids, doc_ids = batch
        logits = self(batch)
        loss = cross_entropy(logits, label_ids, weight=self.weights.to(self.device))
        preds = logits.argmax(dim=1)
        
        batch_dict = {
            'loss': loss,
            'logits': logits,
            'preds': preds,
            'labels': label_ids
        }

        self.log('train_loss', loss)
        self.log('train_acc', accuracy(logits, label_ids))

        return batch_dict


    def training_epoch_end(self, outputs):
        
        #####  Print Progress  ###############
        if self.trainer.current_epoch < self.hparams.burn_in_epochs:
            print('[PROGRESS] Burning in classifier epoch %i'%self.current_epoch)

        #####  Calculate Metircs  ############
        avg_loss = torch.stack([b['loss'] for b in outputs]).mean()

        conf_mtx = confusion_matrix(torch.cat([b['preds'] for b in outputs]), 
                                    torch.cat([b['labels'] for b in outputs]),
                                    normalize=False,
                                    num_classes=5)
        avg_acc = torch.diag(conf_mtx).sum()/conf_mtx.sum()

        self.logger.experiment.add_scalar('train_avg_loss', avg_loss, self.current_epoch)
        self.logger.experiment.add_scalar('train_avg_acc', avg_acc, self.current_epoch)

        #print("[EPOCH]: %i, [TRAIN LOSS]: %.6f, [TRAIN ACCURACY]: %.3f" % (self.current_epoch, avg_loss, avg_acc))


    def validation_step(self, batch, batch_idx):
        input_ids, input_mask, label_ids, doc_ids = batch
        logits = self(batch)
        loss = cross_entropy(logits, label_ids, weight=self.weights.to(self.device))
        preds = logits.argmax(dim=1)

        batch_dict = {
            'loss': loss,
            'logits': logits,
            'preds': preds,
            'labels': label_ids
        }
        self.log('val_loss', loss)
        self.log('val_acc', accuracy(logits, label_ids))

        return batch_dict


    def validation_epoch_end(self, outputs):
        
        avg_loss = torch.stack([b['loss'] for b in outputs]).mean()

        conf_mtx = confusion_matrix(torch.cat([b['preds'] for b in outputs]), 
                                    torch.cat([b['labels'] for b in outputs]),
                                    normalize=False,
                                    num_classes=5)
        avg_acc = torch.diag(conf_mtx).sum()/conf_mtx.sum()

        self.logger.experiment.add_scalar('val_avg_loss', avg_loss, self.current_epoch)
        self.logger.experiment.add_scalar('val_avg_acc', avg_acc, self.current_epoch)

        #print("[EPOCH]: %i, [VAL LOSS]: %.6f, [VAL ACCURACY]: %.3f \n" % (self.current_epoch, avg_loss, avg_acc))
    

    def test_step(self, batch, batch_idx):
        input_ids, input_mask, label_ids, doc_ids = batch
        logits = self(batch)
        preds = logits.argmax(dim=1)

        batch_dict = {
            'logits': logits,
            'preds': preds,
            'labels': label_ids,
            'doc_ids': doc_ids
        }

        return batch_dict

    
    def test_epoch_end(self, outputs):

        if self.incorrect_type != 'boundary':
            #####  Confusion Matrix  #####
            conf_mtx = confusion_matrix(torch.cat([b['preds'] for b in outputs]), 
                                        torch.cat([b['labels'] for b in outputs]),
                                        normalize=False,
                                        num_classes=5)

            #####  Normalized Confusion Matrix  #####
            conf_mtx_normalized = confusion_matrix(torch.cat([b['preds'] for b in outputs]), 
                                        torch.cat([b['labels'] for b in outputs]),
                                        normalize=True,
                                        num_classes=5)

            #####  Weighted Confusion Matrix  #####
            conf_mtx_weighted = conf_mtx.clone()
            for c, w in enumerate(self.weights):
                conf_mtx_weighted[c, :] *= w

            #####  ACCURACY  #####
            accuracy = torch.diag(conf_mtx).sum()/conf_mtx.sum()
            accuracy_weighted = torch.diag(conf_mtx_weighted).sum()/conf_mtx_weighted.sum()

            #####  AUC_SCORE  #####
            roc_results = multiclass_roc(torch.cat([b['logits'] for b in outputs]), 
                                        torch.cat([b['labels'] for b in outputs]),
                                        num_classes=5)
            AUROC_str = ''
            AUROC_list = {}
            for cls, roc_cls in enumerate(roc_results):
              fpr, tpr, threshold = roc_cls
              self.logger.experiment.add_scalar(f'val_AUC[{cls}]', auc(fpr, tpr), self.current_epoch)
              AUROC_str += '\tAUC_SCORE[CLS %d]: \t%.4f\n'%(cls, auc(fpr, tpr))
              AUROC_list['AUC_SCORE[CLS %d]'% cls] = auc(fpr,tpr)
              
            #####  F1  #####
            f1_score = f1(torch.cat([b['preds'] for b in outputs]),
                        torch.cat([b['labels'] for b in outputs]),
                        num_classes=5)
            
            #####  Average Precision  #####
            # TO DO
            
            #####  PRINT RESULTS  #####
            print('='*100)
            print(f'[MODEL NAME]: {self.model_name} \t [INCORRECT TYPE]: {self.incorrect_type}')
            print('RESULTS:')
            print('\tAccuracy: \t\t%.4f'%accuracy)
            print('\tWeighted Accuracy: \t%.4f'%accuracy_weighted)
            print('\tF1 Score: \t\t%.4f'%f1_score)
            print(AUROC_str)
            
            self.metrics_result[self.incorrect_type][self.model_name] = {'Accuracy':round(float(accuracy),4),'Weighted Accuracy':round(float(accuracy_weighted),4),'F1_score':round(float(f1_score),4)}
            for key,val in AUROC_list.items():
                self.metrics_result[self.incorrect_type][self.model_name].update({key:round(float(val),4)})
            print('Confusion Matrix')
            fig, ax = plt.subplots(figsize=(4,4))
            sn.heatmap(conf_mtx.cpu(), annot=True, cbar=False, annot_kws={"size": 15}, fmt='g', cmap='mako')
            plt.show()
            fig, ax = plt.subplots(figsize=(4,4))
            sn.heatmap(conf_mtx_normalized.cpu(), annot=True, cbar=False, annot_kws={"size": 12}, fmt='.2f', cmap='mako')
            plt.show()
            print('='*100)
        
        
        else:
            tol_correct = 0
            tol_samples = 0
            tol_drop = 0 
            for batch in outputs:
                preds = batch['preds']
                labels = batch['labels']
                slope_id = batch['doc_ids']
            ##### Change lizhong's code #### 
                for idx,slop_idx in enumerate(slope_id):
                  agree_by_user = bool(slope_df[slope_df['slope_id']== slop_idx.item() ]['sentiment_correct'].values[0])
                  possible_classes = slope_df[slope_df['slope_id']==slop_idx.item()]['label_from_score'].values[0]

                  pred_class = preds[idx]
                  # difference between pred and true label
                  diff = torch.abs(pred_class - possible_classes) 

                  # if correct label 
                  if agree_by_user: # True 
                     if diff == 0:
                       # correct prediction
                       tol_correct += 1
                       tol_samples += 1
                     elif diff == 1:
                       # discard
                       tol_drop += 1
                     else:
                       # wrong prediction
                       tol_samples += 1
                  # if incorrect label
                  else: # False 
                      if diff == 0:
                        # wrong
                        tol_samples += 1
                      elif diff == 1:
                        # discard
                        tol_drop += 1
                      else:
                        # Correct 
                        tol_correct += 1
                        tol_samples += 1

            
            boundary_accuracy = round(tol_correct / tol_samples,4)
            self.metrics_result[self.incorrect_type][self.model_name] = {}
            self.metrics_result[self.incorrect_type][self.model_name]['boundary_acc'] = boundary_accuracy
            self.metrics_result[self.incorrect_type][self.model_name]['total_drop_sample'] = tol_drop
            print('='*100)
            print(f'[MODEL NAME]: {self.model_name} \t [INCORRECT TYPE]: {self.incorrect_type}')
            print('\tBoundary Accuracy: \t\t%.4f'%boundary_accuracy)
            print('\tDrop Total Sample: \t\t%.4f'%tol_drop)

    
    def configure_optimizers(self):
        
        ###############  Optimizer  ##############################
        lr = self.hparams.lr
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        dft_rate = 1.2

        try:
            num_layers = len(self.bert.encoder.layer)
            distil_bert = False
        except:
            num_layers = len(self.bert.transformer.layer)
            distil_bert = True

        ################ Distil BERT ################
        if distil_bert:
            if self.hparams.discriminate:
                encoder_params = []
                for i in range(num_layers):
                    encoder_decay = {
                        'params': [p for n, p in list(self.bert.transformer.layer[i].named_parameters()) if
                                not any(nd in n for nd in no_decay)],
                        'weight_decay': 0.01,
                        'lr': lr / (dft_rate ** (num_layers - i))}
                    encoder_nodecay = {
                        'params': [p for n, p in list(self.bert.transformer.layer[i].named_parameters()) if
                                any(nd in n for nd in no_decay)],
                        'weight_decay': 0.0,
                        'lr': lr / (dft_rate ** (num_layers - i))}
                    encoder_params.append(encoder_decay)
                    encoder_params.append(encoder_nodecay)

                optimizer_grouped_parameters = [
                    {'params': [p for n, p in list(self.bert.embeddings.named_parameters()) if
                                not any(nd in n for nd in no_decay)],
                    'weight_decay': 0.01,
                    'lr': lr / (dft_rate ** (num_layers+1))},
                    {'params': [p for n, p in list(self.bert.embeddings.named_parameters()) if
                                any(nd in n for nd in no_decay)],
                    'weight_decay': 0.0,
                    'lr': lr / (dft_rate ** (num_layers+1))},
                    {'params': [p for n, p in list(self.pooler.named_parameters()) if
                                not any(nd in n for nd in no_decay)],
                    'weight_decay': 0.01,
                    'lr': lr},
                    {'params': [p for n, p in list(self.pooler.named_parameters()) if
                                any(nd in n for nd in no_decay)],
                    'weight_decay': 0.0,
                    'lr': lr},
                    {'params': [p for n, p in list(self.classifier.named_parameters()) if
                                not any(nd in n for nd in no_decay)],
                    'weight_decay': 0.01,
                    'lr': lr},
                    {'params': [p for n, p in list(self.classifier.named_parameters()) if any(nd in n for nd in no_decay)],
                    'weight_decay': 0.0,
                    'lr': lr}]

                optimizer_grouped_parameters.extend(encoder_params)

            else:
                param_optimizer = list(self.named_parameters())

                optimizer_grouped_parameters = [
                    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
                    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
                ]

        ################ BERT #################
        else:
            if self.hparams.discriminate:
                # apply the discriminative fine-tuning. discrimination rate is governed by dft_rate.
                encoder_params = []
                for i in range(num_layers):
                    encoder_decay = {
                        'params': [p for n, p in list(self.bert.encoder.layer[i].named_parameters()) if
                                not any(nd in n for nd in no_decay)],
                        'weight_decay': 0.01,
                        'lr': lr / (dft_rate ** (num_layers - i))}
                    encoder_nodecay = {
                        'params': [p for n, p in list(self.bert.encoder.layer[i].named_parameters()) if
                                any(nd in n for nd in no_decay)],
                        'weight_decay': 0.0,
                        'lr': lr / (dft_rate ** (num_layers - i))}
                    encoder_params.append(encoder_decay)
                    encoder_params.append(encoder_nodecay)

                optimizer_grouped_parameters = [
                    {'params': [p for n, p in list(self.bert.embeddings.named_parameters()) if
                                not any(nd in n for nd in no_decay)],
                    'weight_decay': 0.01,
                    'lr': lr / (dft_rate ** (num_layers+1))},
                    {'params': [p for n, p in list(self.bert.embeddings.named_parameters()) if
                                any(nd in n for nd in no_decay)],
                    'weight_decay': 0.0,
                    'lr': lr / (dft_rate ** (num_layers+1))},
                    {'params': [p for n, p in list(self.bert.pooler.named_parameters()) if
                                not any(nd in n for nd in no_decay)],
                    'weight_decay': 0.01,
                    'lr': lr},
                    {'params': [p for n, p in list(self.bert.pooler.named_parameters()) if
                                any(nd in n for nd in no_decay)],
                    'weight_decay': 0.0,
                    'lr': lr},
                    {'params': [p for n, p in list(self.classifier.named_parameters()) if
                                not any(nd in n for nd in no_decay)],
                    'weight_decay': 0.01,
                    'lr': lr},
                    {'params': [p for n, p in list(self.classifier.named_parameters()) if any(nd in n for nd in no_decay)],
                    'weight_decay': 0.0,
                    'lr': lr}]

                optimizer_grouped_parameters.extend(encoder_params)

            else:
                param_optimizer = list(self.named_parameters())

                optimizer_grouped_parameters = [
                    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
                    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
                ]

        company_prameters = [{'params': self.company_layer.parameters(), 'weight_decay': 0.01, 'lr': lr}]
        optimizer_grouped_parameters.extend(company_prameters)   

        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.lr, correct_bias=False)

        ###############  Scheduler  ##############################
        # num_steps_per_epoch = len(self.train_dataloader())
        # num_total_epochs = self.trainer.max_epochs
        # num_warmup_epochs = self.hparams.burn_in_epochs

        # num_training_steps = num_steps_per_epoch * num_total_epochs
        # num_warmup_steps = num_steps_per_epoch * num_warmup_epochs

        # scheduler = get_linear_schedule_with_warmup(optimizer, 
        #                                             num_warmup_steps=num_warmup_steps, 
        #                                             num_training_steps=num_training_steps)

        return [optimizer]#, [scheduler]



######################################################################################################################################
######################################################################################################################################
######################################################################################################################################

def tokenize_and_dataloader(X, y, tokenizer, doc_ids, batch_size, num_workers=0, random=True):

    input_ids = []
    attention_masks = []

    for doc in X:
        encoded_dict = tokenizer.encode_plus(
                            doc,                                    # Sentence to encode.
                            add_special_tokens = True,              # Add '[CLS]' and '[SEP]'
                            max_length = 32,     # Pad & truncate all sentences.
                            padding='max_length',
                            return_attention_mask = True,           # Construct attn. masks.
                            return_tensors = 'pt',                  # Return pytorch tensors.
                            truncation = True)
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
    
    # Construct Dataset
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(y)
    doc_ids = torch.tensor(doc_ids)
    dataset = TensorDataset(input_ids, attention_masks, labels, doc_ids)
    
    # Build DataLoader
    if random:
        dataloader = DataLoader(dataset, sampler=RandomSampler(dataset), batch_size=batch_size, num_workers=num_workers)
    else:
        dataloader = DataLoader(dataset, sampler=SequentialSampler(dataset), batch_size=batch_size, num_workers=num_workers)
        
    return dataloader


def get_ticker_id(slope_ids, company_embedding_df, slope_df):
    ticker_id_list = []
    for slope_id in slope_ids:
        #print(slope_id)
        ticker = slope_df[slope_df['slope_id']==slope_id.item()]['ticker'].values[0]
        ticker_id = torch.tensor(company_embedding_df[company_embedding_df['ticker']==ticker].index.values)
        if ticker_id.numel() == 0:
            ticker_id = torch.tensor([0])
        else:
            ticker_id += 1
        ticker_id_list.append(ticker_id)
    return  torch.stack(ticker_id_list)


def get_company_embedding(slope_ids, company_embedding_df, slope_df):
    
    embedding_list = []
    for slope_id in slope_ids:
        #print(slope_id)
        ticker = slope_df[slope_df['slope_id']==slope_id.item()]['ticker'].values[0]
        try:  # if ticker exists in the company list
            embedding = company_embedding_df[company_embedding_df['ticker']==ticker].values[0][1:].astype('float')
        except: # if ticker does not exsit, then embedding is zero since mean is zero
            embedding = np.zeros(768)
        embedding_list.append(torch.tensor(embedding))
    return  torch.stack(embedding_list)


def evaluate_model(model, tokenizer, trainer, existing_company_only, dir_path='data/',batch_size=16, num_workers=4):
    
    (train_ids,X_train,y_train), (val_ids,X_val,y_val), (test_ids,X_test,y_test), _ = read_data('correct', dir_path=dir_path, existing_company_only=existing_company_only, sample=None)
    correct_test_dataloader = tokenize_and_dataloader(X_test, y_test, tokenizer, test_ids, batch_size, num_workers, random=False)

    (train_ids,X_train,y_train), (val_ids,X_val,y_val), (test_ids,X_test,y_test), _ = read_data('inverse', dir_path=dir_path, existing_company_only=existing_company_only, sample=None)
    inverse_test_dataloader = tokenize_and_dataloader(X_test, y_test, tokenizer, test_ids, batch_size, num_workers, random=False)

    (train_ids,X_train,y_train), (val_ids,X_val,y_val), (test_ids,X_test,y_test), _ = read_data('boundary', dir_path=dir_path, existing_company_only=existing_company_only, sample=None)
    boundary_test_dataloader = tokenize_and_dataloader(X_test, y_test, tokenizer, test_ids, batch_size, num_workers, random=False)

    model.incorrect_type = 'correct'
    trainer.test(model, correct_test_dataloader)

    model.incorrect_type = 'inverse'
    trainer.test(model, inverse_test_dataloader)

    model.incorrect_type = 'boundary'
    trainer.test(model, boundary_test_dataloader)

