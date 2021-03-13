#!/usr/bin/env python
# -*- coding: utf-8 -*-
import math
from torch import Tensor
from torch.nn import Module, GRU, Linear, Dropout
import torch
import torch.nn as nn
import numpy as np
import random
import operator
import torch
import torch.nn as nn
import torch.nn.functional as F
from queue import PriorityQueue

__author__ = 'Helin Wang -- Peking University'
__docformat__ = 'reStructuredText'
__all__ = ['Decoder','AttDecoder']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def sigmoid_decay(ite, epoch, rate=10.):
    ra = (float(epoch / 1.5) - float(ite))/float(epoch / rate)
    ra = 1.-(1. / (1.+ math.exp(ra)))
    return ra
def linear_decay(ite, epoch, rate=1.):
    ra = 1 - (1 - rate)*float(ite)/float(epoch)
    return ra
    
class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_out, decoder_hidden):
        att1 = self.encoder_att(encoder_out)
        att2 = self.decoder_att(decoder_hidden)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)
        alpha = self.softmax(att)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)
        return attention_weighted_encoding, alpha

class AttDecoder(Module):

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 att_dim: int,
                 maxlength: int,
                 nb_classes: int,
                 dropout_p: float) \
            -> None:
        """Decoder with attention.
        :param input_dim: Input features in the decoder.
        :type input_dim: int
        :param output_dim: Output features of the RNN.
        :type output_dim: int
        :param nb_classes: Number of output classes.
        :type nb_classes: int
        :param dropout_p: RNN dropout.
        :type dropout_p: float
        """
        super().__init__()

        self.dropout: Module = Dropout(p=dropout_p)
        self.attention = Attention(input_dim, output_dim, att_dim)
#         self.decode_step = nn.LSTMCell(2*input_dim, output_dim, bias=True)
        self.decode_step = nn.LSTMCell(input_dim, output_dim, bias=True)
        self.init_h = nn.Linear(input_dim, output_dim)
        self.init_c = nn.Linear(input_dim, output_dim)
        self.f_beta = nn.Linear(input_dim, output_dim)   # linear layer to create a sigmoid-activated gate
        self.sigmoid = nn.Sigmoid()
        self.maxlength = maxlength
        self.nb_classes=nb_classes
        self.classifier: Module = Linear(
            in_features=output_dim,
            out_features=nb_classes)
        self.word_emb = nn.Embedding(4367,output_dim)
        self.word_drop = nn.Dropout(p=0.5)
            
    def init_hidden_state(self, encoder_out):
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)
        c = self.init_c(mean_encoder_out)
        return h, c
    
    def sample(self, x):
        batch_size = x.size(0)
        encoder_dim = x.size(-1)
        x = x.transpose(1,2)
        num_frames = x.size(1)
        h, c = self.init_hidden_state(x)
        predictions = torch.zeros(batch_size, self.maxlength, self.nb_classes).to(device)
        word = self.word_emb(torch.zeros(batch_size).long().to(device))
        word = self.word_drop(word)

        for t in range(self.maxlength):
            attention_weighted_encoding, alpha = self.attention(x, h)
            gate = self.sigmoid(self.f_beta(h))
            attention_weighted_encoding = gate * attention_weighted_encoding
            h, c = self.decode_step(
                torch.cat([word,attention_weighted_encoding],dim=1),(h, c))
            preds = self.classifier(self.dropout(h))
            predictions[:, t, :] = preds
            word = self.word_emb(preds.max(1)[1])
                
        return predictions

    def trainer(self, x, y, epoch, max_epoch):
        batch_size = x.size(0)
        encoder_dim = x.size(-1)
        x = x.transpose(1,2)
        num_frames = x.size(1)
        h, c = self.init_hidden_state(x)
        predictions = torch.zeros(batch_size, self.maxlength, self.nb_classes).to(device)
        word = self.word_emb(torch.zeros(batch_size).long().to(device))
        word = self.word_drop(word)

        for t in range(y.shape[1]):
            teacher_focing_ratio = linear_decay(epoch, max_epoch, rate=.7)
            use_teacher_focing = random.random() < teacher_focing_ratio
            attention_weighted_encoding, alpha = self.attention(x, h)
            gate = self.sigmoid(self.f_beta(h))
            attention_weighted_encoding = gate * attention_weighted_encoding
            h, c = self.decode_step(word+attention_weighted_encoding,(h, c))
#             h, c = self.decode_step(torch.cat([word,attention_weighted_encoding],dim=1),(h, c))
            preds = self.classifier(self.dropout(h))
            predictions[:, t, :] = preds

            if use_teacher_focing and t < y.shape[1]-1:
                word = self.word_emb(y[:,t+1])
            else:
                word = self.word_emb(preds.max(1)[1])
            word = self.word_drop(word)

        return predictions
    



    def forward(self,
                x: Tensor,y: Tensor,epoch,max_epoch) \
            -> Tensor:
        """Forward pass of the decoder.
        :param x: Input tensor. Encoder output : (Batch_size, feature_maps, time_steps)
        :param y: Input tensor. Ground Truth : (Batch_size, length)
        :type x: torch.Tensor.
        :return: Output predictions.
        :rtype: torch.Tensor
        """
        if epoch == -1:
            #predictions=self.sample(x)
            #predictions=self.translate_greedy(x)
            predictions=self.translate_beam_search(x, beam_width=5, alpha=1., topk=1, number_required=5)
        else :
            predictions=self.trainer(x,y,epoch,max_epoch)
                
        return predictions

    def step(self, x, h, c, it):        
        attention_weighted_encoding, alpha = self.attention(x, h)
        gate = self.sigmoid(self.f_beta(h))
        attention_weighted_encoding = gate * attention_weighted_encoding
        
        word = self.word_emb(it)
        h, c = self.decode_step(word+attention_weighted_encoding,(h, c))
#         h, c = self.decode_step(
#                 torch.cat([word,attention_weighted_encoding],dim=1),(h, c))
        word_prob = self.classifier(self.dropout(h))
            
        return h, c, word_prob

    def translate_greedy(self, x):
        BOS = 0
        EOS = 9
        max_len = self.maxlength

        batch_size = x.size(0)
        
        # prepare feats, h and c
        x = x.transpose(1,2)
        h, c = self.init_hidden_state(x)
        it = x.new(batch_size).fill_(BOS).long().to(device)

        # to collect greedy results
        preds = []

        for t in range(max_len):
            h, c, word_prob = self.step(x, h, c, it)
            # word_prob: [batch_size, vocab_size]

            preds.append(word_prob)
            it = word_prob.max(1)[1]

        return torch.stack(preds, dim=1)


    def translate_beam_search(self, x, beam_width=2, alpha=1.15, topk=1, number_required=2):
        BOS = 0
        EOS = 9
        max_len = self.maxlength
        # beam_width = opt.get('beam_size', 5)
        # alpha = opt.get('beam_alpha', 1.0)
        # topk = opt.get('topk', 1)
        # number_required = opt.get('beam_candidate', 5)

        batch_size, device = x.size(0), x.device
        # prepare feats, h and c
        x = x.transpose(1,2)
        h, c = self.init_hidden_state(x)

        seq_preds = []
        # decoding goes sample by sample
        for idx in range(batch_size):
            encoder_output = x[idx, :].unsqueeze(0)

            endnodes = []
            
            # starting node -  hidden vector, previous node, word id, logp, length
            node = BeamSearchNode(
                hiddenstate=(h[idx, :].unsqueeze(0), c[idx, :].unsqueeze(0)), 
                previousNode=None, 
                wordId=torch.LongTensor([BOS]).to(device), 
                logProb=0, 
                selflp=0, 
                length=0, 
                alpha=alpha
                )
            nodes = PriorityQueue()
            tmp_nodes = PriorityQueue()
            # start the queue
            nodes.put((-node.eval(), node))

            # start beam search
            round = 0
            while True:
                while True:
                    # fetch the best node
                    if nodes.empty(): 
                        break
                    score, n = nodes.get()
                    if (n.wordid[0].cpu().item() == EOS and n.prevNode != None) or (n.leng >= max_len-1):
                        endnodes.append((score, n))
                        # if we reached maximum # of sentences required
                        if len(endnodes) >= number_required: 
                            break
                        else: 
                            continue

                    # decode for one step using decoder
                    now_h, now_c = n.h
                    it = n.wordid
                    
                    # foward one step
                    new_h, new_c, word_prob = self.step(encoder_output, now_h, now_c, it)
                    word_prob = F.log_softmax(word_prob, dim=-1)
                    # get beam_width candidates
                    log_prob, indexes = torch.topk(word_prob, beam_width)
                    
                    for new_k in range(beam_width):
                        decoded_t = torch.LongTensor([indexes[0][new_k]]).to(device)
                        log_p = log_prob[0][new_k].item()
                        node = BeamSearchNode((new_h, new_c), n, decoded_t, n.logp, log_p, n.leng + 1, alpha)
                        tmp_nodes.put((-node.eval(), node))



                if len(endnodes) >= number_required or tmp_nodes.empty(): 
                    break
                
                round += 1
                assert nodes.empty()

                # normally, tmp_nodes will have beam_width * beam_width candidates
                # we only keep the most possible beam_width candidates
                for i in range(beam_width):
                    nodes.put(tmp_nodes.get())
                tmp_nodes = PriorityQueue()
                assert tmp_nodes.empty()

            # choose nbest paths, back trace them
            if len(endnodes) < topk:
                for _ in range(topk - len(endnodes)):
                    endnodes.append(nodes.get())

            utterances = []
            count = 1
            for score, n in sorted(endnodes, key=operator.itemgetter(0)):
                if count > topk: break
                count += 1
                utterance = []

                utterance.append(n.wordid[0].cpu().item())
                # back trace
                while n.prevNode != None:
                    n = n.prevNode
                    if n.wordid[0].cpu().item() == BOS: break
                    utterance.append(n.wordid[0].cpu().item())

                # reverse
                utterance = utterance[::-1]
                for i in range(self.maxlength-len(utterance)):
                    utterance.append(EOS)
                utterances.append(utterance)
                
            seq_preds.append(utterances)
            
        seq_preds = np.array(seq_preds)
        seq_preds = torch.from_numpy(seq_preds[:,0,:][:,:,None]).to(device)
        seq_preds = torch.zeros(batch_size, self.maxlength, self.nb_classes).to(device).scatter_(2,seq_preds,1)

        return seq_preds


class BeamSearchNode(object):
    def __init__(self, hiddenstate, previousNode, wordId, logProb, selflp, length, alpha):
        '''
        :param hiddenstate:
        :param previousNode:
        :param wordId:
        :param logProb:
        :param length:
        '''
        # lstm
        assert isinstance(hiddenstate, tuple)
        self.h = (hiddenstate[0].clone(), hiddenstate[1].clone())
        
        self.prevNode = previousNode
        self.wordid = wordId
        self.logp = logProb + selflp
        self.selflp = selflp
        self.leng = length
        self.alpha = alpha

    def __lt__(self, other): 
        return (-self.eval()) < (-other.eval())

    def eval(self):
        #reward = 0
        # Add here a function for shaping a reward
        # + alpha * reward
        return (self.logp / float(self.leng)**self.alpha) if self.leng else -1e6 
    
    
# class AttDecoder(Module):

#     def __init__(self,
#                  input_dim: int,
#                  output_dim: int,
#                  att_dim: int,
#                  maxlength: int,
#                  nb_classes: int,
#                  dropout_p: float) \
#             -> None:
#         """Decoder with attention.

#         :param input_dim: Input features in the decoder.
#         :type input_dim: int
#         :param output_dim: Output features of the RNN.
#         :type output_dim: int
#         :param nb_classes: Number of output classes.
#         :type nb_classes: int
#         :param dropout_p: RNN dropout.
#         :type dropout_p: float
#         """
#         super().__init__()

#         self.dropout: Module = Dropout(p=dropout_p)
#         self.attention = Attention(input_dim, output_dim, att_dim)
#         self.decode_step = nn.LSTMCell(2*input_dim, output_dim, bias=True)
#         self.init_h = nn.Linear(input_dim, output_dim)
#         self.init_c = nn.Linear(input_dim, output_dim)
#         self.f_beta = nn.Linear(input_dim, output_dim)   # linear layer to create a sigmoid-activated gate
#         self.sigmoid = nn.Sigmoid()
#         self.maxlength = maxlength
#         self.nb_classes=nb_classes
#         self.classifier: Module = Linear(
#             in_features=output_dim,
#             out_features=nb_classes)
#         self.word_emb = nn.Embedding(4367,output_dim)
#         self.word_drop = nn.Dropout(p=0.5)
            
#     def init_hidden_state(self, encoder_out):
#         mean_encoder_out = encoder_out.mean(dim=1)
#         h = self.init_h(mean_encoder_out)
#         c = self.init_c(mean_encoder_out)
#         return h, c
    
#     def sample(self, x):
#         batch_size = x.size(0)
#         encoder_dim = x.size(-1)
#         x = x.transpose(1,2)
#         num_frames = x.size(1)
#         h, c = self.init_hidden_state(x)
#         predictions = torch.zeros(batch_size, self.maxlength, self.nb_classes).to(device)
#         word = self.word_emb(torch.zeros(batch_size).long().to(device))
#         word = self.word_drop(word)

#         for t in range(self.maxlength):
#             attention_weighted_encoding, alpha = self.attention(x, h)
#             gate = self.sigmoid(self.f_beta(h))
#             attention_weighted_encoding = gate * attention_weighted_encoding
#             h, c = self.decode_step(
#                 torch.cat([word,attention_weighted_encoding],dim=1),(h, c))
#             preds = self.classifier(self.dropout(h))
#             predictions[:, t, :] = preds
#             word = self.word_emb(preds.max(1)[1])
                
#         return predictions
    
#     def trainer(self, x, y, epoch, max_epoch):
#         batch_size = x.size(0)
#         encoder_dim = x.size(-1)
#         x = x.transpose(1,2)
#         num_frames = x.size(1)
#         h, c = self.init_hidden_state(x)
#         predictions = torch.zeros(batch_size, self.maxlength, self.nb_classes).to(device)
#         word = self.word_emb(torch.zeros(batch_size).long().to(device))
#         word = self.word_drop(word)

#         for t in range(y.shape[1]):
#             teacher_focing_ratio = linear_decay(epoch, max_epoch, rate=.7)
#             use_teacher_focing = random.random() < teacher_focing_ratio
#             attention_weighted_encoding, alpha = self.attention(x, h)
#             gate = self.sigmoid(self.f_beta(h))
#             attention_weighted_encoding = gate * attention_weighted_encoding
#             h, c = self.decode_step(torch.cat([word,attention_weighted_encoding],dim=1),(h, c))
#             preds = self.classifier(self.dropout(h))
#             predictions[:, t, :] = preds

#             if use_teacher_focing and t < y.shape[1]-1:
#                 word = self.word_emb(y[:,t+1])
#             else:
#                 word = self.word_emb(preds.max(1)[1])
#             word = self.word_drop(word)

#         return predictions
    
#     def forward(self,
#                 x: Tensor,y: Tensor,epoch,max_epoch) \
#             -> Tensor:
#         """Forward pass of the decoder.

#         :param x: Input tensor.
#         :type x: torch.Tensor
#         :return: Output predictions.
#         :rtype: torch.Tensor
#         """
#         if epoch == -1:
#             predictions=self.sample(x)
#         else :
#             predictions=self.trainer(x,y,epoch,max_epoch)
                
#         return predictions
    
class Decoder(Module):

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 nb_classes: int,
                 dropout_p: float) \
            -> None:
        """Decoder with no attention.

        :param input_dim: Input features in the decoder.
        :type input_dim: int
        :param output_dim: Output features of the RNN.
        :type output_dim: int
        :param nb_classes: Number of output classes.
        :type nb_classes: int
        :param dropout_p: RNN dropout.
        :type dropout_p: float
        """
        super().__init__()

        self.dropout: Module = Dropout(p=dropout_p)

        self.rnn: Module = GRU(
            input_size=input_dim,
            hidden_size=output_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=False)

        self.classifier: Module = Linear(
            in_features=output_dim,
            out_features=nb_classes)

    def forward(self,
                x: Tensor) \
            -> Tensor:
        """Forward pass of the decoder.

        :param x: Input tensor.
        :type x: torch.Tensor
        :return: Output predictions.
        :rtype: torch.Tensor
        """
        h = self.rnn(self.dropout(x))[0]
        return self.classifier(h)


# EOF
