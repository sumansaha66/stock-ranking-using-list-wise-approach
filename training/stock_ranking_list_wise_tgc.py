
# This version will implement the temporal graph convolution
# It has the option to use point pair loss (reg_rank) or list-wise loss
# It will not implement Node2Vec

""" Description of variables"""

'''inner_prod -> in_pro'''
# Determine whether to use implicit modelling or explicit modelling as per Feng et al. paper.
# If True, explicit modeling. If false, implicit modeling

'''embedding -> emb_batch -> Feature'''
# Pre-trained sequential embedding (e^t as per Feng et al.). Shape: (num_company, num_days, embedding_shape).

'''mask_data -> mask_batch -> mask'''
# mask_data: mask for time series data, all 1 or 0, (num_company,num_days) shape, numpy array
# mask_data is to deal with missing time series data. It will be 0 if there
# is any missing data on a day for a company

'''rel_encoding'''
# rel_encoding: True relations not masked. (num_companies, num_companies, rel_types)
# it can be either 0 or 1

'''rel_weight'''
# This is the output of a fully connected layer whose input is rel_encoding
# It is the feature importance of eq. 12 as per Feng et al.

'''rel_mask'''
# rel_mask: mask for relation (num_company, num_company).
# If there is a relation the mask will be 0, otherwise, a large negative number there -1e9

'''weight'''
# Relation strength function g of the Feng et al. paper according to eq. 12 or eq. 13

'''weight_masked'''
# Softmax function output. This is the normalized relation strength.

'''outputs_proped'''
# Time-aware embedding propagation. e^t^bar as per eq. 11 of Feng et al.

'''outputs_concated'''
# Concatenated sequential embedding (e^t) and relational embedding (e^t^bar). [e^t, e^t^bar] as per Feng et al.

'''prediction'''
# predicted return of the prediction layer


# Import packages
import argparse
import copy
import numpy as np
import os
import pandas as pd
import random
from time import time


import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense

from load_data import load_EOD_data, load_relation_data
from evaluator import evaluate, make_df_loss
from loss_functions_tgc import reg_loss_tgc, rank_loss_tgc, listnet_loss

# Set up random seeds
seed = 123456789
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# Function for initialising arrays
def prediction_ground_truth_mask_initializer(dim1,dim2):
    pred_array = np.zeros([dim1, dim2],dtype=float)
    gt_array=np.zeros([dim1, dim2],dtype=float)
    mask_array=np.zeros([dim1, dim2],dtype=float)
    return pred_array, gt_array, mask_array
    

class MyModel(Model):
  def __init__(self, nCom, rel_mask, inner_prod, flat,rel_encoding,units = 0):
    super(MyModel, self).__init__()
    self.rel_mask = rel_mask
    self.inner_prod = inner_prod
    self.flat = flat
    self.all_one = tf.ones([nCom, 1], dtype=tf.float32)
    self.rel_encoding=rel_encoding.astype('float32')
    self.relation_layer=Dense(1,activation=tf.keras.layers.LeakyReLU(),name='relation_layer')
    self.head_weight_layer = Dense(1,activation=tf.keras.layers.LeakyReLU())
    self.tail_weight_layer = Dense(1,activation=tf.keras.layers.LeakyReLU())
    self.prediction_layer = Dense(1,activation=tf.keras.layers.LeakyReLU(), 
                                  kernel_initializer='glorot_uniform',name='prediction_layer')
    if self.flat:
        print('one more hidden layer')
        self.hidden_layer =  Dense(units, activation=tf.keras.layers.LeakyReLU(),
                                   kernel_initializer='glorot_uniform')
    else:
        self.hidden_layer = None

    
  def call(self, Feature):
      rel_weight = self.relation_layer(self.rel_encoding)[:,:,-1] # relation importance of eq. 12 in the Feng et al. paper
      if self.inner_prod:
          '''explicit modeling'''
          # Feature is sequential embedding
          inner_weight = tf.matmul(Feature, Feature, transpose_b=True) # similarity of eq. 12 in the Feng et al. paper
          weight = tf.multiply(inner_weight, rel_weight) # relation strength function g of the paper eq. 12 in the Feng et al. paper
      else:
          '''implicit modeling'''
          head_weight = self.head_weight_layer(Feature)
          tail_weight = self.tail_weight_layer(Feature)
          # using mask to make sure only masked values are used
          weight = tf.add(
                  tf.add(
                          tf.matmul(head_weight, self.all_one, transpose_b=True),
                          tf.matmul(self.all_one, tail_weight, transpose_b=True)
                          ), rel_weight
                  ) # relation strength function g of eq. 13 in the Feng et al. paper
      
      # Normalize the relation strength in between 0 to 1
      # Addition of rel_mask will make all the values to 0 where there was no predefined relation
      weight_masked = tf.keras.activations.softmax(tf.add(self.rel_mask, weight), axis=0)
      outputs_proped = tf.matmul(weight_masked, Feature) # e^t^bar as per eq. 11 in the Feng et al. paper
      if self.flat:
          outputs_concated = self.hidden_layer(
              tf.concat([Feature, outputs_proped], axis=1)) # Concatenated sequential embedding (e^t) and relational embedding (e^t^bar). [e^t, e^t^bar]
      else:
          outputs_concated = tf.concat([Feature, outputs_proped], axis=1) # Concatenated sequential embedding (e^t) and relational embedding (e^t^bar). [e^t, e^t^bar]
      prediction = self.prediction_layer(outputs_concated) # predicted return of the prediction layer
      print('prediction layer input shape: ',outputs_concated.shape)
      print('prediction layer output shape: ',prediction.shape)
      return rel_weight,prediction

class ReRaLSTM:
    def __init__(self, data_path, market_name, tickers_fname, relation_name,
                 emb_fname, parameters, depth, loss_name, steps=1, epochs=50, batch_size=None, 
                 flat=False, gpu=False, in_pro=False):
    
        seed = 123456789
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
        self.data_path = data_path
        self.market_name = market_name
        self.tickers_fname = tickers_fname
        self.relation_name = relation_name
        self.df_loss=pd.DataFrame() # to store the loss values and evaluation metrices
        self.depth=depth # required to calculate nrbo
        self.loss_name=loss_name # point pair loss or list wise loss
        self.parameters = copy.copy(parameters)
        self.steps = steps
        self.epochs = epochs
        self.flat = flat
        self.inner_prod = in_pro
        self.valid_index = 756
        self.test_index = 1008
        self.fea_dim = 5
        self.gpu = gpu
        
        # load data
        self.tickers = np.genfromtxt(os.path.join(data_path, '..', tickers_fname),
                                         dtype=str, delimiter='\t', skip_header=False)
        
        print('#tickers selected:', len(self.tickers))

        # mask_data: mask for time series data, all 1, (num_company,num_days) shape, numpy array
        # mask_data is to deal with missing time series data. It will be 0 if there
        # is any missing data on a day for a company
        # price_data contains normalized price of all days for all stocks. (num_company,num_days)
        # gt_data is ground truth or actual daily return. shape (num_company,num_days)
        
        self.eod_data, self.mask_data, self.gt_data, self.price_data = load_EOD_data(data_path, market_name, self.tickers, steps)
        print('price_data shape: ', self.price_data.shape)
        print('gt_data shape ', self.gt_data.shape)        
        
        # relation data
        rname_tail = {'sector_industry': '_industry_relation.npy',
                      'wikidata': '_wiki_relation.npy'}
        # rel_encoding: True relations not masked. (num_companies, num_companies, rel_types)
        # rel_mask: mask for relation (num_company, num_company).
        # If there is a relation the mask will be 0, otherwise, a large negative number there -1e9
        if self.relation_name in ['sector_industry','wikidata']:
            self.rel_encoding, self.rel_mask = load_relation_data(
                    os.path.join(self.data_path,'..', 'relation', self.relation_name,
                                 self.market_name + rname_tail[self.relation_name])
                    )
            # The next part is only relevant if the number of nodes is less than the total
            # number of available nodes in the original study. I am assuming that the nodes are 
            # in the same order in the adjacency matrix as in the ticker file
            self.rel_encoding=self.rel_encoding[:self.gt_data.shape[0],:self.gt_data.shape[0],:]
            self.rel_mask=self.rel_mask[:self.gt_data.shape[0],:self.gt_data.shape[0]]
                    
        self.rel_mask = self.rel_mask.astype('float32')
        print('relation encoding shape:', self.rel_encoding.shape)
        print('relation mask shape:', self.rel_mask.shape)
                
        # trained pre-trained sequential embedding (num_company, num_days, embedding dimension).
        # The last dimension is U or embedding shape
        self.embedding = np.load(
            os.path.join(data_path, '..', 'pretrain', emb_fname))
        print('embedding shape:', self.embedding.shape)
        
        # The next part is only relevant if the number of nodes is less than the total
        # number of available nodes in the original study. I am assuming that the nodes are 
        # in the same order in the adjacency matrix as in the ticker file'''
        self.embedding=self.embedding[:self.gt_data.shape[0],:,:] # sequential embedding
        
        print('embedding shape:', self.embedding.shape)
        if batch_size is None:
            self.batch_size = len(self.tickers)
        else:
            self.batch_size = batch_size

        self.trade_dates = self.mask_data.shape[1]
        self.numCompany = self.rel_mask.shape[0]
        
        self.model = MyModel(self.numCompany, self.rel_mask, 
                             self.inner_prod, self.flat,self.rel_encoding,self.parameters['unit'])


    def get_batch(self, offset=None):
        if offset is None:
            offset = random.randrange(0, self.valid_index)
        seq_len = self.parameters['seq']
        mask_batch = self.mask_data[:, offset: offset + seq_len + self.steps]
        mask_batch = np.min(mask_batch, axis=1)
        # ret sequential_embedding, mask for time series data, price data and ground truth data
        return self.embedding[:, offset, :], np.expand_dims(mask_batch, axis=1), np.expand_dims(
                self.price_data[:, offset + seq_len - 1], axis=1), np.expand_dims(
                        self.gt_data[:, offset + seq_len + self.steps - 1], axis=1)
    def train(self):
        seed = 123456789
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
        if self.gpu == True:
            device_name = '/gpu:0'
        else:
            device_name = '/cpu:0'
        print('device name:', device_name) 

        optimizer = tf.keras.optimizers.Adam()
 
        @tf.function
        # Feature: sequential_embedding, mask for time series data, price data and ground truth data
        def train_step(Feature, base_price, ground_truth, mask):
          with tf.GradientTape() as tape:
            # training=True is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            rel_weight, prediction = self.model(Feature, training=True)
            return_ratio = tf.divide(tf.subtract(prediction, base_price), base_price) #(num_company,1) tensor
            reg_loss=reg_loss_tgc(ground_truth, return_ratio, mask)
            rank_loss=rank_loss_tgc(ground_truth, return_ratio, mask, self)
            if self.loss_name=='reg_rank_loss':
                loss = reg_loss + tf.cast(parameters['alpha'], tf.float32) * rank_loss # point pair loss
            elif self.loss_name=='listnet_loss':
                loss= listnet_loss(ground_truth, return_ratio, mask, self) # list-wise loss
          gradients = tape.gradient(loss, self.model.trainable_variables)
          print('trainable variables: ', self.model.trainable_variables)
          optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
          return rel_weight, loss, reg_loss, rank_loss, return_ratio
        
        @tf.function
        def test_step(Feature, base_price, ground_truth, mask):
            # The test step is not doing any further training. It is using the
            # model trained in the train_step. training=False
            
            rel_weight, prediction = self.model(Feature, training=False)
            return_ratio = tf.divide(tf.subtract(prediction, base_price), base_price)
            reg_loss=reg_loss_tgc(ground_truth, return_ratio, mask)
            rank_loss=rank_loss_tgc(ground_truth, return_ratio, mask, self)
            if self.loss_name=='reg_rank_loss':
                loss = reg_loss + tf.cast(parameters['alpha'], tf.float32) * rank_loss # point pair loss
            elif self.loss_name=='listnet_loss':
                loss= listnet_loss(ground_truth, return_ratio, mask, self) # list-wise loss
            return loss, reg_loss, rank_loss, return_ratio


        best_valid_pred, best_valid_gt, best_valid_mask=prediction_ground_truth_mask_initializer(
                len(self.tickers),
                self.test_index - self.valid_index)
        best_test_pred, best_test_gt, best_test_mask=prediction_ground_truth_mask_initializer(
                len(self.tickers),
                self.trade_dates - self.parameters['seq'] -self.test_index - self.steps + 1)
        best_valid_loss = np.inf

        batch_offsets = np.arange(start=0, stop=self.valid_index, dtype=int)
        
        '''train on training data'''
        for epoch in range(self.epochs):
            t1 = time()
            np.random.shuffle(batch_offsets)
            tra_loss = 0.0
            tra_reg_loss = 0.0
            tra_rank_loss = 0.0
            for j in range(self.valid_index - self.parameters['seq'] -
                                   self.steps + 1):
                # Reset the metrics at the start of the next epoch
                emb_batch, mask_batch, price_batch, gt_batch = self.get_batch(
                    batch_offsets[j]) # sequential_embedding, mask for time series data, price data and ground truth data
                
                rel_weight, train_cur_loss, train_cur_reg_loss, train_cur_rank_loss, cur_rr= train_step(
                        emb_batch, price_batch, gt_batch, mask_batch)
                
                 
                tra_loss += train_cur_loss
                tra_reg_loss += train_cur_reg_loss
                tra_rank_loss += train_cur_rank_loss
            
            print('Train Loss:',
                  tra_loss.numpy() / (self.valid_index - self.parameters['seq'] - self.steps + 1))
            
            
            '''test on validation set'''
            cur_valid_pred, cur_valid_gt, cur_valid_mask = prediction_ground_truth_mask_initializer(
                    len(self.tickers),
                    self.test_index - self.valid_index)
            val_loss = 0.0
            val_reg_loss = 0.0
            val_rank_loss = 0.0
            for cur_offset in range(
                        self.valid_index - self.parameters['seq'] - self.steps + 1,
                        self.test_index - self.parameters['seq'] - self.steps + 1
                    ):
                emb_batch, mask_batch, price_batch, gt_batch = self.get_batch(
                            cur_offset) # sequential_embedding, mask for time series data, price data and ground truth data
                # using test_step to get the validation loss
                val_cur_loss, val_cur_reg_loss, val_cur_rank_loss, cur_rr = test_step(emb_batch, price_batch, gt_batch, mask_batch)
                val_loss += val_cur_loss
                val_reg_loss += val_cur_reg_loss
                val_rank_loss += val_cur_rank_loss
        
                cur_valid_pred[:, cur_offset - (self.valid_index -
                            self.parameters['seq'] - self.steps + 1)] = copy.copy(cur_rr[:, 0])
                cur_valid_gt[:, cur_offset - (self.valid_index -
                            self.parameters['seq'] - self.steps + 1)] = copy.copy(gt_batch[:, 0]) 
                cur_valid_mask[:, cur_offset - (self.valid_index -
                            self.parameters['seq'] - self.steps + 1)] = copy.copy(mask_batch[:, 0])
            print('Valid Loss:',
                          val_loss.numpy()  / (self.test_index - self.valid_index))
          
            
            '''test on test set'''
            cur_test_pred,cur_test_gt,cur_test_mask = prediction_ground_truth_mask_initializer(
                    len(self.tickers),
                    self.trade_dates - self.test_index)
            test_loss = 0.0
            test_reg_loss = 0.0
            test_rank_loss = 0.0
            for cur_offset in range(
                    self.test_index - self.parameters['seq'] - self.steps + 1,
                    self.trade_dates - self.parameters['seq'] - self.steps + 1):
                emb_batch, mask_batch, price_batch, gt_batch = self.get_batch(
                            cur_offset) # sequential_embedding, mask for time series data, price data and ground truth data
                # using test step to get the test loss for current epoch
                test_cur_loss, test_cur_reg_loss, test_cur_rank_loss, cur_rr = test_step(emb_batch, price_batch, gt_batch, mask_batch)
        
                test_loss += test_cur_loss
                test_reg_loss += test_cur_reg_loss
                test_rank_loss += test_cur_rank_loss
        
                cur_test_pred[:, cur_offset - (self.test_index -
                            self.parameters['seq'] - self.steps + 1)] = copy.copy(cur_rr[:, 0])
                cur_test_gt[:, cur_offset - (self.test_index -
                            self.parameters['seq'] - self.steps + 1)] = copy.copy(gt_batch[:, 0])
                cur_test_mask[:, cur_offset - (self.test_index -
                            self.parameters['seq'] - self.steps + 1)] = copy.copy(mask_batch[:, 0])
            print('Test loss:',
                          test_loss.numpy() / (self.trade_dates - self.test_index))

            if val_loss / (self.test_index - self.valid_index) < best_valid_loss:
                     best_valid_loss = val_loss.numpy() / (self.test_index - self.valid_index)
                     best_valid_gt = copy.copy(cur_valid_gt)
                     best_valid_pred = copy.copy(cur_valid_pred)
                     best_valid_mask = copy.copy(cur_valid_mask)
                     best_test_gt = copy.copy(cur_test_gt)
                     best_test_pred = copy.copy(cur_test_pred)
                     best_test_mask = copy.copy(cur_test_mask)
                     print('Better valid loss:', best_valid_loss)
            '''Calculate the evaluation performance after certain epochs. If epoch==15000, use it 50 or 100'''
            if epoch%10==0:
                cur_valid_perf = evaluate(cur_valid_pred, cur_valid_gt, cur_valid_mask)
                print('\t Valid preformance:', cur_valid_perf)
                cur_test_perf = evaluate(cur_test_pred, cur_test_gt, cur_test_mask)
                print('\t Test performance:', cur_test_perf)
                self.df_loss=make_df_loss(self, epoch, cur_valid_perf, cur_test_perf, tra_loss, tra_reg_loss,
                                          val_loss, test_loss)
                self.df_loss.to_csv('df_loss_'+self.loss_name+'_'+self.market_name+'_'+
                                    self.relation_name+'_'+
                                    str(self.epochs)+'_epochs_'+str(self.inner_prod)+'_inner_prod'+'.csv',index=False)
            t4 = time()
            print('epoch:', epoch, ('time: %.4f ' % (t4 - t1)))              
        # Returning model in addition to other stats
        return self.model, best_valid_pred, best_valid_gt, best_valid_mask, best_test_pred, best_test_gt, best_test_mask



if __name__ == '__main__':
    desc = 'train a relational rank lstm model'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-p', help='path of EOD data',
                        default='../data/2013-01-01')
    parser.add_argument('-m', help='market name', default='NASDAQ')
    parser.add_argument('-ls',default='reg_rank_loss', 
                        help='listnet_loss or reg_rank_loss')
    parser.add_argument('-t', help='fname for selected tickers')
    parser.add_argument('-l', default=4,
                        help='length of historical sequence for feature')
    parser.add_argument('-u', default=64,
                        help='number of hidden units in lstm')
    parser.add_argument('-s', default=1,
                        help='steps to make prediction')
    parser.add_argument('-r', default=0.001,
                        help='learning rate')
    parser.add_argument('-a', default=1,
                        help='alpha, the weight of ranking loss')
    parser.add_argument('-g', '--gpu', type=int, default=0, help='use gpu')

    parser.add_argument('-e', '--emb_file', type=str,
                        default='NASDAQ_rank_lstm_seq-16_unit-64_2.csv.npy',
                        help='fname for pretrained sequential embedding') #NASDAQ_rank_lstm_seq-16_unit-64_2.csv.npy#NYSE_rank_lstm_seq-8_unit-32_0.csv.npy
    parser.add_argument('-rn', '--rel_name', type=str,
                        default='wikidata',
                        help='relation type: sector_industry,wikidata')
    parser.add_argument('-ip', '--inner_prod', type=int, default=1)
    parser.add_argument('-depth', type=int, default=5)
    parser.add_argument('-epoch_num', type=int, default=15)
    args = parser.parse_args()
    
    # Change the ticker file if you want to experiment with less number of stocks
    if args.t is None:
        args.t = args.m + '_tickers_qualify_dr-0.98_min-5_smooth_test.csv'
    args.gpu = (args.gpu == 1)

    args.inner_prod = (args.inner_prod == 1)

    parameters = {'seq': int(args.l), 'unit': int(args.u), 'lr': float(args.r),
                  'alpha': float(args.a)}
    print('arguments:', args)
    print('parameters:', parameters)

    RR_LSTM = ReRaLSTM(
        data_path=args.p,
        market_name=args.m,
        tickers_fname=args.t,
        relation_name=args.rel_name,
        emb_fname=args.emb_file,
        parameters=parameters,
        steps=1, epochs=args.epoch_num, batch_size=None, gpu=args.gpu,
        in_pro=args.inner_prod, depth=args.depth,
        loss_name=args.ls
    )
    
    df_loss=RR_LSTM.df_loss
    df_loss.to_csv('df_loss_'+RR_LSTM.loss_name+'_'+RR_LSTM.market_name+'_'+
                   RR_LSTM.relation_name+'_'+
                   str(RR_LSTM.epochs)+'_epochs_'+str(RR_LSTM.inner_prod)+'_inner_prod'+'.csv',index=False)
