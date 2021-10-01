'''This code is the test with inclusion of node2vec'''

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
from graph_embedding import relation_node2vec

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



# Another model for 'inner product weight' can be similarly built
class MyModel(Model):
  def __init__(self, nCom, rel_mask, inner_prod, flat,rel_encoding,num_random_walks,len_random_walk,
               p_val,q_val,n2vemb_size,units = 0):
    super(MyModel, self).__init__()
    self.rel_mask = rel_mask
    self.inner_prod = inner_prod
    self.flat = flat
    self.all_one = tf.ones([nCom, 1], dtype=tf.float32)
    self.rel_encoding=rel_encoding.astype('float32')
    self.num_random_walks=num_random_walks
    self.len_random_walk=len_random_walk
    self.p_val=p_val
    self.q_val=q_val
    self.n2vemb_size=n2vemb_size
    self.prediction_layer = Dense(1,activation=tf.keras.layers.LeakyReLU(), 
                                  kernel_initializer='glorot_uniform')
    if self.flat:
        print('one more hidden layer')
        self.hidden_layer =  Dense(units, activation=tf.keras.layers.LeakyReLU(),
                                   kernel_initializer='glorot_uniform')
    else:
        self.hidden_layer = None
    
  def call(self, Feature):
      weight_masked=relation_node2vec(self.rel_encoding,self.num_random_walks,self.len_random_walk,
                                      self.p_val,self.q_val,self.n2vemb_size) # we are directly using embedding from node2vec
      rel_weight=weight_masked
      outputs_proped=weight_masked
      
      if self.flat:
          outputs_concated = self.hidden_layer(
              tf.concat([Feature, outputs_proped], axis=1))
      else:
          outputs_concated = tf.concat([Feature, outputs_proped], axis=1)
      prediction = self.prediction_layer(outputs_concated)
      print('prediction layer input shape: ',outputs_concated.shape)
      print('prediction layer output shape: ',prediction.shape)
       
      return rel_weight,prediction

class ReRaLSTM:
    def __init__(self, data_path, market_name, tickers_fname, relation_name,
                 emb_fname, parameters, depth, loss_name, num_random_walks,len_random_walk,
                 p_val, q_val, n2vemb_size, steps=1,
                 epochs=50, batch_size=None, flat=False, gpu=False, in_pro=False):
    
        seed = 123456789
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
        self.data_path = data_path
        self.market_name = market_name
        self.tickers_fname = tickers_fname
        self.relation_name = relation_name
        self.df_loss=pd.DataFrame()
        self.depth=depth
        self.loss_name=loss_name
        self.parameters = copy.copy(parameters)
        self.steps = steps
        self.epochs = epochs
        self.flat = flat
        self.inner_prod = in_pro
        self.valid_index = 756
        self.test_index = 1008
        self.fea_dim = 5
        self.gpu = gpu
        self.num_random_walks=num_random_walks
        self.len_random_walk=len_random_walk
        self.p_val=p_val
        self.q_val=q_val
        self.n2vemb_size=n2vemb_size
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
                             self.inner_prod, self.flat,self.rel_encoding, self.num_random_walks,self.len_random_walk,
                             self.p_val,self.q_val, self.n2vemb_size, self.parameters['unit'])


    def get_batch(self, offset=None):
        if offset is None:
            offset = random.randrange(0, self.valid_index)
        seq_len = self.parameters['seq']
        mask_batch = self.mask_data[:, offset: offset + seq_len + self.steps]
        mask_batch = np.min(mask_batch, axis=1)
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
        #train_loss = tf.keras.metrics.Mean(name='train_loss')
 
        @tf.function
        def train_step(Feature, base_price, ground_truth, mask):
          with tf.GradientTape() as tape:
            # training=True is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            rel_weight, prediction = self.model(Feature, training=True)
            return_ratio = tf.divide(tf.subtract(prediction, base_price), base_price) #(num_company,1) tensor
            reg_loss=reg_loss_tgc(ground_truth, return_ratio, mask)
            rank_loss=rank_loss_tgc(ground_truth, return_ratio, mask, self)
            if self.loss_name=='reg_rank_loss':
                loss = reg_loss + tf.cast(parameters['alpha'], tf.float32) * rank_loss
            elif self.loss_name=='listnet_loss':
                loss= listnet_loss(ground_truth, return_ratio, mask, self)
          gradients = tape.gradient(loss, self.model.trainable_variables)
          print('trainable variables: ', self.model.trainable_variables)
          optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
          '''new code: added rel_weight in return list'''
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
                loss = reg_loss + tf.cast(parameters['alpha'], tf.float32) * rank_loss
            elif self.loss_name=='listnet_loss':
                loss= listnet_loss(ground_truth, return_ratio, mask, self)
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
                emb_batch, mask_batch, price_batch, gt_batch = self.get_batch(batch_offsets[j])
                
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
                            cur_offset)
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
            print('Valid loss:',
                          val_loss.numpy()  / (self.test_index - self.valid_index))
          
            
            '''test on testing set'''
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
                self.df_loss.to_csv('df_loss_node2vec_'+self.loss_name+'_'+self.market_name+'_'+
                                    self.relation_name+'_'+
                                    str(self.epochs)+'_epochs_'+str(RR_LSTM.num_random_walks)+'_num_random_walks_'+
                                    str(RR_LSTM.len_random_walk)+'_len_random_walk_'+str(RR_LSTM.p_val)+'_p_val_'+
                                    str(RR_LSTM.q_val)+'_q_val_'+str(RR_LSTM.n2vemb_size)+'_n2vemb_size'+'.csv',index=False)
            t4 = time()
            print('epoch:', epoch, ('time: %.4f ' % (t4 - t1)))
        
                
        # The function is returning model in addition to other stats
        return self.model, best_valid_pred, best_valid_gt, best_valid_mask, best_test_pred, best_test_gt, best_test_mask



if __name__ == '__main__':
    desc = 'train a relational rank lstm model'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-paths', help='path of EOD data',
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
                        help='relation type: sector_industry or wikidata')
    parser.add_argument('-ip', '--inner_prod', type=int, default=1)
    parser.add_argument('-depth', type=int, default=5)
    parser.add_argument('-epoch_num', type=int, default=15)
    parser.add_argument('-num_rw',type=int,default=20)
    parser.add_argument('-len_rw',type=int, default=8)
    parser.add_argument('-p_val',type=int, default=1)
    parser.add_argument('-q_val',type=int, default=1)
    parser.add_argument('-n2vemb_size',type=int, default=64)
    args = parser.parse_args()

    if args.t is None:
        args.t = args.m + '_tickers_qualify_dr-0.98_min-5_smooth_test.csv'
    args.gpu = (args.gpu == 1)

    args.inner_prod = (args.inner_prod == 1)

    parameters = {'seq': int(args.l), 'unit': int(args.u), 'lr': float(args.r),
                  'alpha': float(args.a)}
    print('arguments:', args)
    print('parameters:', parameters)

    RR_LSTM = ReRaLSTM(
        data_path=args.paths,
        market_name=args.m,
        tickers_fname=args.t,
        relation_name=args.rel_name,
        emb_fname=args.emb_file,
        parameters=parameters,
        steps=1, epochs=args.epoch_num, batch_size=None, gpu=args.gpu,
        in_pro=args.inner_prod, depth=args.depth,
        loss_name=args.ls,
        num_random_walks=args.num_rw,
        len_random_walk=args.len_rw,
        p_val=args.p_val,
        q_val=args.q_val,
        n2vemb_size=args.n2vemb_size
    )
    pred_all = RR_LSTM.train() 
    df_loss=RR_LSTM.df_loss
    df_loss.to_csv('df_loss_node2vec_'+RR_LSTM.loss_name+'_'+RR_LSTM.market_name+'_'+
                   RR_LSTM.relation_name+'_'+
                   str(RR_LSTM.epochs)+'_epochs_'+str(RR_LSTM.num_random_walks)+'_num_random_walks_'+
                   str(RR_LSTM.len_random_walk)+'_len_random_walk_'+str(RR_LSTM.p_val)+'_p_val_'+
                   str(RR_LSTM.q_val)+'_q_val_'+str(RR_LSTM.n2vemb_size)+'_n2vemb_size'+'.csv',index=False)