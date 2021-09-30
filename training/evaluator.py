import math
import numpy as np
import scipy.stats as sps
from rbo import rbo_at_k, rbo_at_k_normalised_w

def bt_long_calculator(pre_topn, ground_truth, bt_longn, i):
    # back testing on top k stocks
    real_ret_rat_topn = 0
    for pre in pre_topn:
        real_ret_rat_topn += ground_truth[pre][i]
    real_ret_rat_topn /= len(pre_topn)
    bt_longn += real_ret_rat_topn
    return bt_longn

def evaluate(prediction, ground_truth, mask, report=False):
    assert ground_truth.shape == prediction.shape, 'shape mis-match'
    # Performance is the dictionary which will contain the mse, mrrt and btl
    performance = {}
    # calculation of mse. this is equivalent to reg_loss or regression loss
    performance['mse'] = np.linalg.norm((prediction - ground_truth) * mask)**2\
        / np.sum(mask)
    mrr_top = 0.0
    all_miss_days_top = 0
    bt_long = 1.0
    bt_long5 = 1.0
    bt_long10 = 1.0
    bt_long20 = 1.0
    bt_long50 = 1.0
    rbo_at_5_normalised=0.0
    rbo_at_10_normalised=0.0
    rbo_at_20_normalised=0.0
    rbo_at_50_normalised=0.0

    for i in range(prediction.shape[1]):
        # prediction.shape[1] is the number of days
        # This loop will iterate over the length of test and validation set
        # Actual rank based on ground truth
        rank_gt = np.argsort(ground_truth[:, i])
        gt_top1 = [] # will contain index of the top 1 stock by actual return
        gt_top5 = [] # will contain index of the top 5 stock by actual return
        gt_top10 = [] # will contain  index of the top 10 stock by actual return
        gt_top20 = [] # will contain index of the top 20 stock by actual return
        gt_top50 = [] # will contain index of the top 50 stock by actual return
        
        # Creasting list of top 1, 5, 10, 20 and 50 based on actual rank
        for j in range(1, prediction.shape[0] + 1):
            # This loop will iterate over the number of stocks (1 to 1026)
            cur_rank = rank_gt[-1 * j] # Actual rank
            if mask[cur_rank][i] < 0.5:
                continue
            if len(gt_top1) < 1:
                gt_top1.append(cur_rank) # index of the top 1 stock by actual return
            if len(gt_top5) < 5:
                gt_top5.append(cur_rank) # index of the top 5 stock by actual return
            if len(gt_top10) < 10:
                gt_top10.append(cur_rank) # index of the top 10 stock by actual return
            if len(gt_top20) < 20:
                gt_top20.append(cur_rank) # index of the top 20 stock by actual return
            if len(gt_top50) < 50:
                gt_top50.append(cur_rank) # index of the top 50 stock by actual return

        # Predicted Rank
        rank_pre = np.argsort(prediction[:, i])

        pre_top1 = [] # index of the top 1 stock by predicted return
        pre_top5 = [] # index of the top 5 stock by predicted return
        pre_top10 = [] # index of the top 10 stock by predicted return
        pre_top20 = [] # index of the top 20 stock by predicted return
        pre_top50 = [] # index of the top 50 stock by predicted return
        for j in range(1, prediction.shape[0] + 1):
            # This loop will iterate over the number of stocks (1 to num_company)
            cur_rank = rank_pre[-1 * j]
            if mask[cur_rank][i] < 0.5:
                continue
            if len(pre_top1) < 1:
                pre_top1.append(cur_rank) # index of the top 1 stock by predicted return
            if len(pre_top5) < 5:
                pre_top5.append(cur_rank) # index of the top 5 stock by predicted return
            if len(pre_top10) < 10:
                pre_top10.append(cur_rank) # index of the top 10 stock by predicted return
            if len(pre_top20) < 20:
                pre_top20.append(cur_rank) # index of the top 20 stock by predicted return
            if len(pre_top50) < 50:
                pre_top50.append(cur_rank) # index of the top 50 stock by predicted return

        # calculate mrr of top1
        top1_pos_in_gt = 0
        for j in range(1, prediction.shape[0] + 1):
            # This loop will iterate over the number of stocks (1 to num_company)
            cur_rank = rank_gt[-1 * j]
            if mask[cur_rank][i] < 0.5:
                continue
            else:
                # top1_pos_in_gt will calculate the rank of the predicted top stock
                # in actual ground truth
                top1_pos_in_gt += 1
                if cur_rank in pre_top1:
                    break
        if top1_pos_in_gt == 0:
            all_miss_days_top += 1
        else:
            # mrr_top will contain sum over all days/length of validation and
            # test set
            mrr_top += 1.0 / top1_pos_in_gt

        # back testing on top 1 to calculate IRR
        real_ret_rat_top = ground_truth[(pre_top1)[0]][i]
        bt_long += real_ret_rat_top

        # back testing
        bt_long5= bt_long_calculator(pre_top5, ground_truth, bt_long5, i) # back testing on top 5        
        bt_long10= bt_long_calculator(pre_top10, ground_truth, bt_long10, i) # back testing on top 10
        bt_long20= bt_long_calculator(pre_top20, ground_truth, bt_long20, i) # back testing on top 20
        bt_long50= bt_long_calculator(pre_top50, ground_truth, bt_long50, i) # back testing on top 50
        
        # rbo calculation
        rbo_at_5_normalised+=rbo_at_k_normalised_w(pre_top5,gt_top5,p=0.80, depth=5)
        rbo_at_10_normalised+=rbo_at_k_normalised_w(pre_top5,gt_top5,p=0.90, depth=10)
        rbo_at_20_normalised+=rbo_at_k_normalised_w(pre_top5,gt_top5,p=0.95,depth=20)
        rbo_at_50_normalised+=rbo_at_k_normalised_w(pre_top5,gt_top5,p=0.98, depth=50)


    performance['mrrt'] = mrr_top / (prediction.shape[1] - all_miss_days_top)
    performance['rbo_at_5_normalized'] = rbo_at_5_normalised / (prediction.shape[1])
    performance['rbo_at_10_normalized'] = rbo_at_10_normalised / (prediction.shape[1])
    performance['rbo_at_20_normalized'] = rbo_at_20_normalised / (prediction.shape[1])
    performance['rbo_at_50_normalized'] = rbo_at_50_normalised / (prediction.shape[1])
    performance['btl'] = bt_long
    performance['bt5_unweighted'] = bt_long5
    performance['bt10_unweighted'] = bt_long10
    performance['bt20_unweighted'] = bt_long20
    performance['bt50_unweighted'] = bt_long50
    return performance

def make_df_loss(rr_lstm,epoch,cur_valid_perf,cur_test_perf,tra_loss,tra_reg_loss,val_loss,test_loss):
    loss_df=rr_lstm.df_loss.append({
                        'epoch':epoch,
                        'market':rr_lstm.market_name,
                        'relation_name':rr_lstm.relation_name,
                        'loss_name':rr_lstm.loss_name,
                        'train_total_loss':tra_loss.numpy() / (rr_lstm.valid_index - rr_lstm.parameters['seq'] - rr_lstm.steps + 1),
                        'train_reg_loss':tra_reg_loss.numpy() / (rr_lstm.valid_index - rr_lstm.parameters['seq'] - rr_lstm.steps + 1),
                        'valid_total_loss':val_loss.numpy()  / (rr_lstm.test_index - rr_lstm.valid_index),
                        'valid_reg_loss':cur_valid_perf['mse'],'valid_mrrt':cur_valid_perf['mrrt'],
                        'valid_bt1':cur_valid_perf['btl'],'valid_bt5_unweighted':cur_valid_perf['bt5_unweighted'],
                        'valid_bt10_unweighted':cur_valid_perf['bt10_unweighted'],
                        'valid_bt20_unweighted':cur_valid_perf['bt20_unweighted'],
                        'valid_bt50_unweighted':cur_valid_perf['bt50_unweighted'],
                        'valid_rbo_at_5_normalized':cur_valid_perf['rbo_at_5_normalized'],
                        'valid_rbo_at_10_normalized':cur_valid_perf['rbo_at_10_normalized'],
                        'valid_rbo_at_20_normalized':cur_valid_perf['rbo_at_20_normalized'],
                        'valid_rbo_at_50_normalized':cur_valid_perf['rbo_at_50_normalized'],
                        'test_total_loss':test_loss.numpy() / (rr_lstm.trade_dates - rr_lstm.test_index),
                        'test_reg_loss':cur_test_perf['mse'],'test_mrrt':cur_test_perf['mrrt'],
                        'test_bt1':cur_test_perf['btl'],'test_bt5_unweighted':cur_test_perf['bt5_unweighted'],
                        'test_bt10_unweighted':cur_test_perf['bt10_unweighted'],
                        'test_bt20_unweighted':cur_test_perf['bt20_unweighted'],
                        'test_bt50_unweighted':cur_test_perf['bt50_unweighted'],
                        'test_rbo_at_5_normalized':cur_test_perf['rbo_at_5_normalized'],
                        'test_rbo_at_10_normalized':cur_test_perf['rbo_at_10_normalized'],
                        'test_rbo_at_20_normalized':cur_test_perf['rbo_at_20_normalized'],
                        'test_rbo_at_50_normalized':cur_test_perf['rbo_at_50_normalized']},
                    ignore_index=True)
    return loss_df