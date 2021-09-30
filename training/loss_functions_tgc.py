# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 21:51:17 2020

@author: suman
"""
import tensorflow as tf

def reg_loss_tgc(ground_truth, return_ratio, mask):
    reg_loss = tf.divide(tf.reduce_sum(
            tf.multiply(tf.square(tf.subtract(ground_truth, return_ratio)), mask)
            ), tf.reduce_sum(mask)) # Tensor, 
    return reg_loss

def rank_loss_tgc(ground_truth, return_ratio, mask, rr_lstm):
    pre_pw_dif = tf.subtract(
            tf.matmul(return_ratio, rr_lstm.model.all_one, transpose_b=True),
            tf.matmul(rr_lstm.model.all_one, return_ratio, transpose_b=True)
            ) # all_one: (1026,1)
    gt_pw_dif = tf.subtract(
            tf.matmul(rr_lstm.model.all_one, ground_truth, transpose_b=True),
            tf.matmul(ground_truth, rr_lstm.model.all_one, transpose_b=True)
            ) # all_one: (1026,1)
    mask_pw = tf.matmul(mask, mask, transpose_b=True)
    rank_loss = tf.reduce_mean(
            tf.nn.relu(
                    tf.multiply(
                            tf.multiply(pre_pw_dif, gt_pw_dif),mask_pw
                            )
                    )
                    )
    return rank_loss

def listnet_loss(ground_truth, return_ratio, mask, rr_lstm):
    pre_prob_dist = tf.nn.softmax(return_ratio, axis=0)
    gt_prob_dist = tf.nn.softmax(ground_truth, axis=0)
    per_example_loss= -100*tf.math.multiply(gt_prob_dist,tf.math.log(pre_prob_dist))
    batch_loss = tf.reduce_mean(per_example_loss)
    return batch_loss

