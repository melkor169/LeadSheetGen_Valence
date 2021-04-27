#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 15:11:06 2020

"""
import os
os.environ["CUDA_VISIBLE_DEVICES"]="3"
import tensorflow as tf
import numpy as np
from aux_files import Encoder,Decoder,CustomSchedule
import pickle


#load the data
train_path = 'train_set.pickle'
test_path = 'test_set.pickle'


with open(train_path, 'rb') as handle:
    trainSet = pickle.load(handle)

with open(test_path, 'rb') as handle:
    testSet = pickle.load(handle)


#set transformer parameters   

num_layers = 4  #4
d_model = 48 #for Embedding 
dff = 1536 #for Dense
num_heads = 8 #8
dropout_rate = 0.1
batch_size = 64
epochs = 120
enc_vocab = 22
dec_vocab = 127


enc_input = tf.keras.layers.Input(shape=(None,), name = 'input_var1')
dec_input = tf.keras.layers.Input(shape=(None,), name = 'input_var2')

encoder = Encoder(enc_vocab+1, num_layers = num_layers, d_model = d_model, num_heads = num_heads, dff = dff, dropout = dropout_rate)
decoder = Decoder(dec_vocab+1, num_layers = num_layers, d_model = d_model*4, num_heads = num_heads, dff = dff, dropout = dropout_rate)

x = encoder(enc_input)
x = decoder([dec_input, x] , mask = encoder.compute_mask(enc_input))

dec_output = tf.keras.layers.Dense(dec_vocab, activation='softmax', name = 'out_var1')
out = dec_output(x)

model = tf.keras.models.Model(inputs=[enc_input, dec_input], outputs=out)

#model.summary()

optimizer = tf.keras.optimizers.Adam( CustomSchedule(d_model), beta_1=0.9, beta_2=0.98, 
                                     epsilon=1e-9)


model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['acc'])

'''Train Model'''
#inputs
enc_inp_train = np.stack(trainSet['Encoder_Input'])
dec_inp_train = np.stack(trainSet['Decoder_Input'])
#outputs
dec_out_train = np.stack(trainSet['Decoder_Output'])
#inputs
enc_inp_test = np.stack(testSet['Encoder_Input'])
dec_inp_test = np.stack(testSet['Decoder_Input'])
#outputs
dec_out_test = np.stack(testSet['Decoder_Output'])


#setting checkpoints
save_model = tf.keras.callbacks.ModelCheckpoint('ChordDurMel_Trans_w.h5', monitor='val_loss', verbose=1,
    save_best_only=True, save_weights_only=True, mode='min', save_freq='epoch')
stop_train = tf.keras.callbacks.EarlyStopping( monitor='val_loss', patience=6, verbose=0, mode='min')

#train
h = model.fit({'input_var1': enc_inp_train, 'input_var2': dec_inp_train},
                    {'out_var1': dec_out_train },
                      batch_size=batch_size,
                      epochs=epochs,
                      validation_data= ([enc_inp_test,dec_inp_test],dec_out_test),
                      callbacks = [save_model, stop_train],
                      verbose=2)         

