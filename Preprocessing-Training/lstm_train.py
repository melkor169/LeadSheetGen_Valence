#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 15:28:05 2020


"""

import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
from tensorflow.keras import models,layers, optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
import pickle

    
#load the data
train_path = 'train_set.pickle'
test_path = 'test_set.pickle'


with open(train_path, 'rb') as handle:
    trainSet = pickle.load(handle)

with open(test_path, 'rb') as handle:
    testSet = pickle.load(handle)   

""" MODEL ARCHITECTURE """

# Hype Parameters
batch_size = 64
epochs = 100
LSTM_dim = 768 #can be used CuDNN of course
EMB_dim_enc = 48
EMB_dim_dec = EMB_dim_enc*4
enc_vocab = 22 #+1 padding
dec_vocab = 127 #+1 padding


'''Encoder'''
enc_input = layers.Input(shape=(None,), name = 'input_var1')

enc_emb = layers.Embedding(output_dim=EMB_dim_enc, input_dim=enc_vocab +1, mask_zero=True, name='encoder_embedding')
emb_input_enc = enc_emb(enc_input)

#fed it to stacked BLSTMs. Share  the states to the corresponding LSTM decoders
encoder_1 = layers.Bidirectional(layers.LSTM(LSTM_dim, return_sequences=True, return_state=True), name='encoder_blstm_1')
encoder1_outputs, forward1_h, forward1_c, backward1_h, backward1_c = encoder_1(emb_input_enc)
state1_h = layers.Concatenate()([forward1_h, backward1_h])
state1_c = layers.Concatenate()([forward1_c, backward1_c])
encoder1_states = [state1_h, state1_c]

encoder1_outputs = layers.Dropout(0.2)(encoder1_outputs)

encoder_2 = layers.Bidirectional(layers.LSTM(LSTM_dim, return_sequences=True, return_state=True), name='encoder_blstm_2')
encoder2_outputs, forward2_h, forward2_c, backward2_h, backward2_c = encoder_2(encoder1_outputs)
state2_h = layers.Concatenate()([forward2_h, backward2_h])
state2_c = layers.Concatenate()([forward2_c, backward2_c])
encoder2_states = [state2_h, state2_c]

encoder2_outputs = layers.Dropout(0.2)(encoder2_outputs)

encoder_3 = layers.Bidirectional(layers.LSTM(LSTM_dim, return_sequences=True, return_state=True), name='encoder_blstm_3')
encoder3_outputs, forward3_h, forward3_c, backward3_h, backward3_c = encoder_3(encoder2_outputs)
state3_h = layers.Concatenate()([forward3_h, backward3_h])
state3_c = layers.Concatenate()([forward3_c, backward3_c])
encoder3_states = [state3_h, state3_c]


'''Decoder'''
# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the
# return states in the training model, but we will use them in inference.
dec_input = layers.Input(shape=(None,), name = 'input_var2')

#create the embedding layer for Decoder
dec_emb = layers.Embedding(output_dim=EMB_dim_dec, input_dim=dec_vocab +1, mask_zero=True, name='decoder_embedding')
emb_input_dec = dec_emb(dec_input)

decoder1 = layers.LSTM(LSTM_dim*2, return_sequences=True, return_state=True, name='decoder_lstm_1')
decoder1_outputs, _, _ = decoder1(emb_input_dec, initial_state=encoder1_states)

decoder1_outputs = layers.Dropout(0.2)(decoder1_outputs)

decoder2 = layers.LSTM(LSTM_dim*2, return_sequences=True, return_state=True, name='decoder_lstm_2')
decoder2_outputs, _, _ = decoder2(decoder1_outputs, initial_state=encoder2_states)

decoder2_outputs = layers.Dropout(0.2)(decoder2_outputs)

decoder3 = layers.LSTM(LSTM_dim*2, return_sequences=True, return_state=True, name='decoder_lstm_3')
decoder3_outputs, _, _ = decoder3(decoder2_outputs, initial_state=encoder3_states)

'''Setting Outputs and Model Parameters'''
decoder_dense = layers.Dense(dec_vocab, activation='softmax', name = 'out_var1')

decoder_outputs = decoder_dense(decoder3_outputs)



model = models.Model(inputs = [enc_input, dec_input], 
                     outputs= [decoder_outputs])

optimizer = optimizers.Adam(0.0005, beta_1=0.9, beta_2=0.98, 
                                     epsilon=1e-9)

model.compile(optimizer=optimizer,
             metrics=['acc'],
             loss={'out_var1': 'categorical_crossentropy'},
             loss_weights={'out_var1': 1.0})

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
save_model = ModelCheckpoint('ChordDurMel_LSTM.h5', monitor='val_loss', verbose=1,
    save_best_only=True, save_weights_only=False, mode='min', save_freq='epoch')
stop_train = EarlyStopping( monitor='val_loss', patience=5, verbose=0, mode='min')

#train
h = model.fit({'input_var1': enc_inp_train, 'input_var2': dec_inp_train},
                    {'out_var1': dec_out_train },
                      batch_size=batch_size,
                      epochs=epochs,
                      validation_data= ([enc_inp_test,dec_inp_test],dec_out_test),
                      callbacks = [save_model, stop_train],
                      verbose=2)     