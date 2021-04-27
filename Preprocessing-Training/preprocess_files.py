#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 15:11:06 2020

"""

import numpy as np
from random import shuffle
import pickle
from train_utils import create_onehot_dict, calc_timesig_dur, cut_chord_dur_seqs_v,\
    calc_chords_val, analyze_valence_density, calc_density

#load the data
all_Data_path ='Data_CMajor_v2.1.pickle'
'''This is the preprocessed Wikiphonia dataset after the steps described in the
 paper. The original wikiphonia you can find it in other github repos easily'''
 
'''For every piece it contained the preprocessed information for every bar which
contains the following:
    -Grouping(Phrasing) Indication
    -Onset (the start of the bar in quarters in the piece)
    -Offset (the end of the bar in quarters in the piece)
    -Time Signature
    -Tonality (already fixed to C/Am)
    -a list of Events of the bar which have the the following info:
        -onset
        -offset
        -duration
        -Chord 
        -Melody'''


with open(all_Data_path, 'rb') as handle:
    allData = pickle.load(handle)
    
  

'''Data Preparation'''
#initialize data. Create a super dict for storing all the data
dataDict = {'Chords': [],
            'Durations': [],
            'Melody': [],
            'TimeSigs': [],
            'Groupings': []}

#do not include files smaller thatn minEvents length
minEvents = 16

bar_exceptions = 0
event_exceptions = 0
for k in allData.keys():
    aOccur = allData[k]
    #initialize lists for Chords Duration and TimeSigs
    ChordList = []
    MelList = []
    DurList = []
    SigList = []
    GroupList = []
    fonset = aOccur['allMeasures'][0]['allEvents'][0]['onset']
    foffset = aOccur['allMeasures'][0]['allEvents'][-1]['offset']
    timesig_dur = calc_timesig_dur(aOccur['allMeasures'][0]['TimeSignature'])
    if fonset != 0.0 or timesig_dur!= foffset:
        #delete the first bar. Begin with the second
        bar_begin = 1
        aOccur['allMeasures'][bar_begin]['Group_ind'] = 'start1'
        aOccur['allMeasures'][bar_begin+1]['Group_ind'] = 'start2'
        bar_exceptions += 1
    else:
        bar_begin = 0
    ChordList.append('bar')
    MelList.append('bar')
    DurList.append('bar')
    SigList.append('bar')
    GroupList.append('bar')
    for b in range(bar_begin,len(aOccur['allMeasures'])):
        #get time sig and grouping
        timeSig = str(aOccur['allMeasures'][b]['TimeSignature'])
        groupInd = aOccur['allMeasures'][b]['Group_ind']
        #for every measure
        for m in range(0,len(aOccur['allMeasures'][b]['allEvents'])):
            chord = aOccur['allMeasures'][b]['allEvents'][m]['Chord']
            mel = str(aOccur['allMeasures'][b]['allEvents'][m]['Melody'])
            if mel == 'rest':
                mel = 'Rest'
            dur =  str(aOccur['allMeasures'][b]['allEvents'][m]['duration'])
            ChordList.append(chord)
            MelList.append(mel)
            DurList.append(dur)
            SigList.append(timeSig)
            GroupList.append(groupInd)
        #when measure is off add BAR flags
        ChordList.append('bar')
        MelList.append('bar')
        DurList.append('bar')
        SigList.append('bar')
        GroupList.append('bar')
    #append them to super dict
    if len(ChordList) > minEvents:

        dataDict['Chords'].append(ChordList)
        dataDict['Melody'].append(MelList)
        dataDict['Durations'].append(DurList)
        dataDict['TimeSigs'].append(SigList)
        dataDict['Groupings'].append(GroupList)
    else:
        event_exceptions += 1

print(bar_exceptions, 'bar exceptions were encountered')     
print(event_exceptions, 'event exceptions were encountered')  


#Set Maximum Seq_Lengths on note events. 
max_seq_length = 128 #fixed

#fix trainDict by cutting to smaller phrases
dataDict_s = cut_chord_dur_seqs_v(dataDict, max_seq_length)

"""Convert the dataDict streams to full event representation"""

#create the new Dictionary incluing also Encoder data
dataDict = {'Encoder': [], #information for Conditions: TimeSig,Grouping,Valence,Density and bar
            'Decoder': [], # [chord,melody, dur form]
            'ValTemplates': [], # will store the testing only
            'DensityTemplates': []} # will store the testing only

#create Encoder-Decoder data
max_enc_length = 0
max_dec_length = 0

for t in range(0, len(dataDict_s['Chords'])):
    aChord_seq = dataDict_s['Chords'][t]
    aDur_seq = dataDict_s['Durations'][t]    
    aMel_seq = dataDict_s['Melody'][t]   
    
    aSig_seq = dataDict_s['TimeSigs'][t]
    aGroup_seq = dataDict_s['Groupings'][t]
    
    ValenceEncList = []
    DensityEncList = []
    
    EncList = ['bar'] #start with bar indication 
    bar_idxs = [i for i, x in enumerate(aSig_seq) if x == "bar"]
    #first for Encoder
    for b in range(0,len(bar_idxs)-1):
        #calculate chord valence within a bar. Get the chords first
        chords_bar = aChord_seq[bar_idxs[b]+1:bar_idxs[b+1]]
        #calculate the density of events
        dense_bar = str(calc_density(len(chords_bar)))
        DensityEncList.append(dense_bar)
        #calculate the overall valence
        val_bar = str(calc_chords_val(chords_bar))
        ValenceEncList.append(val_bar)
        idx = bar_idxs[b] + 1
        sig_bar = aSig_seq[idx]
        group_bar = aGroup_seq[idx]
        #append it to the EncList with order TimeSig,Grouping,Valence,Density and bar
        EncList.extend([sig_bar,group_bar,val_bar,dense_bar,'bar'])
    #check for encoder seq length
    if len(EncList) > max_enc_length:
        max_enc_length = len(EncList)
    #Then Decoder
    DecList = ['bar'] #start with bar indication 
    for b in range(0,len(bar_idxs)-1):
        #get all chords and durations
        chords_bar = aChord_seq[bar_idxs[b]+1:bar_idxs[b+1]]
        durs_bar = aDur_seq[bar_idxs[b]+1:bar_idxs[b+1]]
        mels_bar = aMel_seq[bar_idxs[b]+1:bar_idxs[b+1]]
        for l in range(0,len(chords_bar)):
            DecList.append(chords_bar[l])
            DecList.append(mels_bar[l])
            DecList.append(durs_bar[l])
        #then add bar flag
        DecList.append('bar')
    #check for decoder seq length
    if len(DecList) > max_dec_length:
        max_dec_length = len(DecList)
      
    #save them
    dataDict['Encoder'].append(EncList)
    dataDict['Decoder'].append(DecList)
    dataDict['ValTemplates'].append(ValenceEncList)
    dataDict['DensityTemplates'].append(DensityEncList)

'''Create OneHotEncoders for Encoder-Decoder'''
#get lists with all occurs to set OneHotEncoders
allEncOccs = []
allDecOccs = []

for k in range(0, len(dataDict['Encoder'])):
    #get allEncoder and Decoder events
    allEncOccs.extend(dataDict['Encoder'][k])
    allDecOccs.extend(dataDict['Decoder'][k])
    
#Add in the vocabulories the EOS SOS flags
allEncOccs.extend(['sos','eos'])
allDecOccs.extend(['sos','eos'])

EncEncoder = create_onehot_dict(allEncOccs)
DecEncoder = create_onehot_dict(allDecOccs)

#vocabulory sizes
enc_vocab = EncEncoder.categories_[0].shape[0]  #22
dec_vocab = DecEncoder.categories_[0].shape[0]  #127

#save the Encoders for the generation stage
encoders_path = 'chords_encoders_all.pickle'
with open(encoders_path, 'wb') as handle:
    pickle.dump([EncEncoder, DecEncoder], handle, protocol=pickle.HIGHEST_PROTOCOL)


'''Transform the dictionaries to one-hot encodings and add padding'''

#set sequence length decoder encoder        
dec_seq_length = max_dec_length + 1 #for sos or eos #359
enc_seq_length = max_enc_length + 2 #for sos and eos indications #263


trainDict = {'Encoder_Input': [],
            'Decoder_Input': [],
            'Decoder_Output': []}

for t in range(0, len(dataDict['Encoder'])):
    #prepare data for encoders decoders
    aEnc_seq = dataDict['Encoder'][t]
    aDec_seq = dataDict['Decoder'][t]
      
    pad_lgt_enc = enc_seq_length-len(aEnc_seq)-2
    pad_lgt_dec = dec_seq_length-len(aDec_seq)-1

    
    '''Encoder'''
    Enc_pad_emb = np.array(pad_lgt_enc*[0])   
    Enc_Input = EncEncoder.transform(np.array(['sos']+aEnc_seq+['eos']).reshape(-1, 1)).toarray()
    Enc_Input = [np.where(r==1)[0][0] for r in Enc_Input] #for embeddings
    Enc_Input = [x+1 for x in Enc_Input] #shift by one in order to have 0 as pad
    trainDict['Encoder_Input'].append(np.concatenate((Enc_Input,Enc_pad_emb), axis = 0))
    
    '''Decoder'''
    Dec_pad_emb = np.array(pad_lgt_dec*[0])   
    Dec_Input = DecEncoder.transform(np.array(['sos']+aDec_seq).reshape(-1, 1)).toarray()
    Dec_Input = [np.where(r==1)[0][0] for r in Dec_Input] #for embeddings
    Dec_Input = [x+1 for x in Dec_Input] #shift by one in order to have 0 as pad
    trainDict['Decoder_Input'].append(np.concatenate((Dec_Input,Dec_pad_emb), axis = 0)) 
    
    Dec_pad = np.zeros((pad_lgt_dec,dec_vocab))
    Dec_Output = DecEncoder.transform(np.array(aDec_seq+['eos']).reshape(-1, 1)).toarray()
    trainDict['Decoder_Output'].append(np.concatenate((Dec_Output, Dec_pad), axis = 0))  


'''Split the dataset to train test 85-15'''
index_shuf = list(range(len(trainDict['Encoder_Input'])))
shuffle(index_shuf)

trainSet = {'Encoder_Input': [],
            'Decoder_Input': [],
            'Decoder_Output': []}

testSet = {'Encoder_Input': [],
            'Decoder_Input': [],
            'Decoder_Output': []}


trIDXs = int(0.85*len(index_shuf))
for i in range(0,trIDXs):
    trainSet['Encoder_Input'].append(trainDict['Encoder_Input'][index_shuf[i]])
    trainSet['Decoder_Input'].append(trainDict['Decoder_Input'][index_shuf[i]])
    trainSet['Decoder_Output'].append(trainDict['Decoder_Output'][index_shuf[i]])

 
allValencesTemps = [] #only for testing dataset
allDensityTemps = []
for i in range(trIDXs,len(index_shuf)):
    testSet['Encoder_Input'].append(trainDict['Encoder_Input'][index_shuf[i]])
    testSet['Decoder_Input'].append(trainDict['Decoder_Input'][index_shuf[i]])
    testSet['Decoder_Output'].append(trainDict['Decoder_Output'][index_shuf[i]])
    allValencesTemps.append(dataDict['ValTemplates'][index_shuf[i]])
    allDensityTemps.append(dataDict['DensityTemplates'][index_shuf[i]])
    
#save them
train_path = 'train_set.pickle'
test_path = 'test_set.pickle'

with open(train_path, 'wb') as handle:
    pickle.dump(trainSet, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
with open(test_path, 'wb') as handle:
    pickle.dump(testSet, handle, protocol=pickle.HIGHEST_PROTOCOL)  

'''analyze test valence and density templates'''
valTemplates,denseTemplates = analyze_valence_density(allValencesTemps, allDensityTemps)
val_path ='Valence_Templates.pickle'
with open(val_path, 'wb') as handle:
    pickle.dump(valTemplates, handle, protocol=pickle.HIGHEST_PROTOCOL)   
    
dense_path ='Density_Templates.pickle'
with open(dense_path, 'wb') as handle:
    pickle.dump(denseTemplates, handle, protocol=pickle.HIGHEST_PROTOCOL) 
    
print('Completed!')
