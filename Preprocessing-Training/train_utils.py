#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 18:43:45 2020

@author: bougatsas
"""
import re
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from statistics import median

'''Global Parameters'''

thres = 64 #threshold for a phrase

''''''''''''''''''''''''''''''''''''''''''''''''

def calc_density(lgt):

    if lgt < 3:
        dense = 'low'
    elif lgt >= 5:
        dense = 'hig'
    else:
        dense = 'med'
        
    return dense

def calc_chords_val (chords_bar):
    
    val = []
    
    for c in chords_bar:
        if 'maj7' in c: #maj 7th
            val.append(0.83)#2
        elif 'm7' in c: #minor 7th
            val.append(-0.46)#-1
        elif '7' in c: #major 7th
            val.append(-0.02)#0
        elif 'm9' in c: #minor 9th
            val.append(-0.15)#0
        elif '9' in c: #major 9th
            val.append(0.51)#1
        elif 'dim' in c: #diminished
            val.append(-0.43)#-1
        elif 'Rest' in c: #Rest
            val.append(0)
        elif 'm' in c: #Minor
            val.append(-0.81)#-2
        else: #major
            val.append(0.87)#2
            
    #get the median
    med_val = median(val)
    
    #check the range
    if med_val > 0.6:
        val_idx = 2
    elif med_val > 0.2 and med_val <= 0.6:
        val_idx = 1
    elif med_val > -0.2 and med_val <= 0.2:
        val_idx = 0
    elif med_val > -0.6 and med_val <= -0.2:
        val_idx = -1
    else:
        val_idx = -2
            
    return val_idx#round(val/len(chords_bar))



def analyze_valence_density(allValencesTemps,allDensityTemps):
    
    valenceDict = {}
    denseDict = {}
    
    for i in range(0,len(allValencesTemps)):
        aVal = allValencesTemps[i]
        aVal = list(map(int, aVal))
        
        aDen = allDensityTemps[i]
        newDen = []
        for k in aDen:
            if k == 'low':
                newDen.append(0)
            elif k == 'med':
                newDen.append(1)
            else:
                newDen.append(2)

        #find score and length
        lgt = str(len(aVal))
        score = sum(aVal) / len(aVal)
        scoreDen = sum(newDen) / len(newDen)
        if score >= 1.5:
            cat = str(2)
        elif score >= 0.5 and score < 1.5:
            cat = str(1)
        elif score >= -0.5 and score <0.5:
            cat = str(0)
        elif score >= -1.5 and score <-0.5:
            cat = str(-1)
        else:
            cat = str(-2)
            
        if scoreDen <= 0.5:
            catD = 'low'
        elif scoreDen >= 1.5:
            catD = 'hig'
        else:
            catD ='med'
        
        #add it to the Dict. Check if the path exists
        if lgt in valenceDict:
            #check if the category exists
            if cat in valenceDict[lgt]:
                valenceDict[lgt][cat].append(aVal)
            else:
                valenceDict[lgt][cat] = []
                valenceDict[lgt][cat].append(aVal)               

        else: #create the whole path
            valenceDict[lgt] = {}
            valenceDict[lgt][cat] = []
            valenceDict[lgt][cat].append(aVal)
            
        if lgt in denseDict:
            #check if the category exists
            if catD in denseDict[lgt]:
                denseDict[lgt][catD].append(aDen)
            else:
                denseDict[lgt][catD] = []
                denseDict[lgt][catD].append(aDen)               

        else: #create the whole path
            denseDict[lgt] = {}
            denseDict[lgt][catD] = []
            denseDict[lgt][catD].append(aDen)
            
    return valenceDict, denseDict 



                 

def cut_chord_dur_seqs_v(dataDict, enc_seq_length):
    
    newDict = {'Chords': [],
            'Durations': [],
            'TimeSigs': [],
            'Groupings': [],
            'Melody': []}
    

    for t in range(0, len(dataDict['Chords'])):
        #prepare data for encoders decoders
        aChord_seq = dataDict['Chords'][t]
        aDur_seq = dataDict['Durations'][t]
        aSig_seq = dataDict['TimeSigs'][t]
        aGroup_seq = dataDict['Groupings'][t]
        aMel_seq = dataDict['Melody'][t]
        if len(aChord_seq) > enc_seq_length:

            aG = ['b' if x=='bar' else x for x in aGroup_seq]
            aG = ['-' if x=='start2' else x for x in aG]
            aG = ['-' if x=='end1' else x for x in aG]
            aG = ['s' if x=='start1' else x for x in aG]
            aG = ['e' if x=='end2' else x for x in aG]
            aGroup_str = ''.join(aG)
            group_pat = re.compile('ebs')
            positions = [match.end() - 1 for match in re.finditer(group_pat, aGroup_str)]
            if not positions:
                pairs = [[0,len(aChord_seq)]]
    
            else:
                positions.insert(0,0)   
                positions.append(len(aChord_seq))     
                pairs = []
                for p in range(1,len(positions)):
                    if p == 1:
                        aPair = [positions[p-1], positions[p]]
                    else:
                        aPair = [positions[p-1]-1, positions[p]]
                    pairs.append(aPair)
                    
                pairs = merge_groups (pairs,enc_seq_length)
                        
            for pa in pairs:
                s_idx = pa[0]
                e_idx = pa[1]
                c_seq = aChord_seq[s_idx:e_idx]
                d_seq = aDur_seq[s_idx:e_idx]
                s_seq = aSig_seq[s_idx:e_idx]
                g_seq = aGroup_seq[s_idx:e_idx]
                m_seq = aMel_seq[s_idx:e_idx]
                if e_idx - s_idx > enc_seq_length:
                    bar_idxs = [i for i, x in enumerate(g_seq) if x == "bar"]
                    np_idx = np.where(np.array(bar_idxs) <= enc_seq_length)[0][-1]
                    cut_idx = bar_idxs[np_idx]
                    #cut the seqs 
                    newDict['Chords'].append(c_seq[:cut_idx])
                    newDict['Durations'].append(d_seq[:cut_idx])
                    newDict['TimeSigs'].append(s_seq[:cut_idx])
                    newDict['Groupings'].append(g_seq[:cut_idx])    
                    newDict['Melody'].append(m_seq[:cut_idx]) 
                else:
                    if len(c_seq) >= thres:
                        newDict['Chords'].append(c_seq)
                        newDict['Durations'].append(d_seq)
                        newDict['TimeSigs'].append(s_seq)
                        newDict['Groupings'].append(g_seq)
                        newDict['Melody'].append(m_seq)
                
        else:
  
             newDict['Chords'].append(aChord_seq)
             newDict['Durations'].append(aDur_seq)
             newDict['TimeSigs'].append(aSig_seq)
             newDict['Groupings'].append(aGroup_seq)
             newDict['Melody'].append(aMel_seq)
             
    return newDict



def create_onehot_dict(all_occs):
    
    onehotEnc = OneHotEncoder(handle_unknown='ignore')
    onehotEnc.fit(np.array(all_occs).reshape(-1,1))
    
    return onehotEnc


def calc_timesig_dur(timesig):
    
    return (timesig[0]*4)/timesig[1]




def merge_groups (pairs, enc_seq_length):
    
    diffs = []
    for i in pairs:
        diffs.append(i[1]-i[0])
    
    groups = []
    p = 1
    cnt = diffs[0]
    while p < len(diffs):
        newcnt = cnt + diffs[p]
        if newcnt >= enc_seq_length:
            groups.append(pairs[p-1])
            cnt = diffs[p]
            p += 1
        else:
            cnt = newcnt
            p += 1
            
    groups.append(pairs[-1])
    
    for g in range(0,len(groups)-1):
        groups[g+1][0] = groups[g][1]-1
        
    return groups






       
        