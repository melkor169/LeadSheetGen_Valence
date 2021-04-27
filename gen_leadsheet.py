#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 16:55:40 2020


"""
import argparse
import pickle
from gen_utils import generate_leadsheet


'''Deployment Mode'''
'''Event Based Represenation'''
'''Transformer Model Only'''


'''0. Get User's Parameters'''
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_folder',default= './Gen_Out/',type=str,
                        help="MIDI file output folder. Default is "
                        "./Gen_Out/")
    
    parser.add_argument('-ts', '--time_sig', default='4/4', type=str,
                        help="Set Time Signature 4/4 or 3/4")

    parser.add_argument('-b', '--num_of_bars', default=16, type=int,
                        help="Desired Number of Bars 4-20")
    
    parser.add_argument('-v', '--chord_progression_valence', default=0, type=int,
                        help="Desired average valence for chord arrangement (integer) " 
                             "-2: Low "
                             "-1: Moderate Low "
                             "0: Neutral "
                             "1: Moderate High "
                             "2: High")

    parser.add_argument('-d', '--density', default='med', type=str,
                        help="Desired average density of events for every bar "
                             "Choose either 'low', 'med' or 'hig")

    parser.add_argument('-t', '--diversity', default=1.0, type=float,
                        help="Set temperature for sampling. Optimal range "
                             "0.7 to 1.3")

    return parser.parse_args()


if __name__== "__main__":
    args = get_args()
    

    '''1. Load Global Variables and Dictionaries'''
    #load pickles for onehotencoders
    encoders_trans = './aux_files/chords_encoders_all.pickle'
    with open(encoders_trans, 'rb') as handle:
        TransEncoders = pickle.load(handle)
    #[Encoder, Decoder]
    
    #load valence and density test templates for setting the Encoder seq
    val_temp_path = './aux_files/Valence_Templates.pickle'
    with open(val_temp_path, 'rb') as handle:
        val_templates = pickle.load(handle)
        
    dense_temp_path = './aux_files/Density_Templates.pickle'
    with open(dense_temp_path, 'rb') as handle:
        dense_templates = pickle.load(handle)
    
    
    '''2. Set User Music Parameters'''
    temperature = args.diversity
    timesig = args.time_sig
    #make it compatible for the functions
    nom = int(timesig.split('/')[0])
    den = int(timesig.split('/')[1])
    fts = str([nom,den])
    numOfBars = args.num_of_bars
    valence = str(args.chord_progression_valence)
    density = args.density
    gen_out = args.output_folder
    
    '''3. Call the function to Generate'''
    
    generate_leadsheet(temperature,fts,numOfBars,valence,density,gen_out,TransEncoders,
                       val_templates,dense_templates)




