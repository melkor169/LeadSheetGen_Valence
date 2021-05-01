# Generating Lead Sheets with Affect: A novel conditional seq2seq framework

## Description

A novel seq2seq framework where high-level musicalities (such us the valence of the chord progression) are fed to the Encoder, and they are "translated" to lead sheet events in the Decoder. For further details please read and cite our paper:


##### Makris, Dimos, Kat R. Agres, and Dorien Herremans. "Generating Lead Sheets with Affect: A Novel Conditional seq2seq Framework." arXiv preprint arXiv:2104.13056 (2021).

 
## Prerequisites

-Tensorflow 2.x (works with 1.x compatibility) <br />
-music21 <br />
-pretty_midi <br />
-numpy <br />

## Usage:

We offer a pre-trained Transformer model for quick inference usage. <br />
Arguments: <br />
	- o output_folder: Desired MIDI/MusicXML output folder. Default './Gen_Out/' <br />
	- ts time_sig: Set Time Signature '4/4' or '3/4'. Default '4/4' <br />
	- b num_of_bars: Desired Number of Bars 4-20. Default 16 <br />
	- v chord_progression_valence: Desired average valence for chord arrangement. <br />
                            [ -2: Low, 
                             -1: Moderate Low, 
                              0: Neutral (Default), 
                              1: Moderate High, 
                              2: High ] <br />
	-d density: Desired average density of events for every bar. 
			Choose either 'low', 'med' (Default) or 'hig'. <br />
	-t diversity: Set temperature for sampling. Optimal range 0.7-1.3. Default 1.0 <br />

Example: [python3 gen_leadsheet.py -ts '4/4' -b 12 -v 2 -d 'hig'] will generate a 12 bar lead sheet on 4/4 Time Signature, with High Valence regarding the Chord Progression and high density of events within the bars.

Please note that warning messages can occur if the network (due to "extreme" for the training dataset settings) cannot provide the exact number of bars or some other mismatches from the user's parameters.

The code and cleaned dataset that was used in the paper is inside the Preprocessing-Training folder.

## Reference

If you use this library, please cite the following work:

##### Makris, Dimos, Kat R. Agres, and Dorien Herremans. "Generating Lead Sheets with Affect: A Novel Conditional seq2seq Framework." arXiv preprint arXiv:2104.13056 (2021).
