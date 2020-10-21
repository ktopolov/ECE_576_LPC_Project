# -*- coding: utf-8 -*-
"""
This script performs Linear Predictive Coding on a dataset
"""
# External
import numpy as np
import os
import sounddevice as sd
import matplotlib.pyplot as plt
import time

# Internal
import audio_processing as ap

# %% Read File
path = 'C:\\Users\ktopo\Desktop\School\Courses\Masters\ECE 591 - Study on Signal Processing\Data\SpokenDigits'

# Shuffle so I don't hear the same speaker every stupid time
is_hear_reconstruct = True  # listen to every Nth speaker reconstructed
n_skip = 10  # skip every _ files

for ii, filename in enumerate(os.listdir(path)):
    file = os.path.join(path, filename)
    data, samp_rate = ap.read_wave(file)

    # Compute frame size
    samp_period = 1/samp_rate
    frame_time = 20e-3
    frame_len = int(frame_time / samp_period)
    
    overlap_time = 10e-3
    overlap_len = int(overlap_time / samp_period)

    # %% Check predictions
    lpc_order = 15
    lpc_coeffs, excitation_sig, gain = ap.lpc(data=data, lpc_order=lpc_order,
                                              frame_len=frame_len,
                                              overlap_len=overlap_len)

    is_debug = False
    if is_debug:
        plt.figure(3, clear=True)
        plt.plot(excitation_sig, 'k-', label='Residual Signal')
        plt.xlabel('Sample Index')
        plt.ylabel('Amplitude')
        plt.title('Excitation Signal in Prediction')
        plt.grid(True)
        plt.legend()

    # %% Quantize LPCs and Residual
    # Make function to do this
    
    # Use Huffman encoding

    # %% Inverse filter residual to obtain original
    reconstruct_sig = ap.reconstruct_lpc(excitation_sig=excitation_sig,
                                         lpc_coeffs=lpc_coeffs,
                                         frame_len=frame_len,
                                         overlap_len=overlap_len)
    is_debug = False
    if is_debug:
        plt.figure(5, clear=True)
        plt.plot(data, 'b-*', fillstyle='none', label='Original')
        plt.plot(reconstruct_sig, 'r-.', label='Reconstruction')
        plt.ylim([-5, 5])
        plt.xlabel('Sample Index')
        plt.ylabel('Amplitude')
        plt.title('Reconstruction from Residual')
        plt.grid(True)
        plt.legend()

    # Play True and Predicted
    if is_hear_reconstruct and (ii % n_skip == 0):
        sd.play(data, samplerate=samp_rate)
        time.sleep(1.2)
        sd.play(reconstruct_sig, samplerate=samp_rate)
