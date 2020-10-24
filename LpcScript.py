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
import random

# Internal
import audio_processing as ap

# %% Read File
path = 'C:\\Users\\ktopo\\Desktop\\School\\Courses\\Masters\\ECE 576 - Information Engineering\\Project\\sample_voice_data-master\\sample_voice_data-master\\females'
files = os.listdir(path)
random.shuffle(files)

print('***************')
n_files = 51  # numebr of files to load
files = files[:n_files]

# Configure LPC
lpc_order = 10  # decreasing will make excitation signal approach speech signal
frame_time = 20e-3
overlap_time = 10e-3

# %% Encoder
# Data transmitted to decoder
TransmitData = {'lpc_coeffs': [],
                'excitation_sigs': []}
orig_sigs = []
orig_data_rate = np.zeros((n_files,))

for ii, filename in enumerate(files):
    print('\rEncoding File ' + str(ii + 1) + '/' + str(n_files), end='')
    file = os.path.join(path, filename)
    data, samp_rate = ap.read_wave(file)
    orig_sigs.append(data)

    # Compute frame size
    samp_period = 1/samp_rate
    frame_len = int(frame_time / samp_period)
    overlap_len = int(overlap_time / samp_period)

    # Compute LPCs and excitation using baseline method
    lpc_coeffs, exc_sig_per_frame, gain = ap.lpc(data=data, lpc_order=lpc_order,
                                                 frame_len=frame_len,
                                                 overlap_len=overlap_len)

    # OUTPUTS
    TransmitData['lpc_coeffs'].append(lpc_coeffs)
    TransmitData['excitation_sigs'].append(exc_sig_per_frame)

# %% View distribution of coefficients and excitation signal
# Coefficients (all together; independent would be more efficient)
lpc_coeff_array = np.concatenate(TransmitData['lpc_coeffs'], axis=0)

n_levels = 16  # only for plotting
plt.figure(10, clear=True)
plt.hist(lpc_coeff_array.flatten(), bins=n_levels)
plt.xlabel('Value')
plt.ylabel('# Occurences')
plt.title('Distribution of LPC Coefficients')
plt.grid(True)

# Excitation signal
excitation_sig_array = np.concatenate(TransmitData['excitation_sigs'], axis=0)

plt.figure(11, clear=True)
plt.hist(excitation_sig_array.flatten(), bins=n_levels)
plt.xlabel('Value')
plt.ylabel('# Occurences')
plt.title('Distribution of Excitation Signal Values')
plt.grid(True)

# %% Quantize LPCs and Residual
is_quantize = True
if is_quantize:
    # LPCs
    coeff_max = lpc_coeff_array.max()
    coeff_min = lpc_coeff_array.min()
    n_coeff_bits = 14  # number of bits per LPC
    
    # Excitation
    exc_max = excitation_sig_array.max()
    exc_min = excitation_sig_array.min()
    n_exc_bits = 12  # number of bits per LPC
    
    for ii in range(n_files):
        print('\rQuantizing Signal ' + str(ii + 1) + '/' + str(n_files), end='')
        ap.uniform_quantize(TransmitData['lpc_coeffs'][ii], coeff_max,
                            coeff_min, n_coeff_bits)
        ap.uniform_quantize(TransmitData['excitation_sigs'][ii], exc_max,
                            exc_min, n_exc_bits)
    # Use Huffman encoding
# Free memory
del excitation_sig_array
del lpc_coeff_array

# %% Data Rate
# Original Data
orig_data_rate = 16 * samp_rate  # 16 bit/samp * samp_rate samps/sec

# LPC Base case
n_lpc_bits = lpc_order * n_coeff_bits  # bits per frame
n_excitation_bits = n_exc_bits * TransmitData['excitation_sigs'][ii].shape[1]
lpc_base_data_rate = (n_lpc_bits + n_excitation_bits) / frame_time

# %% Decoder
reconstruct_sigs = []

for ii, filename in enumerate(files):
    # INPUTS
    exc_sig_per_frame = TransmitData['excitation_sigs'][ii]
    lpc_coeffs = TransmitData['lpc_coeffs'][ii]

    # Reconstruction
    reconstruct_sig = ap.reconstruct_lpc(exc_sig_per_frame=exc_sig_per_frame,
                                         lpc_coeffs=lpc_coeffs,
                                         frame_len=frame_len,
                                         overlap_len=overlap_len)
    reconstruct_sigs.append(reconstruct_sig.flatten())  # still zero-padded; must adjust to compare to original

# %% Analyze Reconstruction Error
is_hear_reconstruct = True  # listen to every Nth speaker reconstructed
is_show_reconstruct = False  # show reconstruction plot
n_listens = 1  # how many random reconstructions to play
random_listens = np.random.randint(low=0, high=n_files, size=(n_listens,))

n_tot_samps = 0
squared_err = 0

for ii, (reconstruct_sig, orig_sig) in enumerate(zip(reconstruct_sigs, orig_sigs)):
    print('\rReconstructing Signal ' + str(ii + 1) + '/' + str(len(orig_sigs)), end='')
    # Align reconstructed data to the original
    n_front = np.ceil(overlap_len//2).astype(int) + lpc_order  # number of samples in front of orig signal that are not used in LPC
    n_long = reconstruct_sig.shape[0]
    orig_sig = orig_sig[n_front:n_front + n_long]
    
    n_tot_samps += n_long
    squared_err += np.sum((orig_sig - reconstruct_sig)**2)
    
    if is_show_reconstruct:
        plt.figure(5, clear=True)
        plt.plot(orig_sig, 'b-*', fillstyle='none', label='Original')
        plt.plot(reconstruct_sig, 'r-.', label='Reconstruction')
        plt.ylim([-5, 5])
        plt.xlabel('Sample Index')
        plt.ylabel('Amplitude')
        plt.title('Reconstruction from Residual')
        plt.grid(True)
        plt.legend()
        plt.pause(0.5)
    
    # Play True and Predicted
    if is_hear_reconstruct and (ii in random_listens):
        time_play = 6.0  # seconds
        n_samps = int(time_play / samp_period)
        sd.play(orig_sig[:n_samps], samplerate=samp_rate)
        time.sleep(1.1*time_play)
        sd.play(reconstruct_sig[:n_samps], samplerate=samp_rate)
        time.sleep(1.1*time_play)

# Mean squared error per sample
lpc_base_error = squared_err / n_tot_samps  # Mean squared error per sample
print('\n# ******************')
print('# LPC nBits: ' + str(n_coeff_bits))
print('# Excitation nBits: ' + str(n_exc_bits))

print('# Squared Error: ' + str(squared_err))
print('# LPC Base Error: ' + str(lpc_base_error))

print('# Original Data Rate: ' + str(orig_data_rate) + ' bps')
print('# LPC Base Data Rate: ' + str(lpc_base_data_rate) + ' bps')

# %% RESULTS
# It appears reconstruction error > 1e-4 sounds bad
# USE: 12 bit for excitation
#      14 bit for LPCs? 

# Weird because lowering LPCs makes reconstruction error way worse but sounds fine still
# While decreasing excitation bits leaves reconstruction fine but sounds horrible

# ******************
# LPC nBits: 16
# Excitation nBits: 12
# Squared Error: 422.5252089736769
# LPC Base Error: 6.830434421616783e-05
# Original Data Rate: 256000 bps
# LPC Base Data Rate: 206000.0 bps

# ******************
# LPC nBits: 16
# Excitation nBits: 11
# Squared Error: 1546.358502166903
# LPC Base Error: 0.000249980358971164
# Original Data Rate: 256000 bps
# LPC Base Data Rate: 189500.0 bps

# # ******************
# # LPC nBits: 12
# # Excitation nBits: 12
# # Squared Error: 585977534.5314329
# # LPC Base Error: 94.72762895922237
# # Original Data Rate: 256000 bps
# # LPC Base Data Rate: 204000.0 bps

# ******************
# LPC nBits: 14
# Excitation nBits: 12
# Squared Error: 842.0827754085831
# LPC Base Error: 0.00013612894693248266
# Original Data Rate: 256000 bps
# LPC Base Data Rate: 205000.0 bps
