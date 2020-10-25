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
import scipy.stats as scistats

# Internal
import audio_processing as ap
import stats as stats

# %% Read File
path = 'C:\\Users\\ktopo\\Desktop\\School\\Courses\\Masters\\ECE 576 - Information Engineering\\Project\\sample_voice_data-master\\sample_voice_data-master\\females'
files = os.listdir(path)
random.shuffle(files)

n_files = 51  # numebr of files to load
files = files[:n_files]
n_quantize_bits_orig = 16

# Configure LPC
lpc_order = 10  # decreasing will make excitation signal approach speech signal
frame_time = 20e-3
overlap_time = 10e-3
n_quantize_bits_lpcc = 14  # number of bits per LPCC
n_quantize_bits_exc = 12  # number of bits per LPC

# %% Encoder
# Data transmitted to decoder
OrigData = {'signals': []}
TransmitData = {'lpccs': [],
                'excitations': []}

print('***************')
for ii, filename in enumerate(files):
    print('\rEncoding File ' + str(ii + 1) + '/' + str(n_files), end='')
    file = os.path.join(path, filename)
    data, samp_rate = ap.read_wave(file)
    OrigData['signals'].append(data)

    # Compute frame size
    samp_period = 1/samp_rate
    frame_len = int(frame_time / samp_period)
    overlap_len = int(overlap_time / samp_period)

    # Compute LPCs and excitation using baseline method
    lpccs, excitation, _ = ap.lpc(data=data, lpc_order=lpc_order,
                                  frame_len=frame_len,
                                  overlap_len=overlap_len)

    # OUTPUTS
    TransmitData['lpccs'].append(lpccs)
    TransmitData['excitations'].append(excitation)

# %% Quantize LPCs and Residual
# Original data
orig_signals_array = np.concatenate(OrigData['signals'], axis=0)

# LPC data
lpccs_array = np.concatenate(TransmitData['lpccs'], axis=0).flatten()
excitations_array = np.concatenate(TransmitData['excitations'], axis=0).flatten()

# LPCs
coeff_max = lpccs_array.max()
coeff_min = lpccs_array.min()
coeff_levels = ap.get_uniform_quantize_levels(vmax=coeff_max,
                                              vmin=coeff_min,
                                              n_bits=n_quantize_bits_lpcc)

# Excitation
exc_max = excitations_array.max()
exc_min = excitations_array.min()
exc_levels = ap.get_uniform_quantize_levels(vmax=exc_max,
                                            vmin=exc_min,
                                            n_bits=n_quantize_bits_exc)

for ii, (coeffs, excitation) in enumerate(zip(TransmitData['lpccs'],
                                              TransmitData['excitations'])):
    print('\rQuantizing Signal ' + str(ii + 1) + '/' + str(n_files), end='')
    ap.uniform_quantize(array=coeffs, levels=coeff_levels)
    ap.uniform_quantize(array=excitation, levels=exc_levels)

# %% Compute Entropy; Determine Optimal Compression Rate
print('\n*** ENTROPIES ***')
# Original signal
unique_vals, rel_freqs = stats.get_rel_freqs(orig_signals_array)
n_zero = 2**n_quantize_bits_orig - rel_freqs.shape[0]  # values with 0 frequency
orig_entropy = scistats.entropy(pk=np.concatenate((rel_freqs, np.zeros((n_zero,))), axis=0))
print('Entropy of Original Speech Signal: %.3f bits/sample' % orig_entropy)

plt.figure(12, clear=True)
plt.plot(unique_vals, rel_freqs)
plt.xlabel('Value')
plt.ylabel('Relative Frequency')
plt.title('Relative Frequencies of Original Signal Values')
plt.grid(True)

# Excitations
excitations_array = np.concatenate(TransmitData['excitations'], axis=0)
unique_vals, rel_freqs = stats.get_rel_freqs(excitations_array)
n_zero = 2**n_quantize_bits_exc - rel_freqs.shape[0]  # values with 0 frequency
exc_entropy = scistats.entropy(pk=np.concatenate((rel_freqs, np.zeros((n_zero,))), axis=0))
print('Entropy of Excitation Signal: %.3f bits/sample' % exc_entropy)

plt.figure(13, clear=True)
plt.plot(unique_vals, rel_freqs)
plt.xlabel('Value')
plt.ylabel('Relative Frequency')
plt.title('Relative Frequencies of Quantized Excitation Signal Values')
plt.grid(True)

# LPCCs
lpccs_array = np.concatenate(TransmitData['lpccs'], axis=0).flatten()
unique_vals, rel_freqs = stats.get_rel_freqs(lpccs_array)
n_zero = 2**n_quantize_bits_lpcc - rel_freqs.shape[0]  # values with 0 frequency
lpcc_entropy = scistats.entropy(pk=np.concatenate((rel_freqs, np.zeros((n_zero,))), axis=0))
print('Entropy of LPCCs: %.3f bits/coeff' % lpcc_entropy)

plt.figure(14, clear=True)
plt.plot(unique_vals, rel_freqs)
plt.xlabel('Value')
plt.ylabel('Relative Frequency')
plt.title('Relative Frequencies of LPCC Values')
plt.grid(True)

# %% Decoder
ReconstructData = {'signals': []}

for ii, filename in enumerate(files):
    # Reconstruction
    reconstruct_sig = ap.reconstruct_lpc(exc_sig_per_frame=TransmitData['excitations'][ii],
                                         lpc_coeffs=TransmitData['lpccs'][ii],
                                         frame_len=frame_len,
                                         overlap_len=overlap_len)
    ReconstructData['signals'].append(reconstruct_sig.flatten())  # still zero-padded; must adjust to compare to original

# %% Analyze Reconstruction Error
is_hear_reconstruct = False  # listen to every Nth speaker reconstructed
is_show_reconstruct = False  # show reconstruction plot
n_listens = 1  # how many random reconstructions to play
random_listens = np.random.randint(low=0, high=n_files, size=(n_listens,))

n_tot_samps = 0  # can't pre-initialize since signals are variable length
squared_err = 0

for ii, (reconstruct_sig, orig_sig) in enumerate(zip(ReconstructData['signals'],
                                                     OrigData['signals'])):
    print('\rReconstructing Signal %d/%d' % (ii + 1, n_files), end='')
    # 1) Align reconstructed data to the original
    # number of samples in front of orig signal that are not used in LPC
    n_front = np.ceil(overlap_len//2).astype(int) + lpc_order
    n_long = reconstruct_sig.shape[0]  # reconstruction stops short of end of data
    orig_sig = orig_sig[n_front:n_front + n_long]
    
    n_tot_samps += n_long
    squared_err += np.sum((orig_sig - reconstruct_sig)**2)
    
    if is_show_reconstruct:
        plt.figure(6, clear=True)
        plt.plot(orig_sig, 'b-o', fillstyle='none', label='Original')
        plt.plot(reconstruct_sig, 'r-.', label='Reconstruction')
        plt.ylim([-2, 2])
        plt.xlabel('Sample Index')
        plt.ylabel('Amplitude')
        plt.title('Reconstruction from Residual, file#: %d/%d' % (ii, n_files))
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

print('Reconstruction Mean Squared Error: %.3e' % (squared_err/n_tot_samps))

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
