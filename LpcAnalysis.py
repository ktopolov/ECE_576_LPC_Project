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
from sklearn.cluster import KMeans
import scipy.signal as scisig

# Internal
import audio_processing as ap
import stats as stats

# %% Read File
path = 'C:\\Users\\Kenny\\Desktop\\School\\Courses\Masters\\ECE 576 - Information Engineering\\Project\\sample_voice_data-master\\sample_voice_data-master\\females'
files = os.listdir(path)
random.shuffle(files)

n_files = 51  # numebr of files to load
files = files[:n_files]
n_quantize_bits_orig = 16

# Configure LPC
method = 'base'  # 'vq' - vector quantization; `base` - base LPC using quantization on excitation signal

if method == 'base':
    LpcConfig = {'order': 10,
                 'overlap_time': 10e-3,
                 'frame_time': 20e-3,
                 'n_bits_lpcc': 14,
                 'n_bits_excite': 12,
                 'n_bits_gain': 12}
elif method == 'vq':
    LpcConfig = {'order': 10,
                 'overlap_time': 10e-3,
                 'frame_time': 20e-3,
                 'n_bits_lpcc': 14,
                 'n_bits_vq': 6,
                 'n_bits_gain': 12}

# %% Encoder
# Data transmitted to decoder
OrigData = {'signals': []}
TransmitData = {'lpccs': [],
                'excitations': [],
                'gains': [],
                'pitch_periods': []}

print('***************')
for ii, filename in enumerate(files):
    print('\rEncoding File ' + str(ii + 1) + '/' + str(n_files), end='')
    file = os.path.join(path, filename)
    data, samp_rate = ap.read_wave(file)

    # Debug (prove data is band-limited to 4kHz)
    # fft_data = np.fft.fftshift(np.fft.fft(data))
    # plt.figure(3, clear=True)
    # freq_axis = np.fft.fftshift(np.fft.fftfreq(fft_data.size, d=1/samp_rate))
    # plt.plot(freq_axis, fft_data)
    # plt.xlabel('Frequency (Hz)')
    # plt.ylabel('Magnitude (dB)')

    # Downsample to normal speech sample rate
    down_samp_factor = 2  # from 16lHz to 8kHz
    data = scisig.decimate(data, down_samp_factor)
    samp_rate /= down_samp_factor

    OrigData['signals'].append(data)

    # Compute frame size
    samp_period = 1/samp_rate
    frame_len = int(LpcConfig['frame_time'] / samp_period)
    overlap_len = int(LpcConfig['overlap_time'] / samp_period)

    # Compute LPCs and excitation using baseline method
    lpccs, excitation, gain, pitch_period = ap.lpc(data=data,
                                                   lpc_order=LpcConfig['order'],
                                                   frame_len=frame_len,
                                                   overlap_len=overlap_len)

    # OUTPUTS
    TransmitData['lpccs'].append(lpccs)
    TransmitData['excitations'].append(excitation)
    TransmitData['gains'].append(gain)
    TransmitData['pitch_periods'].append(pitch_period)

is_save = True
if is_save:
    np.save('excitations', np.concatenate(TransmitData['excitations'], axis=0))
    
# %% Quantize Original Signal
# Must re-quantize since we downsampled
orig_signals_array = np.concatenate(OrigData['signals'], axis=0).flatten()
vmin, vmax = orig_signals_array.min(), orig_signals_array.max()

for ii, signal in enumerate(OrigData['signals']):
    print('\rQuantizing original signal - file %d/%d' % (ii + 1, n_files), end='')
    quant_idx, quant_levels = ap.quantize(array=signal, vmin=vmin, vmax=vmax, n_bits=n_quantize_bits_orig)
    OrigData['signals'][ii] = quant_levels[quant_idx]

# %% Quantize LPCCs
lpccs_array = np.concatenate(TransmitData['lpccs'], axis=0).flatten()
vmin, vmax = lpccs_array.min(), lpccs_array.max()

for ii, lpcc in enumerate(TransmitData['lpccs']):
    print('\rQuantizing LPCCs signal - file %d/%d' % (ii + 1, n_files), end='')
    quant_idx, quant_levels = ap.quantize(array=lpcc, vmin=vmin, vmax=vmax, n_bits=LpcConfig['n_bits_lpcc'])
    TransmitData['lpccs'][ii] = quant_levels[quant_idx]

# %% Quantize Gain
gain_array = np.concatenate(TransmitData['gains'], axis=0).flatten()
vmin, vmax = gain_array.min(), gain_array.max()

for ii, gain in enumerate(TransmitData['gains']):
    print('\rQuantizing Gain - file %d/%d' % (ii + 1, n_files), end='')
    quant_idx, quant_levels = ap.quantize(array=gain, vmin=vmin, vmax=vmax, n_bits=LpcConfig['n_bits_gain'])
    TransmitData['gains'][ii] = quant_levels[quant_idx]

# %% Quantize Residual/Excitation
if method == 'base':
    excitations_array = np.concatenate(TransmitData['excitations'], axis=0).flatten()
    vmin, vmax = excitations_array.min(), excitations_array.max()

    for ii, excitation in enumerate(TransmitData['excitations']):
        print('\rQuantizing excitation signal - file %d/%d' % (ii + 1, n_files), end='')
        quant_idx, quant_levels = ap.quantize(array=excitation, vmin=vmin, vmax=vmax, n_bits=LpcConfig['n_bits_excite'])
        TransmitData['excitations'][ii] = quant_levels[quant_idx]

elif method == 'vq':
    excitations_array = np.concatenate(TransmitData['excitations'], axis=0)
    n_vectors = 2**LpcConfig['n_bits_vq']  # num vectors in codebook; power of 2
    codebook_vec_count = np.zeros((n_vectors,))

    is_cluster = True
    if is_cluster:
        print('\nClustering to find codebook vectors...')
        kmeans = KMeans(n_clusters=n_vectors).fit(excitations_array)
        codebook = kmeans.cluster_centers_  # [n_vectors, lpc_order + frame_len]
        print('Done clustering')
    
        excite_fft = np.abs(np.fft.fft(excitations_array, axis=1))
        kmeans = KMeans(n_clusters=n_vectors).fit(excite_fft)
        codebook = np.real(np.fft.ifft(kmeans.cluster_centers_, axis=1))  # [n_vectors, lpc_order + frame_len]
    else:
        # Select random excitation  frames from dataset for codebook
        rand_idx = np.random.randint(low=0, high=excitations_array.shape[0], size=(n_vectors,))
        codebook = excitations_array[rand_idx, :]
    
    # is_debug = False
    # if is_debug:
    #     plt.figure(1, clear=True)
    #     i_exc = 20
    #     plt.subplot(2, 2, 1)
    #     plt.plot(excite_fft[i_exc, :])
    #     plt.xlabel('Sample Index')
    #     plt.ylabel('Amplitude')
    #     plt.title('Excitation Signal %d/%d in Frequency Domain' % (i_exc, n_files))
    #     plt.grid(True)
        
    #     plt.subplot(2, 2, 2)
    #     i_code = 0
    #     plt.plot(codebook[i_code, :])
    #     plt.xlabel('Sample Index')
    #     plt.ylabel('Amplitude')
    #     plt.title('Codebook Signal %d/%d in Frequency Domain' % (i_exc, n_files))
    #     plt.grid(True)
        
    #     plt.subplot(2, 2, 3)
    #     plt.plot(excitations_array[i_exc, :])
    #     plt.xlabel('Sample Index')
    #     plt.ylabel('Amplitude')
    #     plt.title('Excitation Signal %d/%d in Time Domain' % (i_exc, n_files))
    #     plt.grid(True)
        
    #     plt.subplot(2, 2, 4)
    #     plt.plot(kmeans.cluster_centers_[i_code, :])
    #     plt.xlabel('Sample Index')
    #     plt.ylabel('Amplitude')
    #     plt.title('Codebook Signal %d/%d in Time Domain' % (i_exc, n_files))
    #     plt.grid(True)

    # Plot Codebook signals
    is_debug = False
    if is_debug:
        i_vec = 5
        plt.figure(1, clear=True)
        plt.plot(codebook[i_vec, :])
        plt.xlabel('Sample Index')
        plt.ylabel('Value')
        plt.title('Codebook Excitation Vector #%d' % (i_vec))
        plt.grid(True)

    for ii in range(n_files):
        print('\rQuantizing excitation signal - file %d/%d' % (ii + 1, n_files), end='')
        closest_idx = ap.vector_quantize(vectors=TransmitData['excitations'][ii], codebook=codebook)
        TransmitData['excitations'][ii] = codebook[closest_idx, :]

        # Count how many times each vector is used
        for jj in range(n_vectors):
            codebook_vec_count[jj] += (closest_idx == jj).sum()

# %% Entropy of Original Signal
# Original signal
unique_vals, rel_freqs = stats.get_rel_freqs(np.concatenate(OrigData['signals'], axis=0))
n_levels = 2**n_quantize_bits_orig
n_zero = n_levels - rel_freqs.shape[0]  # values with 0 frequency
orig_entropy = scistats.entropy(pk=np.concatenate((rel_freqs, np.zeros((n_zero,))), axis=0))
print('\n=== Original Speech Signal ===')
print('Original # Bits: %d\tEntropy: %.3f bits/sample' % (n_quantize_bits_orig, orig_entropy))

plt.figure(12, clear=True)
plt.plot(unique_vals, rel_freqs)
plt.xlabel('Value')
plt.ylabel('Relative Frequency')
plt.title('Relative Frequencies of Original Signal Values')
plt.grid(True)

# %% Entropy of Excitations
if method == 'base':
    unique_vals, rel_freqs = stats.get_rel_freqs(np.concatenate(TransmitData['excitations'], axis=0))
    n_levels = 2**LpcConfig['n_bits_excite']
    n_zero = n_levels - rel_freqs.shape[0]  # values with 0 frequency
    orig_entropy = scistats.entropy(pk=np.concatenate((rel_freqs, np.zeros((n_zero,))), axis=0))
    print('\n=== Excitation Signal ===')
    print('Original # Bits: %d\tEntropy: %.3f bits/sample' % (n_quantize_bits_orig, orig_entropy))

    plt.figure(13, clear=True)
    plt.plot(unique_vals, rel_freqs)
    plt.xlabel('Value')
    plt.ylabel('Relative Frequency')
    plt.title('Relative Frequencies of Excitation Signal Values')
    plt.grid(True)

elif method == 'vq':
    # Relative frequencies of codebook vectors
    n_frames = excitations_array.shape[0]
    codebook_rel_freqs = codebook_vec_count / n_frames
    vq_entropy = scistats.entropy(pk=codebook_rel_freqs)
    print('\n=== Vector Quantized Index ===')
    print('Original # Codebook Bits: %d\tEntropy: %.3f bits/sample' % (LpcConfig['n_bits_vq'], vq_entropy))

    plt.figure(13, clear=True)
    sort_idx = np.argsort(codebook_rel_freqs)
    plt.plot(codebook_rel_freqs[sort_idx])
    plt.ylabel('Relative Frequency')
    plt.title('Sorted Relative Frequencies of Codebook Vectors')
    plt.grid(True)

# %% Entropy of LPCCs
unique_vals, rel_freqs = stats.get_rel_freqs(np.concatenate(TransmitData['lpccs'], axis=0))
n_zero = 2**LpcConfig['n_bits_lpcc'] - rel_freqs.shape[0]  # values with 0 frequency
lpcc_entropy = scistats.entropy(pk=np.concatenate((rel_freqs, np.zeros((n_zero,))), axis=0))
print('\n=== LPCCs ===')
print('Original LPCC # Bits: %d\tEntropy: %.3f bits/sample' % (LpcConfig['n_bits_lpcc'], lpcc_entropy))

plt.figure(14, clear=True)
plt.plot(unique_vals, rel_freqs)
plt.xlabel('Value')
plt.ylabel('Relative Frequency')
plt.title('Relative Frequencies of LPCC Values')
plt.grid(True)

# %% Entropy of Gains
gain_array = np.concatenate(TransmitData['gains'], axis=0).flatten()
unique_vals, rel_freqs = stats.get_rel_freqs(gain_array)
n_zero = 2**LpcConfig['n_bits_gain'] - rel_freqs.shape[0]  # values with 0 frequency
gain_entropy = scistats.entropy(pk=np.concatenate((rel_freqs, np.zeros((n_zero,))), axis=0))
print('\n=== Gain ===')
print('Original Gain # Bits: %d\tEntropy: %.3f bits/sample' % (LpcConfig['n_bits_gain'], gain_entropy))

plt.figure(15, clear=True)
plt.plot(unique_vals, rel_freqs)
plt.xlabel('Value')
plt.ylabel('Relative Frequency')
plt.title('Relative Frequencies of Gain Values')
plt.grid(True)

# %% Decoder
ReconstructData = {'signals': []}

for ii, filename in enumerate(files):
    lpccs = TransmitData['lpccs'][ii]
    gains = TransmitData['gains'][ii]

    is_gen_excitation = True
    if is_gen_excitation:
        # Reconstruction
        n_frame = lpccs.shape[0]
        n_samps_per_frame = frame_len + LpcConfig['order']
        pitch_periods = TransmitData['pitch_periods'][ii]
        excitations = np.zeros((n_frame, n_samps_per_frame))
        scale = 2
        for ii, period in enumerate(pitch_periods):
            excitations[ii, :] = scale * ap.gen_deltas(n_samps=n_samps_per_frame, period=period*4)
    else:
        excitations = TransmitData['excitations'][ii]
    
    reconstruct_sig = ap.reconstruct_lpc(exc_sig_per_frame=excitations,
                                         lpc_coeffs=lpccs,
                                         gain=gains,
                                         frame_len=frame_len,
                                         overlap_len=overlap_len)
    # n_frames, n_samps_per_frame = TransmitData['excitations'][ii].shape
    # delta_sig = 0.2 * gen_deltas(n_samps=n_samps_per_frame, period=40)
    # delta_sig = np.repeat(delta_sig[np.newaxis, :], axis=0, repeats=n_frames)
    
    # reconstruct_sig = ap.reconstruct_lpc(exc_sig_per_frame=delta_sig,
    #                                      lpc_coeffs=TransmitData['lpccs'][ii],
    #                                      frame_len=frame_len,
    #                                      overlap_len=overlap_len)
    ReconstructData['signals'].append(reconstruct_sig.flatten())  # still zero-padded; must adjust to compare to original

# %% Analyze Reconstruction Error
is_hear_reconstruct = True  # listen to every Nth speaker reconstructed
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
    n_front = np.ceil(overlap_len//2).astype(int) + LpcConfig['order']
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

print('\n\nReconstruction Mean Squared Error: %.3e' % (squared_err/n_tot_samps))
