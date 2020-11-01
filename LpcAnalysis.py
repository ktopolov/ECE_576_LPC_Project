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
# path = 'C:\\Users\\ktopo\\Desktop\\School\\Courses\Masters\\ECE 576 - Information Engineering\\Project\\sample_voice_data-master\\sample_voice_data-master\\females'
path = 'C:\\Users\\ktopo\\Desktop\\School\\Courses\Masters\\ECE 576 - Information Engineering\\Project\\data\\timit'

fullfiles = []
for subdir, dirs, files in os.walk(path):
    [fullfiles.append(os.path.join(subdir, file)) for file in files if file.endswith('.wav')]

n_files = len(fullfiles)
random.shuffle(fullfiles)

# Configure LPC
#   `base` - base LPC using quantization on excitation signal
#   'parameterize' - parameterize the excitation using pitch period and reconstruct it at decoder
#   'vq' - vector quantization of excitation signal
#   'autoencoder' - Use autoencoder to encode and decode excitation signal
method = 'autoencoder'

LpcConfig = {'order': 10,
             'overlap_time': 10e-3,
             'frame_time': 20e-3}

TransmitData = {}
if method == 'base':
    TransmitData = {'lpccs': [], 'excitations': [], 'gains': []}
    QuantizeConfig = {'n_bit_lpccs': 14, 'n_bit_excitations': 6, 'n_bit_gains': 12}

elif method == 'vq':
    TransmitData = {'lpccs': [], 'vq_index': [], 'gains': []}
    QuantizeConfig = {'n_bit_lpccs': 14, 'n_bit_vq_index': 10, 'n_bit_gains': 12}

elif method == 'parameterize':
    TransmitData = {'lpccs': [], 'pitch_periods': [], 'is_voiceds': [], 'gains': []}
    QuantizeConfig = {'n_bit_lpccs': 14, 'n_bit_pitch_periods': 6, 'n_bit_is_voiceds': 1, 'n_bit_gains': 12}

elif method == 'autoencoder':
    TransmitData = {'lpccs': [], 'gains': []}
    QuantizeConfig = {'n_bit_lpccs': 14, 'n_bit_gains': 12, 'n_bit_encodings': 12}  # 12 bits used in GoogleColab script

# Original signal
n_bit_orig = 16
max_val = 0.8  # normalize audio -0.8 to 0.8

# Voiced/Unvoiced detection
z_cross_thresh = 0.25  # anything above is unvoiced

# %% Encoder
# Original data stored here
OrigData = {'signals': []}

# LPC parameters stored here
lpccs = []
excitations = []
gains = []
pitch_periods = []
is_voiceds = []

print('***************')
for ii, fullfile in enumerate(fullfiles):
    print('\rEncoding File ' + str(ii + 1) + '/' + str(n_files), end='')
    data, samp_rate = ap.read_wave(fullfile)
    max_mag = np.abs(data).max()
    data *= (max_val / max_mag)  # normalize to 0.8

    # Downsample to normal speech sample rate
    down_samp_factor = 2  # from 16lHz to 8kHz
    data = scisig.decimate(data, down_samp_factor)
    samp_rate /= down_samp_factor

    # Quantize original signal
    quant_idx, quant_levels = ap.quantize(array=data,
                                          vmin=-max_val,
                                          vmax=max_val,
                                          n_bits=n_bit_orig)
    data = quant_levels[quant_idx]
    OrigData['signals'].append(data)
    
    # Zero-crossings plot; Don't remove! Super helpful
    is_show = False
    if is_show:
        z_cross_rate = np.zeros(data.shape)
        energy = np.zeros(data.shape)
        window_size = 100  # samples
        end_idx = np.arange(start=window_size, stop=data.size, step=1)
        st_idx = end_idx - window_size
        mid_idx = ((end_idx + st_idx) // 2).astype(int)

        for st, mid, end in zip(st_idx, mid_idx, end_idx):
            audio_clip = data[st:end]
            z_cross_rate[mid] = ap.get_zero_cross_rate(audio_clip)
            energy[mid] = ap.get_energy(audio_clip)
        
        plt.figure(1, clear=True)
        plt.plot(data, 'k-', label='Speech Signal')
        plt.plot(z_cross_rate, 'r-', label='Zero-Crossing Rate')
        plt.plot(energy / window_size, 'c-', label='Energy per Sample')
        #plt.plot(np.array([0, data.size-1]), np.ones((2,)) * z_cross_thresh, 'g-*', label='Zero-Crossings Voiced/Unvoiced Threshold')
        plt.legend()
        plt.grid()
        plt.title('Speech Signal vs. Voiced Detector Parameters')
        plt.xlabel('Slow-time Index')
        plt.ylim((-1, 1))
        plt.pause(1.0)

    # Compute frame size
    samp_period = 1/samp_rate
    frame_len = int(LpcConfig['frame_time'] / samp_period)
    overlap_len = int(LpcConfig['overlap_time'] / samp_period)

    # Compute LPCs and excitation using baseline method
    lpcc, excitation, gain, pitch_period, is_voiced = ap.lpc(data=data,
                                                             lpc_order=LpcConfig['order'],
                                                             frame_len=frame_len,
                                                             overlap_len=overlap_len,
                                                             zero_cross_thresh=z_cross_thresh)

    # OUTPUTS
    lpccs.append(lpcc)
    excitations.append(excitation)
    gains.append(gain)
    pitch_periods.append(pitch_period)
    is_voiceds.append(is_voiced)

is_save = False
if is_save:
    np.save('excitations', np.concatenate(excitations, axis=0))
    
# %% Quantize LPC Parameters
# 1) Setup Quantization
# Excitations
if method == 'vq':
    pass
    excitations_array = np.concatenate(excitations, axis=0)
    vq_index = []
    n_vectors = 2**QuantizeConfig['n_bit_vq_index']  # num vectors in codebook; power of 2
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

elif method == 'base':
    excitations_array = np.concatenate(excitations, axis=0)
    excitation_min, excitation_max = excitations_array.min(), excitations_array.max()

elif method == 'parameterize':
    pitch_period_array = np.concatenate(pitch_periods, axis=0).flatten()
    pitch_period_min, pitch_period_max = pitch_period_array.min(), pitch_period_array.max()

# LPCCs
lpccs_array = np.concatenate(lpccs, axis=0).flatten()
lpcc_min, lpcc_max = lpccs_array.min(), lpccs_array.max()

# Gain
gain_array = np.concatenate(gains, axis=0).flatten()
gain_min, gain_max = gain_array.min(), gain_array.max()

# 2) Perform Quantization
for ii, (lpcc, gain, pitch_period, excitation) in enumerate(zip(lpccs, gains, pitch_periods, excitations)):
    print('\rQuantizing for signal - file %d/%d' % (ii + 1, n_files), end='')
    
    # LPCCs
    quant_idx, quant_levels = ap.quantize(array=lpcc, vmin=lpcc_min, vmax=lpcc_max, n_bits=QuantizeConfig['n_bit_lpccs'])
    lpccs[ii] = quant_levels[quant_idx]
    
    # Gains
    quant_idx, quant_levels = ap.quantize(array=gain, vmin=gain_min, vmax=gain_max, n_bits=QuantizeConfig['n_bit_gains'])
    gains[ii] = quant_levels[quant_idx]
    
    # Excitation Signals and Pitch Periods
    if method == 'base':
        quant_idx, quant_levels = ap.quantize(array=excitation, vmin=excitation_min, vmax=excitation_max, n_bits=QuantizeConfig['n_bit_excitations'])
        excitations[ii] = quant_levels[quant_idx]
    elif method == 'vq':
        closest_idx = ap.vector_quantize(vectors=excitation, codebook=codebook)
        vq_index.append(closest_idx)
    elif method == 'parameterize':
        # Pitch Periods
        quant_idx, quant_levels = ap.quantize(array=pitch_period, vmin=pitch_period_min, vmax=pitch_period_max, n_bits=QuantizeConfig['n_bit_pitch_periods'])
        pitch_periods[ii] = quant_levels[quant_idx]

# %% Quantize Residual/Excitation
TransmitData['lpccs'] = lpccs
TransmitData['gains'] = gains

if method == 'base':
    TransmitData['excitations'] = excitations
elif method == 'vq':
    TransmitData['vq_index'] = vq_index
elif method == 'parameterize':
    TransmitData['pitch_periods'] = pitch_periods
    TransmitData['is_voiceds'] = is_voiceds
elif method == 'autoencoder':
    encoding = np.load('encodings_post_quantize.npy')
    encodings = []
    ifr = 0
    for i_file in range(n_files):
        n_frames_in_file = lpccs[i_file].shape[0]
        encodings.append(encoding[ifr:ifr + n_frames_in_file, :])
        ifr += n_frames_in_file
    TransmitData['encodings'] = encodings

# %% Entropies
print('\n ****** ENTROPY ******')

# Original signal
unique_vals, rel_freqs = stats.get_rel_freqs(np.concatenate(OrigData['signals'], axis=0))
entropy = scistats.entropy(pk=rel_freqs, base=2)
print('\n%s\n-----------' % ('Original Signals'))
print('# Quantize Bits: %d\nEntropy: %.3f bits/sample' % (n_bit_orig, entropy))

plt.figure(9, clear=True)
plt.plot(unique_vals, rel_freqs)
plt.xlabel('Value')
plt.ylabel('Relative Frequency')
plt.title('Relative Frequencies of Original Signal Values')
plt.grid(True)

# Entropy of Everything in TransmitData
fig_start = 10
for ii, (key, val) in enumerate(TransmitData.items()):
    if val :  # if not empty list
        unique_vals, rel_freqs = stats.get_rel_freqs(np.concatenate(val, axis=0))
        entropy = scistats.entropy(pk=rel_freqs, base=2)
        n_quantize_bits = QuantizeConfig['n_bit_' + key]
        
        print('\n%s\n-----------' % (key))
        print('# Quantize Bits: %d\nEntropy: %.3f bits/sample' % (n_quantize_bits, entropy))
    
        plt.figure(fig_start + ii, clear=True)
        plt.plot(unique_vals, rel_freqs)
        plt.xlabel('Value')
        plt.ylabel('Relative Frequency')
        plt.title('Relative Frequencies of {} Values'.format(key))
        plt.grid(True)

# %% Decoder
ReconstructData = {'signals': []}

# Encoding/quantization/decoding done in Google Colab script
if method == 'autoencoder':
    decoding = np.load('decodings.npy')
    ifr = 0

for ii in range(n_files):
    lpccs = TransmitData['lpccs'][ii]
    gains = TransmitData['gains'][ii]

    if method == 'vq':
        closest_idx = TransmitData['vq_index'][ii]
        excitations = codebook[closest_idx, :]
            
    elif method == 'base':
        excitations = TransmitData['excitations'][ii]

    elif method == 'parameterize':
        # Reconstruction
        n_frame = lpccs.shape[0]
        n_samps_per_frame = frame_len + LpcConfig['order']
        pitch_periods = TransmitData['pitch_periods'][ii]
        is_voiceds = TransmitData['is_voiceds'][ii]
        excitations = np.zeros((n_frame, n_samps_per_frame))
        for jj, (period, is_voiced) in enumerate(zip(pitch_periods, is_voiceds)):
            if is_voiced:
                excitations[jj, :] = ap.gen_deltas(n_samps=n_samps_per_frame, period=period)
            else:
                excitations[jj, :] = np.random.randn(n_samps_per_frame)
    elif method == 'autoencoder':
        n_frames_in_file = lpccs.shape[0]
        excitations = decoding[ifr:ifr + n_frames_in_file, :]
        
        for f, gain in enumerate(gains):
            frame_gain = np.sqrt(ap.get_energy(excitations[f, :]))
            excitations[f, :] /= frame_gain
        ifr += n_frames_in_file
    
    reconstruct_sig = ap.reconstruct_lpc(exc_sig_per_frame=excitations,
                                         lpc_coeffs=lpccs,
                                         gain=gains,
                                         frame_len=frame_len,
                                         overlap_len=overlap_len)

    ReconstructData['signals'].append(reconstruct_sig.flatten())  # still zero-padded; must adjust to compare to original

# %% Analyze Reconstruction Error
is_hear_reconstruct = True  # listen to every Nth speaker reconstructed
is_show_reconstruct = False  # show reconstruction plot
n_listens = 1  # how many random reconstructions to play
random_listens = np.random.randint(low=0, high=n_files, size=(n_listens,))

reconstruct_error = []

for ii, (reconstruct_sig, orig_sig) in enumerate(zip(ReconstructData['signals'],
                                                     OrigData['signals'])):
    print('\rReconstructing Signal %d/%d' % (ii + 1, n_files), end='')
    # Align reconstructed data to the original
    # number of samples in front of orig signal that are not used in LPC
    n_front = np.ceil(overlap_len//2).astype(int) + LpcConfig['order']
    n_long = reconstruct_sig.shape[0]  # reconstruction stops short of end of data
    orig_sig = orig_sig[n_front:n_front + n_long]  
    reconstruct_error.append(orig_sig - reconstruct_sig)
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
        recording_length = n_long * samp_period
        time_play = 1.1 * np.minimum(recording_length, 5.0)
        n_samps = int(time_play / samp_period)
        sd.play(orig_sig[:n_samps], samplerate=samp_rate)
        time.sleep(1.1*time_play)
        sd.play(reconstruct_sig[:n_samps], samplerate=samp_rate)
        time.sleep(1.1*time_play)

mse = np.mean(np.concatenate(reconstruct_error, axis=0)**2)
print('\n\nReconstruction Mean Squared Error: %.3e' % (mse))
