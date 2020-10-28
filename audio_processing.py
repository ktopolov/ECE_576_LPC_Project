# -*- coding: utf-8 -*-
"""
Functions to assist with speech and other audio processing
"""
# External
import numpy as np
import wave
import scipy.signal as sig
import scipy.linalg as scilin
import matplotlib.pyplot as plt

# %% Functions
def frame_data(array, samp_rate, frame_len):
    """
    Takes array of sound data and frames it into blocks
    
    Parameters
    ----------
    array : [N,], decimal
        Array of sound data

    samp_rate : decimal, scalar
        Sample rate in Hz

    frame_len : decimal, scalar
        Desired length of frames, in seconds

    Returns
    -------
    framed_data : [n_frame, n_samps]
        Data framed into n_frame, each containing n_samps. The last frame may be zero-padded
    """
    n_samps = array.shape[0]
    samp_period = 1/samp_rate
    speech_time = n_samps * samp_period

    n_frames = np.ceil(speech_time / frame_len).astype(int)
    n_samps_per_frame = np.ceil(frame_len / samp_period).astype(int)
    n_pad_samps = (n_frames * n_samps_per_frame) - n_samps
    
    # Pad data and reshape it
    framed_data = np.concatenate((array, np.zeros(n_pad_samps,)), axis=0)
    framed_data = framed_data.reshape((n_frames, n_samps_per_frame))

    return framed_data


def read_wave(filename):
    """
    Read WAV file into numpy array normalized from -1.0 to 1.0
    
    Parameters
    ----------
    filename : str
        Name (and path) of file

    Returns
    -------
    data : decimal, [N,]
        Speech data

    samp_rate : decimal, scalar
        Sampling rate
    """
    # Read file to get buffer                                                                                               
    Wav = wave.open(filename)
    samp_rate = Wav.getframerate()
    samples = Wav.getnframes()
    audio = Wav.readframes(samples)  # in bytes
    
    # Convert buffer to float32 using NumPy                                                                                 
    audio_as_np_int16 = np.frombuffer(audio, dtype=np.int16)
    audio_as_np_float32 = audio_as_np_int16.astype(np.float32)
    
    # Normalise float32 array so that values are between -1.0 and +1.0                                                      
    max_int16 = 2**15
    data = audio_as_np_float32 / max_int16
    return data, samp_rate

# %% LPC
def lpc(data, frame_len, overlap_len, lpc_order=15, zero_cross_thresh=0.25):
    """
    Perform Linear Predictive Coding
    
    Parameters
    ----------
    data : decimal, [N,]
        Speech audio data for a single channel

    frame_len : int, scalar
        Frame length, in samples (non-overlapping piece)

    overlap_len : int, scalar
        Number of samples overlapping from consecutive windows

    lpc_order : int, scalar
        Order of the LPC filter to fit; # filter coefficients = lpc_order + 1

    zero_cross_thresh : decimal, scalar
        Zero-crossing rate (per-sample) threshold for voiced/unvoiced detection.
        If above this threshold, speech is unvoiced; otherwise, voiced

    Returns
    -------
    coeffs : decimal, [n_frames, lpc_order+1]
        LPC coefficients

    excitation_signal : [N,]
        Residual error signal from LPC; first lpc_order+1 values will be
        garbage. M is the number of sampler per frame. Use np.flatten() to
        view as array

    gain : decimal, [n_frames,]
        Gain of the filter

    pitch_period : decimal, [n_frames,]
        Pitch period, in samples
    """
    n_samps = data.shape[0]  
    n_samps_fit = int(frame_len + overlap_len + lpc_order)  # number total samples used for fitting LPCCs

    # Get overlapping (fit) and non-overlapping data
    fit_end_vec = np.arange(start=n_samps_fit, stop=n_samps, step=frame_len, dtype=int)  # end of data for fitting autocorr and LPCs
    fit_st_vec = fit_end_vec - n_samps_fit  # first one should be 0 if done right
    frame_st_vec = fit_st_vec + np.ceil(overlap_len//2).astype(int) + lpc_order
    frame_end_vec = frame_st_vec + frame_len
    
    n_frames = fit_end_vec.shape[0]
    coeffs = np.zeros((n_frames, lpc_order))

    exc_sig_per_frame = np.zeros((n_frames, frame_len + lpc_order))
    gain = np.zeros((n_frames,))
    pitch_period = np.zeros((n_frames,))
    is_voiced = np.zeros((n_frames,), dtype=bool)

    for ifr in range(n_frames):
        # Get data indices for fitting
        fit_st = fit_st_vec[ifr]
        fit_end = fit_end_vec[ifr]
        fit_data = data[fit_st:fit_end]

        # %% Get Autocorrelation and solve for LPCs
        corr_half_idx = fit_data.shape[0] // 2
        corr = sig.correlate(in1=fit_data,
                             in2=fit_data,
                             mode='same')[corr_half_idx:]
        c = corr[:lpc_order]  # col and row of autocorr toeplitz matrix
        b = corr[1:lpc_order+1]

        # Compute zero-crossings; TODO-USE!
        zero_cross_rate = get_zero_cross_rate(fit_data)
        is_voiced[ifr] = zero_cross_rate < zero_cross_thresh
        
        min_period = 4  # 4kHz ish
        # max_period = 
        pitch_period[ifr] = (np.argmax(corr[min_period:]) + min_period)  # in samples
        
        # Use pseudo-inverse since solve_toeplitz becomes singular
        # Using Levinson-Durbin solving got singular matrix error
        auto_corr = scilin.toeplitz(c)
        auto_corr_inv = np.linalg.pinv(auto_corr)
        x = np.matmul(auto_corr_inv, b)
        coeffs[ifr, :] = x  # -a1 ... -aN in vocal tract all-pole filter

        # %% Inverse filter or residual for excitation signal
        frame_st = frame_st_vec[ifr]
        frame_end = frame_end_vec[ifr]

        s = data[frame_st - lpc_order:frame_end]  # True speech signal for frame (take lpc_order extra for prediction)
        num = np.concatenate((np.array([0]), x), axis=0) # y(n) = 0x(n) + a1x(n-1) + a2x(n-2) + ...
        den = np.array([1])
        s_hat = sig.lfilter(b=num, a=den, x=s)

        exc_sig = s - s_hat
        
        # %% Get gain and pitch? TODO
        n_exc_samps = exc_sig.shape[0]
        corr_half_idx = n_exc_samps // 2
        exc_corr = sig.correlate(in1=exc_sig, in2=exc_sig, mode='same')[corr_half_idx:]  # from t=0 onward by 1 sample step
        t_axis = np.arange(n_exc_samps // 2)
        
        frame_gain = np.sqrt(get_energy(exc_sig))
        exc_sig /= frame_gain
        
        exc_sig_per_frame[ifr, :] = exc_sig
        gain[ifr] = frame_gain
        

        # %% Debug
        is_debug = False
        if is_debug:
            plt.figure(num=4, clear=True)
            plt.plot(t_axis, exc_corr)
            plt.title('Autocorrelation of excitation signal to find pitch period')
            plt.xlabel('Lag (in samples)')  # take this and multiply by samp_period to get fundamental period in seconds
            plt.ylabel('Value')
            plt.show()
            plt.pause(1.0)

        # plot prediction
        is_debug = False
        if is_debug:
            plt.figure(1, clear=True)
            plt.plot(s_hat, 'r-o', label='Reconstruction', fillstyle='none')
            plt.plot(s, 'b-.', label='Actual')
            plt.xlabel('Sample Index')
            plt.ylabel('Voltage')
            plt.title('Reconstruction vs. Actual - Frame ' + str(ifr))
            plt.grid(True)
            plt.legend()
            plt.show()
            plt.pause(1.0)
        
        # plot residual
        is_debug = False
        if is_debug:
            plt.figure(2, clear=True)
            plt.plot(exc_sig, 'r-', label='Residual')
            plt.xlabel('Sample Index')
            plt.ylabel('Voltage')
            plt.title('Residual - Frame ' + str(ifr))
            plt.grid(True)
            plt.legend()
            plt.show()
            plt.pause(1.0)

        # plot residual passing through vocal tract filter
        is_debug = False
        if is_debug:
            # first lpc_order predictions are not good since insufficient data
            vocal_filt_num = np.array([1])
            vocal_filt_den = np.concatenate((np.array([1]), -x), axis=0)
            reconstruct = sig.lfilter(b=vocal_filt_num, a=vocal_filt_den, x=exc_sig)

            plt.figure(3, clear=True)
            plt.plot(reconstruct[lpc_order:], 'r-o', label='Reconstruction', fillstyle='none')
            plt.plot(s[lpc_order:], 'b-.', label='Actual')
            plt.xlabel('Sample Index')
            plt.ylabel('Voltage')
            plt.title('Reconstruction vs. Actual - Frame ' + str(ifr))
            plt.grid(True)
            plt.legend()
            plt.show()
            plt.pause(1.0)

    return coeffs, exc_sig_per_frame, gain, pitch_period, is_voiced


def reconstruct_lpc(exc_sig_per_frame, lpc_coeffs, gain, frame_len, overlap_len):
    """
    Given LPC coefficients and excitation signal, reconstruct speech
    
    Parameters
    ----------
    exc_sig_per_frame : decimal, [n_frames, lpc_order + frame_len]
        Excitation signal

    coeffs : decimal, [n_frames, lpc_order + 1]
        LPC coefficients

    gain : decimal, [n_frames,]
        Gain of the LPC filter

    frame_len : int, scalar
        Frame length, in samples (non-overlapping piece)

    overlap_len : int, scalar
        Number of samples overlapping from consecutive windows

    Returns
    -------
    reconstruct_sig : decimal, [n_samps,]
        Reconstructed signal
    """
    # Must match between this and the LPC function
    n_frames, lpc_order = lpc_coeffs.shape
    n_samp_per_frame = exc_sig_per_frame.shape[1] - lpc_order
    reconstruct_sig = np.zeros((n_frames, n_samp_per_frame))

    for ifr in range(n_frames):
        # Filter excitation signal with vocal tract filter
        coeffs = lpc_coeffs[ifr, :]
        vocal_filt_den = np.concatenate((np.array([1]), -coeffs), axis=0)
        vocal_filt_num = np.array([1])
        
        # FIX: Choose only one of these
        is_use_prev_frame = False  # doesn't work great; static noise
        if is_use_prev_frame and ifr > 0:
            excitation = gain[ifr] * np.concatenate((exc_sig_per_frame[ifr-1, -lpc_order:],
                                                     exc_sig_per_frame[ifr, lpc_order:]), axis=0)
        else:
            excitation = gain[ifr] * exc_sig_per_frame[ifr, :]

        # TODO-KT: do i need to take lpc_order extra samples of exc_sig?
        reconstruct = sig.lfilter(b=vocal_filt_num, a=vocal_filt_den, x=excitation)
        reconstruct_sig[ifr, :] = reconstruct[lpc_order:]
    return reconstruct_sig


def quantize(array, vmin, vmax, n_bits):
    """
    Quantize signal with uniform quantization
    
    Parameters
    ----------
    array : decimal, [N, M, ...]
        Input data array

    vmin : decimal, scalar
        Minumum quantization reference

    vmax : decimal, scalar
        Maxumum quantization reference

    n_bits : int, scalar
        Number of bits used for quantization

    Returns
    -------
    quant_idx : int, [N, M, ...]
        Index of the quantization level, from [0, n_levels-1]

    quant_levels : decimal, [2**n_bits,]
        Quantization level values
    """
    step = (vmax - vmin) / 2**n_bits
    quant_idx = np.floor((array - vmin) / step)
    quant_idx = np.maximum(np.minimum(quant_idx, 2**n_bits-1), 0).astype(int)
    quant_levels = np.arange(start=vmin, stop=vmax, step=step)
    return quant_idx, quant_levels

def vector_quantize(vectors, codebook):
    """
    Quantize vectors using the closest vector in codebook using L2
    
    Parameters
    ----------
    vectors : decimal, [n_data, n_features]
        Input vectors to be quantized; this is directly modified

    codebook : decimal, [n_vectors, n_features]
        Codebook of vectors

    Returns
    -------
    closest_vec_idx : int, [n_data,]
        Index of the closest bector in the codebook to the input example
    """
    dist_matrix = np.linalg.norm(vectors[:, np.newaxis, :] - codebook[np.newaxis, :, :], axis=2)
    arg_min = np.argmin(dist_matrix, axis=1)
    return arg_min

def is_voiced(signal):
    """
    Detect whether signal is voiced

    Parameters
    ----------
    signal : decimal, [N,]
        Audio signal

    Returns
    -------
    is_voiced : bool, scalar
        True if input signal is voiced, false if unvoiced
    """
    corr_half_idx = signal.shape[0] // 2
    corr = sig.correlate(in1=signal,
                         in2=signal,
                         mode='same')[corr_half_idx:]
    plt.figure(1, clear=True)
    plt.plot(corr)
    plt.xlabel('Samples')
    plt.ylabel('Correlation Value')
    plt.title('Autocorrelation')
    
def get_zero_cross_rate(signal):
    """
    Compute zero crossings rate (zero crossings per sample)

    Parameters
    ----------
    signal : decimal, [N,]
        Audio signal

    Returns
    -------
    zero_cross_rate : decimal, scalar
        Zero crossings rate (crossings per sample)
    """
    sgn = np.ones(signal.shape)
    sgn[signal < 0] = -1
    diff_sgn = np.diff(sgn)
    n_zero_crossings = (diff_sgn != 0).sum()
    zero_cross_rate = n_zero_crossings / diff_sgn.size
    return zero_cross_rate

def get_energy(signal):
    """
    Compute energy

    Parameters
    ----------
    signal : decimal, [N,]
        Audio signal

    Returns
    -------
    energy : decimal, scalar
        Energy of signal in Joules
    """
    energy = np.dot(signal, signal)
    return energy

def gen_deltas(n_samps, period):
    """
    Generate Dirac Delta pulse train with given period

    Parameters
    ----------
    n_samps : int, scalar
        Number of samples long

    period : int, scalar
        Period of the impulses, in samples

    Returns
    -------
    pulse_train : decimal, [n_samps,]
        Pulse train
    """
    sig = np.zeros((n_samps,))
    sig[np.arange(start=period//2, stop=n_samps, step=period, dtype=int)] = 1
    return sig

# sig = gen_deltas(n_samps=clip.size, period=pitch_period)
# plt.figure(6, clear=True)
# plt.plot(sig)
# plt.title('Impulse Signal')
# plt.grid(True)

