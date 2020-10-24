# -*- coding: utf-8 -*-
"""
Functions to assist with speech and other audio processing

@author: ktopo
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
def lpc(data, frame_len, overlap_len, lpc_order=15):
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

    method : str
        Method to obtain excitation
            'inv_filter': use inverse of estimated vocal tract filter to obtain exact residual to reconstruct signal
            'residual': use the estimated vocal tract to predict the signal outcome, then take difference between true and estimated s(n)

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
    """
    n_samps = data.shape[0]
    n_frames = (n_samps - (lpc_order + overlap_len)) // frame_len
    coeffs = np.zeros((n_frames, lpc_order))
    excitation_signal = np.zeros((data.shape))
    
    exc_sig_per_frame = np.zeros((n_frames, frame_len + lpc_order))
    gain = np.zeros((n_frames,))
  
    # Indices: overap frames to fit LPCs but only use computation for non-overlapping portion
    n_samps_fit = int(frame_len + overlap_len + lpc_order)

    # Data lengths is overlap_len + lpc_order + frame_len
    fit_end_vec = np.arange(start=n_samps_fit, stop=n_samps, step=frame_len, dtype=int)  # end of data for fitting autocorr and LPCs
    fit_st_vec = fit_end_vec - n_samps_fit  # first one should be 0 if done right
    frame_st_vec = fit_st_vec + np.ceil(overlap_len//2).astype(int) + lpc_order
    frame_end_vec = frame_st_vec + frame_len

    for ifr in range(n_frames):
        # Get data indices for fitting
        fit_st = fit_st_vec[ifr]
        fit_end = fit_end_vec[ifr]
        fit_data = data[fit_st:fit_end]

        # %% Get Autocorrelation and solve for LPCs
        # Window, get autocorr matrix and solve for coeffs
        # fit_data *= sig.windows.hamming(n_samps_fit)

        corr_half_idx = fit_data.shape[0] // 2
        corr = sig.correlate(in1=fit_data,
                             in2=fit_data,
                             mode='same')[corr_half_idx:]
        c = corr[:lpc_order]  # col and row of autocorr toeplitz matrix
        r = c
        b = corr[1:lpc_order+1]

        # Method 1) Use pseudo-inverse since solve_toeplitz becomes singular
        auto_corr = scilin.toeplitz(c)
        auto_corr_inv = np.linalg.pinv(auto_corr)
        x = np.matmul(auto_corr_inv, b)
        
        # Method 2) Levinson Durbin algorithm; errors out and says data singular
        # x = scilin.solve_toeplitz((c, r), b)

        coeffs[ifr, :] = x  # -a1 ... -aN in vocal tract all-pole filter

        # %% Inverse filter or residual for excitation signal
        frame_st = frame_st_vec[ifr]
        frame_end = frame_end_vec[ifr]

        s = data[frame_st - lpc_order:frame_end]  # True speech signal for frame (take lpc_order extra for prediction)
        # num = np.flip(np.concatenate((np.array([0]), x), axis=0), axis=0) # y(n) = 0x(n) + a1x(n-1) + a2x(n-2) + ...
        num = np.concatenate((np.array([0]), x), axis=0) # y(n) = 0x(n) + a1x(n-1) + a2x(n-2) + ...
        den = np.array([1])
        s_hat = sig.lfilter(b=num, a=den, x=s)

        # Debug
        exc_sig_per_frame[ifr, :] = s - s_hat

        # first lpc_order predictions are not good since insufficient data
        s_hat = s_hat[lpc_order:]
        s = s[lpc_order:]
        exc_sig = s - s_hat
        
        excitation_signal[frame_st:frame_end] = exc_sig
        
        # %% Get gain and pitch? TODO
        n_exc_samps = exc_sig.shape[0]
        corr_half_idx = n_exc_samps // 2
        exc_corr = sig.correlate(in1=exc_sig, in2=exc_sig, mode='same')[corr_half_idx:]  # from t=0 onward by 1 sample step
        t_axis = np.arange(n_exc_samps // 2)
        
        gain = np.linalg.norm(exc_corr, axis=0)
        
        is_debug = False
        if is_debug:
            plt.figure(num=4, clear=True)
            plt.plot(t_axis, exc_corr)
            plt.title('Autocorrelation of excitation signal to find pitch period')
            plt.xlabel('Lag (in samples)')  # take this and multiply by samp_period to get fundamental period in seconds
            plt.ylabel('Value')
            plt.show()
            plt.pause(1.0)
        
        # %% Debug
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
            vocal_filt_num = np.array([1])
            vocal_filt_den = np.concatenate((np.array([1]), -x), axis=0)
            
            # TODO-KT: do i need to take lpc_order extra samples of exc_sig?
            reconstruct = sig.lfilter(b=vocal_filt_num, a=vocal_filt_den, x=exc_sig_per_frame[ifr, :])
            reconstruct = reconstruct[lpc_order:]

            plt.figure(3, clear=True)
            plt.plot(reconstruct, 'r-o', label='Reconstruction', fillstyle='none')
            plt.plot(s, 'b-.', label='Actual')
            plt.xlabel('Sample Index')
            plt.ylabel('Voltage')
            plt.title('Reconstruction vs. Actual - Frame ' + str(ifr))
            plt.grid(True)
            plt.legend()
            plt.show()
            plt.pause(1.0)

    return coeffs, exc_sig_per_frame, gain


def reconstruct_lpc(exc_sig_per_frame, lpc_coeffs, frame_len, overlap_len):
    """
    Given LPC coefficients and excitation signal, reconstruct speech
    
    Parameters
    ----------
    exc_sig_per_frame : decimal, [n_frames, lpc_order + frame_len]
        Excitation signal

    coeffs : decimal, [n_frames, lpc_order + 1]
        LPC coefficients

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
    #step = frame_len
    #excess_start = int((lpc_order + 1) + overlap_len//2)

    for ifr in range(n_frames):
        # Indices of data
        #val_st = ifr * step + excess_start
        #val_end = val_st + frame_len
        
        # Filter excitation signal with vocal tract filter
        coeffs = lpc_coeffs[ifr, :]
        vocal_filt_den = np.concatenate((np.array([1]), -coeffs), axis=0)
        vocal_filt_num = np.array([1])

        # TODO-KT: do i need to take lpc_order extra samples of exc_sig?
        reconstruct = sig.lfilter(b=vocal_filt_num, a=vocal_filt_den, x=exc_sig_per_frame[ifr, :])
        reconstruct_sig[ifr, :] = reconstruct[lpc_order:]
    return reconstruct_sig

def uniform_quantize(array, vmax, vmin, n_bits):
    """
    # FIX-KT: Totally broken!
    Quantize array of data uniformly using the parameters given in input

    Parameters
    ----------
    array : decimal, [...]
        Array of data directly changed

    vmax : decimal, scalar
        Maximum quantization level

    vmin : decimal, scalar
        Minimum quantization level

    n_bits : int
        Number of bits used for quantization

    Returns
    -------
    """
    n_levels = 2**n_bits
    step = (vmax - vmin) / n_levels
    levels = np.flip(np.arange(start=vmin, stop=vmax, step=step), axis=0)  # descending
    top_level = np.inf
    for bottom_level in levels:
        array[np.logical_and(array >= bottom_level, array < top_level)] = bottom_level
        top_level = bottom_level

