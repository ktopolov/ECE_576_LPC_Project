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
    n_frames = (n_samps - (lpc_order + 1 + overlap_len)) // frame_len
    coeffs = np.zeros((n_frames, lpc_order))
    excitation_signal = np.zeros((data.shape))
    gain = np.zeros((n_frames,))
  
    # Indices: overap frames to fit LPCs but only use computation for non-overlapping portion
    n_samps_fit = frame_len + overlap_len + (lpc_order + 1)
    step = frame_len
    excess_start = int((lpc_order + 1) + overlap_len//2)

    for ifr in range(n_frames):
        # Get data indices for fitting
        fit_st = ifr * step
        fit_end = fit_st + n_samps_fit

        val_st = fit_st + excess_start
        val_end = val_st + frame_len

        fit_data = data[fit_st:fit_end]

        # Window, get autocorr matrix and solve for coeffs
        frame_data = fit_data# * sig.windows.hamming(n_samps_fit)

        # Get autocorrelation and solve for LPCs
        corr_half_idx = frame_data.shape[0] // 2
        corr = sig.correlate(in1=frame_data,
                             in2=frame_data,
                             mode='same')[corr_half_idx:]
        c = corr[:lpc_order]  # col and row of autocorr toeplitz matrix
        r = c
        b = corr[1:lpc_order+1]

        x = scilin.solve_toeplitz((c, r), b)
        coeffs[ifr, :] = x  # -a1 ... -aN in vocal tract all-pole filter

        # Inverse filter for excitation signal
        inv_filt_num = np.concatenate((np.array([1]), -x), axis=0)
        inv_filt_den = np.array([1])
        
        # use only valid data region to get excitation sig
        predict_data = fit_data[excess_start:excess_start + frame_len]
        exc_sig = sig.lfilter(b=inv_filt_num, a=inv_filt_den, x=predict_data)

        excitation_signal[val_st:val_end] = exc_sig
        gain[ifr] = np.linalg.norm(exc_sig, axis=0)  # use if reconstructing excitation signal
        
        # DEBUG - Check Prediction
        is_debug = False
        if is_debug:
            vocal_filt_num = np.array([1])
            vocal_filt_den = np.concatenate((np.array([1]), -x), axis=0)
            speech_sig = sig.lfilter(b=vocal_filt_num, a=vocal_filt_den, x=exc_sig)

            plt.figure(1, clear=True)
            plt.plot(speech_sig, 'r-o', label='Reconstruction', fillstyle='none')
            plt.plot(predict_data, 'b-.', label='Actual')
            plt.xlabel('Sample Index')
            plt.ylabel('Voltage')
            plt.title('Reconstruction vs. Actual - Frame ' + str(ifr))
            plt.grid(True)
            plt.legend()
            plt.show()
            plt.pause(1.0)

    return coeffs, excitation_signal, gain


def reconstruct_lpc(excitation_sig, lpc_coeffs, frame_len, overlap_len):
    """
    Given LPC coefficients and excitation signal, reconstruct speech
    
    Parameters
    ----------
    excitation_sig : decimal, [n_samps,]
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
    reconstruct_sig = np.zeros((excitation_sig.shape))
    step = frame_len
    lpc_order = lpc_coeffs.shape[1]
    excess_start = int((lpc_order + 1) + overlap_len//2)
    n_frames = lpc_coeffs.shape[0]

    for ifr in range(n_frames):
        # Indices of data
        val_st = ifr * step + excess_start
        val_end = val_st + frame_len
        
        # Filter excitation signal with vocal tract filter
        coeffs = lpc_coeffs[ifr, :]
        vocal_filt_den = np.concatenate((np.array([1]), -coeffs), axis=0)
        vocal_filt_num = np.array([1])
        predict_data = excitation_sig[val_st:val_end]
        reconstruct_sig[val_st:val_end] = sig.lfilter(b=vocal_filt_num,
                                                      a=vocal_filt_den,
                                                      x=predict_data)
    return reconstruct_sig
    