import numpy as np
import matplotlib.pyplot as plt


def plot_simEEG(EEG, chan=0, fignum=1):
    """
    plot_simEEG - plot function for MXC's course on neural time series analysis
    (Python version of plot_simEEG.m)

    INPUTS:  EEG : EEG structure (SimpleNamespace) with fields
                   data (channels x time x trials), times, srate, pnts, trials
            chan : channel to plot (default = 0, the first channel in Python)
          fignum : figure to plot into (default = 1)
    """

    fig = plt.figure(fignum, figsize=(10, 7))
    fig.clf()

    # ERP

    plt.subplot(211)
    plt.plot(EEG.times, EEG.data[chan, :, :], color=[.75, .75, .75], linewidth=.5)
    plt.plot(EEG.times, np.mean(EEG.data[chan, :, :], axis=1), 'k', linewidth=3)
    plt.xlabel('Time (s)')
    plt.ylabel('Activity')
    plt.title('ERP from channel ' + str(chan))

    # static power spectrum

    hz = np.linspace(0, EEG.srate, EEG.pnts)
    pw = np.mean((2 * np.abs(np.fft.fft(EEG.data[chan, :, :], axis=0) / EEG.pnts))**2, axis=1)

    plt.subplot(223)
    plt.plot(hz, pw, linewidth=2)
    plt.xlim([0, 40])
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power')
    plt.title('Static power spectrum')

    # time-frequency analysis

    # frequencies in Hz (hard-coded to 2 to 30 in 40 steps)
    frex = np.linspace(2, 30, 40)

    # number of wavelet cycles (hard-coded to 3 to 10)
    waves = 2 * (np.linspace(3, 10, len(frex)) / (2 * np.pi * frex))**2

    # setup wavelet and convolution parameters
    wavet = np.arange(-2 * EEG.srate, 2 * EEG.srate + 1) / EEG.srate
    halfw = len(wavet) // 2 + 1
    nConv = EEG.pnts * EEG.trials + len(wavet) - 1

    # initialize time-frequency matrix
    tf = np.zeros((len(frex), EEG.pnts))

    # spectrum of data (all trials one after the other, like MATLAB's reshape)
    dataX = np.fft.fft(EEG.data[chan, :, :].flatten(order='F'), nConv)

    # loop over frequencies
    for fi in range(len(frex)):

        # create wavelet
        waveX = np.fft.fft(np.exp(2j * np.pi * frex[fi] * wavet) * np.exp(-wavet**2 / waves[fi]), nConv)
        waveX = waveX / waveX[np.argmax(np.abs(waveX))]  # normalize

        # convolve
        as_ = np.fft.ifft(waveX * dataX)
        # trim and reshape
        as_ = as_[halfw - 1:len(as_) - halfw + 1]
        as_ = as_.reshape(EEG.trials, EEG.pnts).T

        # power
        tf[fi, :] = np.mean(np.abs(as_), axis=1)

    # show a map of the time-frequency power
    plt.subplot(224)
    plt.contourf(EEG.times, frex, tf, 40)
    plt.xlabel('Time')
    plt.ylabel('Frequency (Hz)')
    plt.title('Time-frequency plot')

    plt.tight_layout()
    plt.show()
