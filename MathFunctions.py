from __future__ import division
from numpy import sqrt, mean, absolute

#matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack



def rms_flat(a):
    """
    Return the root mean square of all the elements of *a*, flattened out.
    """
    return sqrt(mean(absolute(a)**2))

def plot_fft(Y):
    # Number of samplepoints
    N = len(Y)
    # sample spacing
    T = 1.0
    yf = scipy.fftpack.fft(Y)
    xf = np.linspace(0.0, 1.0 / (2.0 * T), N / 2)

    fig, ax = plt.subplots()
    ax.plot(xf, 2.0 / N * np.abs(yf[:N // 2]))
    plt.show()


    return None