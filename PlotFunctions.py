#import pylab as plt
import matplotlib.pyplot as plt
import numpy
import scipy


def PlotEnergy(y1, y2):

    N = len(y1)
    t = numpy.arange(0, N, 1)
    print t.shape
    print y1.shape
    plt.plot(t, y1, 'r-', label='Modular Neural Network')
    plt.plot(t, y2, 'b-', label='PSB actual power')
    #plt.plot(t/24, y3, 'r-', label='Case II: Modular Network without schedules')
    plt.xlabel('day')
    plt.ylabel('Hourly electric load (KWh)')
    plt.legend(loc='upper left')
    plt.show()


    return None


def PlotEnergyDaily(y1, y2):

    N = len(y1)
    t = numpy.arange(0, N, 1)
    print t.shape
    print y1.shape
    plt.plot(t/60, y1, 'ro', label='Modular Neural Network')
    plt.plot(t/60, y2, 'ko', label='Acutal Energy Consumption')
    #plt.plot(t, y3, 'r-', label='Case II: Modular Network without schedules')
    #plt.axis([0, N, 0, 1])
    plt.xlabel('hours')
    plt.ylabel('Convenience Power (normalized)')
    plt.legend(loc='upper left')
    plt.show()


    return None

def plot_PSB_daily(y1, intv_min):
    z1 = 24*(60/intv_min)
    N_day = int(len(y1)/z1)

    y_daily = numpy.zeros((N_day, 1))

    for day in range(0, N_day):
        y_daily[day] = numpy.sum(y1[(day*z1):(day+1)*z1])/z1

    t = numpy.arange(0, N_day, 1)
    plt.plot(t, y_daily, 'ro')
    plt.xlabel('Days')
    plt.ylabel('Convience Power Consumption (KWH)')
    plt.show()

    return None

def Plot_single(y1):

    N = len(y1)
    t = numpy.arange(0, N, 1)
    plt.plot(t, y1, 'r-', label='Modular Neural Network')
    #plt.plot(t/24, y3, 'r-', label='Case II: Modular Network without schedules')
    plt.xlabel('Hours')
    plt.ylabel('Hourly electric load (KWh)')
   # plt.legend(loc='upper left')
    plt.show()


    return None

def Plot_double(t1, y1, t2, y2, Legend1, Legend2, color1, color2, figure_name):

    y_max = numpy.amax(y1) + 0.4

    plt.plot(t1, y1, color1, label=Legend1)
    plt.plot(t2, y2, color2, label=Legend2)
    #plt.plot(t/24, y3, 'r-', label='Case II: Modular Network without schedules')
    plt.xlabel('Hours')
    plt.ylabel('Hourly electric load (normalized)')
    plt.axis([0, 2200, 0, y_max])
    plt.legend(loc='upper right')
    plt.savefig(figure_name)
    plt.show()

    return None


def Plot_triple(t1, y1, t2, y2, t3, y3, Legend1, Legend2, Legend3, color1, color2, color3, figure_name):

    t_lim = t1[-1]
    t_max = t2[-1]



    plt.plot(t1, y1, color1, label=Legend1) #replacing plt with ax
    plt.plot(t2, y2, color2, label=Legend2)
    plt.plot(t3, y3, color3, label=Legend3)
    plt.plot((t_lim, t_lim), (0, 1.8), 'k--')
    #plt.plot(t/24, y3, 'r-', label='Case II: Modular Network without schedules')

    #Adding Annotations
    plt.annotate('', xy=(t_lim, 1.2), xycoords='data',xytext=(0, 1.2), textcoords='data', arrowprops={'arrowstyle': '<->'})
    plt.annotate('Training Phase', xy=(int(t_lim/2), 1.0), xycoords='data', xytext=(4000, 1.15), textcoords='data')
    plt.annotate('', xy=(t_max, 1.2), xycoords='data',xytext=(t_lim, 1.2), textcoords='data', arrowprops={'arrowstyle': '<->'})
    plt.annotate('Test Phase', xy=(int((t_max -t_lim)/2), 1.0), xycoords='data', xytext=(9100, 1.15), textcoords='data')

    plt.xlabel('Hours')
    plt.ylabel('Hourly electric load (normalized)')
    plt.axis([0, t_max,  0, 1.8])
    plt.legend(loc='upper right')
    plt.savefig(figure_name)
    plt.show()

    return None


def Plot_quadruple(t1, y1, t2, y2, t3, y3, t4, y4, Legend1, Legend2, Legend3, Legend4, color1, color2, color3, color4, figure_name):

    t_lim = t1[-1]
    t_max = t2[-1]

    plt.plot(t1, y1, color1, label=Legend1)
    plt.plot(t2, y2, color2, label=Legend2)
    plt.plot(t3, y3, color3, label=Legend3)
    plt.plot(t4, y4, color4, label=Legend4)
    plt.plot((t_lim, t_lim), (0, 1.8), 'k--')
    #plt.plot(t/24, y3, 'r-', label='Case II: Modular Network without schedules')

    #Adding Annotations
    plt.annotate('', xy=(t_lim, 1.2), xycoords='data',xytext=(0, 1.2), textcoords='data', arrowprops={'arrowstyle': '<->'})
    plt.annotate('Training Phase', xy=(int(t_lim/2), 1.0), xycoords='data', xytext=(4000, 1.15), textcoords='data')
    plt.annotate('', xy=(t_max, 1.2), xycoords='data',xytext=(t_lim, 1.2), textcoords='data', arrowprops={'arrowstyle': '<->'})
    plt.annotate('Test Phase', xy=(int((t_max -t_lim)/2), 1.0), xycoords='data', xytext=(9100, 1.15), textcoords='data')


    plt.xlabel('Hours')
    plt.ylabel('Hourly electric load (normalized)')
    plt.axis([0, t_max, 0, 1.8])
    plt.legend(loc='upper right')
    plt.savefig(figure_name)
    plt.show()

    return None

def Plot_Iterations(iterations, loss):
    plt.plot(iterations, loss, 'ko')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.show()


    return None

def Plot_fft(signal, t):
    FFT = abs(scipy.fft(signal))
    freqs = scipy.fftpack.fftfreq(signal.size, t[1] - t[0])

    plt.subplot(211)
    plt.plot(t, signal)
    plt.subplot(212)
    plt.plot(freqs, 20 * scipy.log10(FFT), 'x')
    plt.show()

    return None

def Plot_interpolate(Y_pre, Y_p, Y_nn, Y_a, Y_post):

    t_pre = numpy.arange(0, len(Y_pre))
    t_int = numpy.arange(len(Y_pre), len(Y_pre) + len(Y_p))
    t_post = numpy.arange(len(Y_pre) + len(Y_p), len(Y_pre) + len(Y_p) + len(Y_post))

    plt.plot(t_pre, Y_pre, 'k-', label='Available data')
    #plt.plot(t_int, Y_fp, 'r--', label='Interpolated data (forward in time)')
    #plt.plot(t_int, Y_bp, 'b--', label='Interpolated data (backward in time)')
    plt.plot(t_int, Y_p, 'r--', label='LSTM-Interpolated data')
    plt.plot(t_int, Y_nn, 'y--', label='NN-Interpolated data')
    plt.plot(t_int, Y_a, 'k--', label='Actual data')
    plt.plot(t_post, Y_post, 'k-')

    plt.xlabel('Hours')
    plt.ylabel('Hourly electric load (normalized)')
    plt.legend(loc='upper right')
    plt.savefig('plot_inpterp.eps')
    plt.show()

    return None


def Plot_interp_params():
    num = numpy.arange(1, 11, 1)
    p1 = numpy.array([8.93, 15.1, 9.073, 6.94, 9.83, 8.39, 10.55, 7.13, 6.72, 7.05])
    p2 = numpy.array([13.23, 11.3, 9.07, 8.07, 9.34, 9.14, 8.99, 8.13, 9.92, 12.1])

    print num.shape
    print p1.shape

    plt.plot(num, p1, 'ro', label='LSTM-Interpolated Data')
    plt.plot(num, p2, 'ko', label='MLP-interpolated Data')

    plt.xlabel('Number of days of missing values')
    plt.ylabel('Relative error, RMS (%)')
    plt.legend(loc='upper right')
    plt.savefig('plot_interp_param')
    plt.show()

    return None