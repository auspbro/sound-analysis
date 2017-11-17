#!/usr/bin/env python
# -*- coding: utf-8 -*-
 
from __future__ import division
from soundfile import SoundFile
wav_loader = 'pysoundfile'


try:
    import easygui
    displayer = 'easygui'
except:
    displayer = 'stdout'


from numpy import array_equal, polyfit, sqrt, mean, real, conj, absolute, log10, arange
from scipy.stats import gmean
from scipy.signal import blackmanharris, butter, lfilter
import matplotlib.pyplot as plt

from waveform_analysis import A_weight, ITU_R_468_weight

def load(filename):
    """
    Load a wave file and return the signal, sample rate and number of channels.

    Can be any format supported by the underlying library (libsndfile or SciPy)
    """
    print "Using sound file backend '" + wav_loader + "'"

    wav_loader == 'pysoundfile'
    sf = SoundFile(filename)
    signal = sf.read()
    channels = sf.channels
    sample_rate = sf.samplerate
    sf.close()


    '''
    print 'Analyzing "' + filename + '"...'
    print '  Signal = ', signal.shape
    print '  Sample_rate = ', sample_rate
    print '  Channels = ', channels
    '''
    return signal, sample_rate, channels


def load_dict(filename):
    """
    Load a wave file and return the signal, sample rate and number of channels.

    Can be any format supported by the underlying library (libsndfile or SciPy)
    """
    soundfile = {}
    if wav_loader == 'pysoundfile':
        sf = SoundFile(filename)
        soundfile['signal'] = sf.read()
        soundfile['channels'] = sf.channels
        soundfile['fs'] = sf.samplerate
        soundfile['samples'] = len(sf)
        soundfile['format'] = sf.format_info + ' ' + sf.subtype_info
        sf.close()
    elif wav_loader == 'scikits.audiolab':
        sf = Sndfile(filename, 'r')
        soundfile['signal'] = sf.read_frames(sf.nframes)
        soundfile['channels'] = sf.channels
        soundfile['fs'] = sf.samplerate
        soundfile['samples'] = sf.nframes
        soundfile['format'] = sf.format
        sf.close()
    elif wav_loader == 'scipy.io.wavfile':
        soundfile['fs'], soundfile['signal'] = read(filename)
        try:
            soundfile['channels'] = soundfile['signal'].shape[1]
        except IndexError:
            soundfile['channels'] = 1
        soundfile['samples'] = soundfile['signal'].shape[0]
        soundfile['format'] = str(soundfile['signal'].dtype)

    return soundfile


def display(header, results):
    """
    Display header string and list of result lines
    """
    if displayer == 'easygui':
        title = 'Waveform properties'
        easygui.codebox(header, title, '\n'.join(results))
    else:
        print('No EasyGUI; printing output to console\n')
        print(header)
        print('-----------------')
        print('\n'.join(results))


def histogram(signal):
    """
    Plot a histogram of the sample values
    """
    try:
        from matplotlib.pyplot import hist, show
    except ImportError:
        print('Matplotlib not installed - skipping histogram')
    else:
        print('Plotting histogram')
        hist(signal)  # TODO: parameters, abs(signal)?
        show()


def properties(signal, sample_rate):
    """
    Return a list of some wave properties for a given 1-D signal
    """
    # Measurements that include DC component
    DC_offset = mean(signal)
    # Maximum/minimum sample value
    # Estimate of true bit rate

    # Remove DC component
    signal -= mean(signal)

    # Measurements that don't include DC
    signal_level = rms_flat(signal)
    peak_level = max(absolute(signal))
    crest_factor = peak_level/signal_level

    # Apply the A-weighting filter to the signal
    Aweighted = A_weight(signal, sample_rate)
    Aweighted_level = rms_flat(Aweighted)

    # Apply the ITU-R 468 weighting filter to the signal
    ITUweighted = ITU_R_468_weight(signal, sample_rate)
    ITUweighted_level = rms_flat(ITUweighted)

    # TODO: rjust instead of tabs

    return [
        'DC offset:\t%f (%.3f%%)' % (DC_offset, DC_offset * 100),
        'Crest factor:\t%.3f (%.3f dB)' % (crest_factor, dB(crest_factor)),
        'Peak level:\t%.3f (%.3f dBFS)' %
        (peak_level, dB(peak_level)),  # Doesn't account for intersample peaks!
        'RMS level:\t%.3f (%.3f dBFS)' % (signal_level, dB(signal_level)),
        'RMS A-weighted:\t%.3f (%.3f dBFS(A), %.3f dB)' %
        (Aweighted_level, dB(Aweighted_level),
         dB(Aweighted_level/signal_level)),
        'RMS 468-weighted:\t%.3f (%.3f dBFS(468), %.3f dB)' %
        (ITUweighted_level, dB(ITUweighted_level),
         dB(ITUweighted_level/signal_level)),
        '-----------------',
    ]


def analyze(filename):
    if wav_loader == 'pysoundfile':
        sf = SoundFile(filename)
        signal = sf.read()
        channels = sf.channels
        sample_rate = sf.samplerate
        samples = len(sf)
        file_format = sf.format_info + ' ' + sf.subtype_info
        sf.close()
    elif wav_loader == 'scikits.audiolab':
        sf = Sndfile(filename, 'r')
        signal = sf.read_frames(sf.nframes)
        channels = sf.channels
        sample_rate = sf.samplerate
        samples = sf.nframes
        file_format = sf.format
        sf.close()
    elif wav_loader == 'scipy.io.wavfile':
        sample_rate, signal = read(filename)
        try:
            channels = signal.shape[1]
        except IndexError:
            channels = 1
        samples = signal.shape[0]
        file_format = str(signal.dtype)

        # Scale common formats
        # Other bit depths (24, 20) are not handled by SciPy correctly.
        if file_format == 'int16':
            signal = signal.astype(float) / (2**15)
        elif file_format == 'uint8':
            signal = (signal.astype(float) - 128) / (2**7)
        elif file_format == 'int32':
            signal = signal.astype(float) / (2**31)
        elif file_format == 'float32':
            pass
        else:
            raise Exception("Don't know how to handle file "
                            "format {}".format(file_format))

    else:
        raise Exception("wav_loader has failed")

    header = 'dBFS values are relative to a full-scale square wave'

    if samples/sample_rate >= 1:
        length = str(samples/sample_rate) + ' seconds'
    else:
        length = str(samples/sample_rate*1000) + ' milliseconds'

    results = [
        "Using sound file backend '" + wav_loader + "'",
        'Properties for "' + filename + '"',
        str(file_format),
        'Channels:\t%d' % channels,
        'Sampling rate:\t%d Hz' % sample_rate,
        'Samples:\t%d' % samples,
        'Length: \t' + length,
        '-----------------',
        ]

    if channels == 1:
        # Monaural
        results += properties(signal, sample_rate)
    elif channels == 2:
        # Stereo
        if array_equal(signal[:, 0], signal[:, 1]):
            results += ['Left and Right channels are identical:']
            results += properties(signal[:, 0], sample_rate)
        else:
            results += ['Left channel:']
            results += properties(signal[:, 0], sample_rate)
            results += ['Right channel:']
            results += properties(signal[:, 1], sample_rate)
    else:
        # Multi-channel
        for ch_no, channel in enumerate(signal.transpose()):
            results += ['Channel %d:' % (ch_no + 1)]
            results += properties(channel, sample_rate)

    display(header, results)

    plot_histogram = False
    if plot_histogram:
        histogram(signal)


def analyze_function(filename, function):
    """
    Given a filename, run the given analyzer function on each channel of the
    file
    """
    
    signal, sample_rate, channels = load(filename)
    print('Analyzing "' + filename + '"...')

    if channels == 1:
        # Monaural
        function(signal, sample_rate)
    elif channels == 2:
        # Stereo
        if array_equal(signal[:, 0], signal[:, 1]):
            print('-- Left and Right channels are identical --')
            function(signal[:, 0], sample_rate)
        else:
            print('-- Left channel --')
            function(signal[:, 0], sample_rate, 1, 1, 5, "left", filename)
            print('-- Right channel --')
            function(signal[:, 1], sample_rate, 1, 1, 5, "right", filename)
    else:
        # Multi-channel
        for ch_no, channel in enumerate(signal.transpose()):
            print('-- Channel %d --' % (ch_no + 1))
            function(channel, sample_rate)
 
 
def rms_flat(a):  # Copied from matplotlib.mlab
    """
    Return the root mean square of all the elements of *a*, flattened out.
    """
    return sqrt(mean(absolute(a)**2))


def rms_fft(spectrum):
    """
    Use Parseval's theorem to find the RMS value of a signal from its fft,
    without wasting time doing an inverse FFT.
    For a signal x, these should produce the same result, to within numerical
    accuracy:
    rms_flat(x) ~= rms_fft(fft(x))
    """
    return rms_flat(spectrum)/sqrt(len(spectrum))


def rms_rfft(spectrum, n=None):
    """
    Use Parseval's theorem to find the RMS value of an even-length signal
    from its rfft, without wasting time doing an inverse real FFT.
    spectrum is produced as spectrum = numpy.fft.rfft(signal)
    For a signal x with an even number of samples, these should produce the
    same result, to within numerical accuracy:
    rms_flat(x) ~= rms_rfft(rfft(x))
    If len(x) is odd, n must be included, or the result will only be
    approximate, due to the ambiguity of rfft for odd lengths.
    """
    if n is None:
        n = (len(spectrum) - 1) * 2
    sq = real(spectrum * conj(spectrum))
    if n % 2:  # odd-length
        mean = (sq[0] + 2*sum(sq[1:])           )/n
    else:  # even-length
        mean = (sq[0] + 2*sum(sq[1:-1]) + sq[-1])/n
    root = sqrt(mean)
    return root/sqrt(n)


def find(condition):
    "Return the indices where ravel(condition) is true"
    res, = np.nonzero(np.ravel(condition))
    return res


def dB(q):
    """
    Return the level of a field quantity in decibels.
    """
    return 20 * log10(q)


def findPeakidx(data, idx, intvel):
    temp = data[:]
    if (idx - intvel) > 0:
        temp[:(idx - intvel)] = np.zeros(len(temp[:(idx - intvel)]))

    if (idx + intvel) < len(data):
        temp[(idx + intvel):] = np.zeros(len(temp[(idx + intvel):]))

    return np.argmax(temp)


def find_nearest_vector(array, value):
    from scipy import spatial
    '''
    # https://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array
    A = np.random.random((10,2))*100
    print A
    pt = [6, 30]  # <-- the point to find
    print find_nearest_vector(A,pt)
    print A[spatial.KDTree(A).query(pt)[1]]
    '''
    idx = np.array([np.linalg.norm(x+y) for (x,y) in array-value]).argmin()
    return array[idx]


def find_closest(A, target):
    '''
    # https://stackoverflow.com/questions/8914491/finding-the-nearest-value-and-return-the-index-of-array-in-python
    A = np.arange(0, 20.)
    print A
    target = np.array([-2, 100., 2., 2.4, 2.5, 2.6])
    print find_closest(A, target)
    '''
    #A must be sorted
    idx = A.searchsorted(target)
    idx = np.clip(idx, 1, len(A)-1)
    left = A[idx-1]
    right = A[idx]
    idx -= target - left < right - target
    return idx


def closest(list, Number):
    aux = []
    for valor in list:
        aux.append(abs(Number-valor))
    return aux.index(min(aux))


def takeClosest(myList, myNumber):
    from bisect import bisect_left
    '''
    # https://stackoverflow.com/questions/12141150/from-list-of-integers-get-number-closest-to-a-given-value
    a = range(-1000, 1000, 10)
    myList = [4, 1, 88, 44, 3]
    myNumber = 5
    print takeClosest(myList, myNumber)
    '''
    """
    Assumes myList is sorted. Returns closest value to myNumber.

    If two numbers are equally close, return the smallest number.
    """
    pos = bisect_left(myList, myNumber)
    if pos == 0:
        return myList[0]
    if pos == len(myList):
        return myList[-1]
    before = myList[pos - 1]
    after = myList[pos]
    if after - myNumber < myNumber - before:
#        print after
#        return before
        return pos
    else:
#        print before
#        return before
        return pos-1


# Graphing helper function
def setup_graph(title='', x_label='', y_label='', fig_size=None):
    fig = plt.figure()
    if fig_size != None:
        fig.set_size_inches(fig_size[0], fig_size[1])
    ax = fig.add_subplot(111)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    '''
    # https://github.com/timsainb/python_spectrograms_and_inversion
    lowcut = 950 # Hz # Low cut for our butter bandpass filter
    highcut = 1050 # Hz # High cut for our butter bandpass filter
    signal = butter_bandpass_filter(signal, lowcut, highcut, sample_rate, order=1)

    if np.shape(signal)[0]/float(sample_rate) > 10:
        signal = signal[0:sample_rate*10] 
    print 'Length in time (s): ', np.shape(signal)[0]/float(sample_rate)
    '''
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

'''
class EnterExitLog():
    def __init__(self, funcName):
        self.funcName = funcName

    def __enter__(self):
        gLog.debug('Started: %s' % self.funcName)
        self.init_time = datetime.datetime.now()
        return self

    def __exit__(self, type, value, tb):
        gLog.debug('Finished: %s in: %s seconds' % (self.funcName, datetime.datetime.now() - self.init_time))


def func_wrapper(*args, **kwargs):
    with EnterExitLog(func.__name__):
        return func(*args, **kwargs)


def func_timer_decorator(func):
    def func_wrapper(*args, **kwargs):
        with EnterExitLog(func.__name__):
            return func(*args, **kwargs)
    return func_wrapper
'''

def spectral_flatness(spectrum):
    """
    The spectral flatness is calculated by dividing the geometric mean of
    the power spectrum by the arithmetic mean of the power spectrum

    I'm not sure if the spectrum should be squared first...
    """
    return gmean(spectrum)/mean(spectrum)


def parabolic(f, x):
    """
    Quadratic interpolation for estimating the true position of an
    inter-sample maximum when nearby samples are known.

    f is a vector and x is an index for that vector.

    Returns (vx, vy), the coordinates of the vertex of a parabola that goes
    through point x and its two neighbors.

    Example:
    Defining a vector f with a local maximum at index 3 (= 6), find local
    maximum if points 2, 3, and 4 actually defined a parabola.

    In [3]: f = [2, 3, 1, 6, 4, 2, 3, 1]

    In [4]: parabolic(f, argmax(f))
    Out[4]: (3.2142857142857144, 6.1607142857142856)
    """
    xv = 1/2. * (f[x-1] - f[x+1]) / (f[x-1] - 2 * f[x] + f[x+1]) + x
    yv = f[x] - 1/4. * (f[x-1] - f[x+1]) * (xv - x)
    return (xv, yv)


def parabolic_polyfit(f, x, n):
    """
    Use the built-in polyfit() function to find the peak of a parabola

    f is a vector and x is an index for that vector.

    n is the number of samples of the curve used to fit the parabola.
    """
    a, b, c = polyfit(arange(x-n//2, x+n//2+1), f[x-n//2:x+n//2+1], 2)
    xv = -0.5 * b/a
    yv = a * xv**2 + b * xv + c
    return (xv, yv)
