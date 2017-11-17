#!/usr/bin/env python
# -*- coding: utf-8 -*-
 
from __future__ import division
import sys
from optparse import OptionParser
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import blackmanharris
from scipy.fftpack import fft
from common import load, analyze, analyze_function, rms_flat, dB, rms_fft, rms_rfft, setup_graph, takeClosest
#from thd import THDN, THD

import traceback

import os
local_dir = os.path.dirname(os.path.realpath(__file__))
'''
# print current info
filename, codeline, funcName, text = traceback.extract_stack()[-1]
print (filename, codeline, funcName, text)
print traceback.extract_stack(None, 1)[0][2] # print current function name

# print caller info
filename, codeline, funcName, text = traceback.extract_stack()[-2]
print (filename, codeline, funcName, text)
print traceback.extract_stack(None, 2)[0][2] # print current function name
'''

'''
def find_range(f, x):
    """
    Find range between nearest local minima from peak at index x
    """
    for i in np.arange(x+1, len(f)):
        if f[i+1] >= f[i]:
            uppermin = i
            break
    for i in np.arange(x-1, 0, -1):
        if f[i] <= f[i-1]:
            lowermin = i + 1
            break
    return (lowermin, uppermin)


def THDN(signal, sample_rate):
    """
    Measure the THD+N for a signal and print the results

    Prints the estimated fundamental frequency and the measured THD+N.  This is
    calculated from the ratio of the entire signal before and after
    notch-filtering.

    Currently this tries to find the "skirt" around the fundamental and notch
    out the entire thing.  A fixed-width filter would probably be just as good,
    if not better.
    """
    # Get rid of DC and window the signal
    signal -= np.mean(signal) # TODO: Do this in the frequency domain, and take any skirts with it?
    windowed = signal * blackmanharris(len(signal))  # TODO Kaiser?

    # Measure the total signal before filtering but after windowing
    total_rms = rms_flat(windowed)
#    print '\ntotal_rms: %f' % (total_rms)
#    print "\n RMS = %f" % ( np.sqrt(np.mean(windowed**2)) )

    # Find the peak of the frequency spectrum (fundamental frequency), and
    # filter the signal by throwing away values between the nearest local
    # minima
    f = np.fft.rfft(windowed)
    i = np.argmax(abs(f))
#    print 'Frequency: %f Hz' % (sample_rate * (i / len(windowed)))  # Not exact
    lowermin, uppermin = find_range(abs(f), i)
    f[lowermin: uppermin] = 0

    # Transform noise back into the signal domain and measure it
    # TODO: Could probably calculate the RMS directly in the frequency domain instead
    noise = np.fft.irfft(f)
    THDN = rms_flat(noise) / total_rms
    print "THD+N:     %.4f%% or %.1f dB" % (THDN * 100, 20 * np.log10(THDN))


def THD(signal, sample_rate):
    """Measure the THD for a signal
 
    This function is not yet trustworthy.
 
    Returns the estimated fundamental frequency and the measured THD,
    calculated by finding peaks in the spectrum.
 
    There are two definitions for THD, a power ratio or an amplitude ratio
    When finished, this will list both
 
    """
    # Get rid of DC and window the signal
    signal -= np.mean(signal) # TODO: Do this in the frequency domain, and take any skirts with it?
    windowed = signal * blackmanharris(len(signal))  # TODO Kaiser?
 
    # Find the peak of the frequency spectrum (fundamental frequency), and
    # filter the signal by throwing away values between the nearest local
    # minima
    f = np.fft.rfft(windowed)
    i = np.argmax(abs(f))
#    print 'Frequency: %f Hz' % (sample_rate * (i / len(windowed)))  # Not exact
#    print 'fundamental amplitude: %.3f' % abs(f[i])

    # Find the values for the first 15 harmonics.  Includes harmonic peaks
    # only, by definition
    # TODO: Should peak-find near each one, not just assume that fundamental
    # was perfectly estimated.
    # Instead of limited to 15, figure out how many fit based on f0 and
    # sampling rate and report this "4 harmonics" and list the strength of each
#    for x in range(2, 15):
#        print('%.3f' % abs(f[i * x]))

    THD = sum([abs(f[i*x]) for x in range(2, 15)]) / abs(f[i])
    print '\nTHD: %f%%' % (THD * 100)
'''


def SaveGraph(signal, sample_rate):
    # https://github.com/calebmadrigal/FourierTalkOSCON/blob/master/06_FFTInPython.ipynb
#    signal = signal / (2.**15) #convert sound array to float pt. values
    print 'sample_rate = ' + str(sample_rate) # samples per sec

    sample = len(signal) # euq signal.shape[0]
    print 'samples = ' + str(sample)

    total_sampling_time = sample/sample_rate
    print 'total_sampling_time = ' + str(total_sampling_time) + ' seconds'

    num_samples = sample_rate * total_sampling_time # euq signal.shape[0] and euq len(signal)
#    print 'num_samples (samples) = ' + str(num_samples)

    t = np.linspace(0, total_sampling_time, num_samples)
#    '''
    setup_graph(title='Signal output', x_label='time (in seconds)', y_label='amplitude', fig_size=(12,6))
    plt.plot(t, signal)
    plt.savefig('signal.png')
#    '''

    # Take the fft
    fft_output = np.fft.fft(signal)
#    '''
    setup_graph(title='FFT output', x_label='FFT bins', y_label='amplitude', fig_size=(12,6))
    plt.plot(fft_output)
    plt.savefig('fft.png')
#    '''

    # For real-valued input, the fft output is always symmetric.
    rfft_output = np.fft.rfft(signal)
#    '''
    setup_graph(title='rFFT output', x_label='frequency (in Hz)', y_label='amplitude', fig_size=(12,6))
    plt.plot(rfft_output)
    plt.savefig('rfft.png')
#   '''

    # Getting frequencies on x-axis labels
    # the x-axis to represent frequency
    # Frequencies range from 0 to the Nyquist Frequency (sample rate / 2)
    rfreqs = [(i*1.0/num_samples)*sample_rate for i in range(int(num_samples//2+1))]
#    '''
    setup_graph(title='Corrected rFFT Output', x_label='frequency (in Hz)', y_label='amplitude', fig_size=(12,6))
    plt.plot(rfreqs, rfft_output)
    plt.savefig('rfft_corrected.png')
#    '''

    # Getting negative values on y-axis labels
    # the y-axis to represent magnitude
#    '''
    rfft_mag = [np.sqrt(i.real**2 + i.imag**2)/len(rfft_output) for i in rfft_output]
    setup_graph(title='Corrected rFFT Output', x_label='frequency (in Hz)', y_label='magnitude', fig_size=(12,6))
    plt.plot(rfreqs, rfft_mag)
    plt.savefig('rfft_magnitude.png')
#    '''

#    '''
    # take the output of the FFT and perform an Inverse FFT to get back to our original wave (using the Inverse Real FFT - irfft).
    irfft_output = np.fft.irfft(rfft_output)
    setup_graph(title='Inverse rFFT Output', x_label='time (in seconds)', y_label='amplitude', fig_size=(12,6))
    plt.plot(t, irfft_output)
    plt.savefig('rfft_inverse.png')
#    '''
#    plt.show()


def DrawGraph(signal, sample_rate):
    # https://github.com/calebmadrigal/FourierTalkOSCON/blob/master/06_FFTInPython.ipynb
#    signal = signal / (2.**15) #convert sound array to float pt. values
    print 'sample_rate = ' + str(sample_rate) # samples per sec

    sample = len(signal) # euq signal.shape[0]
    print 'samples = ' + str(sample)

    total_sampling_time = sample/sample_rate
    print 'total_sampling_time = ' + str(total_sampling_time) + ' seconds'

    num_samples = sample_rate * total_sampling_time # euq signal.shape[0] and len(signal)
#    print 'num_samples (samples) = ' + str(num_samples)

#    '''    
    time_array = np.arange(0, num_samples*1.0/sample_rate, 1.0/sample_rate) #start, stop, step
    setup_graph(title='Time domain Output', x_label='Time (in Sec)', y_label='amplitude', fig_size=(12,4))
    plt.plot(time_array, signal)
#    '''

    rfft_output = np.fft.rfft(signal)

    # Plotting the Frequency Content
    amplitude = [np.sqrt(i.real**2 + i.imag**2)/len(rfft_output) for i in rfft_output]
    frequencies = [(i*1.0/num_samples)*sample_rate for i in range(int(num_samples//2+1))]

    setup_graph(title='Frequency domain Output', x_label='frequency (in Hz)', y_label='amplitude', fig_size=(12,4))
    plt.plot(frequencies, amplitude, color='r')

    power = dB(amplitude)
    setup_graph(title='Frequency domain Output', x_label='frequency (in Hz)', y_label='power (dB)', fig_size=(12,4))
    plt.plot(frequencies, power, color='k')

    setup_graph(title='Spectrogram', x_label='time (in seconds)', y_label='frequency', fig_size=(14,7))
    plt.specgram(signal, Fs=sample_rate)

    plt.show()

 
def wave_analyzer(files):
    if files:
        for filename in files:
            try:
                load(filename)
            except IOError:
                print 'Couldn\'t analyze "' + filename + '"\n'
            print ''
    else:
        # TODO: realtime analyzer goes here
        sys.exit("You must provide at least one file to analyze:\n"
                 "python wave_analyzer.py filename.wav")


def CalculateBandRMS(signal, sample_rate, band=1, percent=5):
#    print sys._getframe().f_code.co_name # print function name

    # https://gist.github.com/endolith/1257010
    # https://scipy.github.io/devdocs/tutorial/fftpack.html
    num_samples = len(signal)

    rfft_array = np.fft.rfft(signal)
    frequencies = [(i*1.0/num_samples)*sample_rate for i in range(num_samples//2+1)] #0,(1/n)*rate,(2/n)*rate,....((n/2)/n)*rate

    '''
    rms = np.mean(map(lambda x: x**2, signal)) ** .5
    print rms
    rms_val = np.sqrt(np.mean(signal**2))
    db_val = 20*np.log10(rms_flat(signal))
    print (rms_val, db_val)

    rms_val_raw = rms_flat(signal)
    print('Original RAW RMS: {:+.2f}, {:+.2f} dB'.format( rms_val_raw, dB(rms_val_raw) ))

    fft_signal = np.fft.fft(signal)
    rms_val_fft = rms_fft( fft_signal )
    print('Original FFT RMS: {:+.2f}, {:+.2f} dB'.format( rms_val_fft, dB(rms_val_fft) ))

    rms_val_fft2ifft = rms_flat( np.fft.ifft(fft_signal) )
    print('Original FFT to iFFT RMS: {:+.2f}, {:+.2f} dB'.format( rms_val_fft2ifft, dB(rms_val_fft2ifft) ))

    rfft_signal = np.fft.rfft(signal)
    # Only approximate for odd n:
    rms_val_rfft2irfft = rms_flat( np.fft.irfft(rfft_signal) )
    print('Original rFFT to irFFT RMS: {:+.2f}, {:+.2f} dB'.format( rms_val_rfft2irfft, dB(rms_val_rfft2irfft) ))

    rms_val_rfft = rms_rfft(rfft_signal)
    print('Original rFFT RMS: {:+.2f}, {:+.2f} dB'.format( rms_val_rfft, dB(rms_val_rfft) ))

    # Accurate for odd n:
    rms_val_rfft2irfft_with_size = rms_flat( np.fft.irfft(rfft_signal, num_samples) )
    print('Original rFFT to irFFT RMS: {:+.2f}, {:+.2f} dB'.format( rms_val_rfft2irfft_with_size, dB(rms_val_rfft2irfft_with_size) ))
    
    rms_val_rfft_with_size = rms_rfft( rfft_signal, num_samples )
    print('Original rFFT RMS: {:+.2f}, {:+.2f} dB'.format( rms_val_rfft_with_size, dB(rms_val_rfft_with_size) ))
    '''

#    irfft_array = np.fft.irfft(rfft_array) # irfft_array must euq signal
    bandwidth = band*1000
    lowcut = 0 # Hz # Remove lower frequencies.
    highcut = 0 # Hz # Remove higher frequencies.

    search_range = ( bandwidth * percent ) / 100

    lowcut = takeClosest(frequencies, (bandwidth-search_range) )
    highcut  = takeClosest(frequencies, (bandwidth+search_range) )
#    print(lowcut, highcut)

    cut_rfft_array = rfft_array.copy()
    cut_rfft_array[:lowcut] = 0 # Hz # Low cut
    cut_rfft_array[highcut:] = 0 # Hz # High cut

    cut_signal = np.fft.irfft(cut_rfft_array)
    rms_val_cut = rms_flat(cut_signal)
#    print('band=%d rms=%.4f power=%.4f' %( band, rms_val_cut, dB(rms_val_cut) ))
    print('SET BAND_%d=%.4f' %( band, dB(rms_val_cut) ))

    '''
    setup_graph(title='Spectrogram (Before)', x_label='time (in seconds)', y_label='frequency', fig_size=(14,7))
    plt.specgram(signal, Fs=sample_rate)

    setup_graph(title='Spectrogram (After)', x_label='time (in seconds)', y_label='frequency', fig_size=(14,7))
    plt.specgram(cut_signal, Fs=sample_rate)


    time_array = np.arange(0, num_samples*1.0/sample_rate, 1.0/sample_rate) #start, stop, step

    rms = rms_flat(signal)
    plt.figure(figsize=(10,5))
    plt.plot(time_array, signal, color="red")
    plt.plot([0, num_samples*1.0/sample_rate], [rms, rms], color="green", linewidth=2, alpha=.8)
    plt.plot([0, num_samples*1.0/sample_rate], [-rms, -rms], color="green", linewidth=2, alpha=.8)
#    plt.plot(signal, color="blue")
#    plt.plot([0, len(signal)], [rms, rms], color="green", linewidth=2, alpha=.8)
#    plt.plot([0, len(signal)], [-rms, -rms], color="green", linewidth=2, alpha=.8)
    plt.title( 'Time domain (Before)\nRMS=' + str(rms) )
    plt.xlabel('time (in seconds)')
    plt.ylabel('amplitude')

    rms = rms_flat(cut_signal)
    plt.figure(figsize=(10,5))
    plt.plot(time_array, cut_signal, color="red")
    plt.plot([0, num_samples*1.0/sample_rate], [rms, rms], color="green", linewidth=2, alpha=.8)
    plt.plot([0, num_samples*1.0/sample_rate], [-rms, -rms], color="green", linewidth=2, alpha=.8)
    plt.title( 'Time domain (After)\nRMS=' + str(rms) )
    plt.xlabel('time (in seconds)')
    plt.ylabel('amplitude')

    plt.show()
    '''


def CalculateBandMAX(signal, sample_rate, band_start=1, band_stop=1, percent=5, left_right="none", filename="none"):
#    print traceback.extract_stack(None, 1)[0][2] # print current function name

    '''
    import traceback
    stack = traceback.extract_stack()
    filename, codeline, funcName, text = stack[-2] # print caller function info
    print (filename, codeline, funcName, text)
    filename, codeline, funcName, text = stack[-1] # print current function info
    print (filename, codeline, funcName, text)
    filename, codeline, funcName, text = stack[0] # print root function info
    print (filename, codeline, funcName, text)
    '''
#    print traceback.extract_stack(None, 2)[0][2] # print caller function name
#    print traceback.extract_stack(None, 1)[0][2] # print current function name

    num_samples = len(signal)

    rfft_array = np.fft.rfft(signal)
    amplitude = [np.sqrt(i.real**2 + i.imag**2)/len(rfft_array) for i in rfft_array]
    frequencies = [(i*1.0/num_samples)*sample_rate for i in range(num_samples//2+1)] #0,(1/n)*rate,(2/n)*rate,....((n/2)/n)*rate
    power = dB(amplitude)


    fft_signal = np.fft.fft(signal)
    freqs = np.fft.fftfreq(len(fft_signal))
    #print(freqs.min(), freqs.max())
    # (-0.5, 0.499975)

    # Find the peak in the coefficients
    idx = np.argmax(np.abs(fft_signal))
    freq = freqs[idx]
    freq_in_hertz = abs(freq * sample_rate)
    print(freq_in_hertz)


#    DrawGraph(signal, sample_rate)

    '''
    np.savetxt("amplitude.txt", amplitude, fmt='%6.3f')
    np.savetxt("frequencies.txt", frequencies, fmt='%6.3f')
    np.savetxt("power.txt", power, fmt='%6.3f')
    '''

#    '''
#    min_val,idx = min( (l[i],i) for i in xrange(len(l)) )
#    max_val,idx = max( (l[i],i) for i in xrange(len(l)) )

    peaks = []
#    print "{:*^80s}".format('Peak List')
    for x in range(band_start, band_stop+1): #check peak value from band_start to band_end kHz
        kx = 1000 * x
        search_range = (kx*percent)/100
        kx_start = kx - search_range
        kx_stop  = kx + search_range

        kx_idx_start = takeClosest(frequencies, kx_start)
        kx_idx_stop  = takeClosest(frequencies, kx_stop) + 1

        #find a good max amplitude.
#        maxAmplitude = np.max( np.absolute( amplitude[kx_idx_start:kx_idx_stop] ) )
#        peak_idx = closest(amplitude, maxAmplitude)
#        peak_idx = int( np.where( amplitude==np.max(amplitude[kx_idx_start:kx_idx_stop]) )[0] )
        peak_idx = int( np.where( power==np.max(power[kx_idx_start:kx_idx_stop]) )[0] )

        '''
        print 'search_range = ' + str(search_range)
        print 'kx = ' + str(kx) + ' [ ' + str(kx_start) + ' ~ ' + str(kx_stop) + ' ]'
        print 'kx_idx_start = ' + str(kx_idx_start)
        print 'kx_idx_stop  = ' + str(kx_idx_stop)

        print '--> peak_idx = ' + str(peak_idx) 

        offset = (abs(frequencies[kx_idx_start] - frequencies[peak_idx])/frequencies[kx_idx_start]) * 100
        print 'Frequency: %.4f Hz amplitude: %.4f, offset of %.4f Hz: %.4%%' %(frequencies[peak_idx], amplitude[peak_idx], frequencies[kx_idx_start], offset)
        '''
#        print 'band=%d frequency=%.4f amplitude=%.4f power=%.4f' %(x, frequencies[peak_idx], amplitude[peak_idx], power[peak_idx] )
        # 1 kilohertz (kHz) to 1000 hertz (Hz) = 1 kHz × 1000 = 1000 Hz
        
        print 'SET %s=%.4f' %(left_right, power[peak_idx] )
        result(power[peak_idx], left_right, filename)
#    print "{:*^80s}".format('')

def result(audio_dB, left_right, filename):
    set_result = "Fail"

    result_file = filename.split(".")[0] + "_out.bat"
    if audio_dB > -10:
        set_result = "PASS"

    result_path = os.path.join(local_dir, result_file)
    result_string = "Set " + left_right + "dB=" + str(audio_dB) + "\n"
    result_string2 = "Set " + left_right + "=" + set_result + "\n"
    #print(result_string, result_path)
    
    f = open(result_path, 'a+')
    f.write(result_string)
    f.write(result_string2)
    f.closed

def usage():
    print("Usage:")
    print(sys.argv[0] + ' -w wavefile -f function -b begin -e end -p percent')
    sys.exit(2)


def parse_args():
    # http://python.usyiyi.cn/translate/python_278/library/argparse.html
    parser = OptionParser(usage='%prog -w wavefile -f function -b begin -e end -p percent')
    parser.add_option('-w', '--wavefile',
                        action="store", dest="wavefile",
                        help="Wave Filename", default="None")
    parser.add_option('-f', '--function',
                        action="store", dest="function",
                        help="Run Function", default="None")
    parser.add_option('-b', '--begin',
                        action="store", dest="begin",
                        help="Start Band", default=1, type=int)
    parser.add_option('-e', '--end',
                        action="store", dest="end",
                        help="Stop Band", default=1, type=int)
    parser.add_option('-p', '--percent',
                        action="store", dest="percent",
                        help="Search Percent", default=5, type=int)

    options, args = parser.parse_args()
#    print "Length : %d" % len(options.__dict__)
#    print options

    if not (options.wavefile and options.function):
        usage()

    return options


if __name__ == '__main__':
    '''
    options = parse_args()

    signal, sample_rate, channels = load(options.wavefile)

    if options.function.lower()=='calculatebandmax':
        CalculateBandMAX(signal, sample_rate, options.begin, options.end, options.percent)
    elif options.function.lower()=='calculatebandrms':
        for band in range(options.begin, options.end+1):
            CalculateBandRMS(signal, sample_rate, band, options.percent)
    else:
        print "Support Function:\n\tCalculateBandMAX\n\tCalculateBandRMS"

    '''
    try:
        import sys
        files = sys.argv[1:]
        if files:
            for filename in files:
                try:
                    if(os.path.isfile(filename)):
                        print("file exist")
                    else:
                        sys.exit("File not found")
#                    analyze(filename)
#                    PrintWavHeader(filename)
#                   analyze_function(filename, SaveGraph)
#                    analyze_function(filename, DrawGraph)
#                    analyze_function(filename, THDN)
                    analyze_function(filename, CalculateBandMAX)
#                    analyze_function(filename, CalculateBandRMS)

                except IOError:
                    print 'Couldn\'t analyze "' + filename + '"\n'
                print ''
        else:
            sys.exit("You must provide at least one file to analyze")
    except BaseException as e:
        print('Error:')
        print(e)
        raise
#    finally:
        # Otherwise Windows closes the window too quickly to read
#        raw_input('(Press <Enter> to close)')
#    '''
