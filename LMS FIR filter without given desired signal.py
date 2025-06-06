import numpy as np
import scipy
import scipy.signal

def lms(u, d, N, mu = 0.001):
    noise = u
    input = d
    valid_iteration = len(u) - N + 1
    w = np.zeros(N)
    y = np.zeros(valid_iteration)
    e = np.zeros(valid_iteration)

    for n in range(valid_iteration):
        x = np.flipud(noise[n:n+N])
        y[n] = np.dot(x,w)
        e[n] = input[n+N-1] - y[n]
        w = w + mu * x * e[n]
    return e, y


while True:
    try:
        desired_signal = input('Type the name of your noise file without .wav: ') + '.wav'
        input_signal = input('Type the name of your noisy input signal file without .wav: ') + '.wav'
        fs1, desired_signal = scipy.io.wavfile.read(desired_signal)
        fs2, input_signal = scipy.io.wavfile.read(input_signal)

        assert fs1 == fs2, "Sampling rates do not match!"

        desired_signal = desired_signal.astype(np.float32) #prevent overflow occurred
        input_signal = input_signal.astype(np.float32)

        min_len = min(len(desired_signal), len(input_signal)) #equalizing two files' length
        desired_signal = desired_signal[:min_len]
        input_signal = input_signal[:min_len]

        corr = scipy.signal.correlate(input_signal, desired_signal)
        lag = np.argmax(np.abs(corr)) - len(desired_signal) + 1
        aligned_noise = np.roll(desired_signal, lag)
    except FileNotFoundError:
        print("One or both files not found, try again")
    else:

        if np.max(np.abs(input_signal)) > 0:
            input_signal /= np.max(np.abs(input_signal))
        if np.max(np.abs(aligned_noise)) > 0:
            aligned_noise /= np.max(np.abs(aligned_noise))

        error_signal, output_signal = lms(aligned_noise, input_signal, 128)

        scipy.io.wavfile.write('output_signal.wav', fs1, (error_signal * 32767).astype(np.int16))
        break
