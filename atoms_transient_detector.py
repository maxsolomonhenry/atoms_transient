import pywt
import copy
import numpy as np
import scipy.io.wavfile
import scipy.signal

from matplotlib import pyplot as plt


class TransientDetector:
    """Transient detection and extraction using matching pursuit in the wavelet domain.
    
    This class extracts transient information of an audio signal via a reconstructed
    "support signal," which attempts to isolate transient-only information from the
    continuous wavelet transform. By finding wavelet-domain peaks, and tracing them
    through scale values within their respective cones-of-influence, the algorithm
    attempts to reconstruct a signal whose detail band is identical to a transient-only
    signal (the support signal). This is then converted to the time domain via a
    inverse nondecimated wavelet transform. Details of the algorithm can be seen in the
    reference:
    
    V. Bruni, S. Marconi, and D. Vitulano, “Time-scale atoms chains for transients detection 
        in audio signals,” IEEE Trans. Audio Speech Lang. Process., vol. 18, no. 3, pp. 420–433,
        Mar. 2010, doi: 10.1109/TASL.2009.2032623.
    
    """
    
    def __init__(self, x, fs=44100, num_scales=100, wavelet_type='gaus6', swt_wavelet_type='bior2.4'):
        self.x = x
        self.fs = fs
        self.S = num_scales
        self.wavelet_type = wavelet_type

        self.swt_wavelet_type = swt_wavelet_type
        self.cwt_ = np.array([])
        self.nMAX = np.array([])
        self.y = np.array([])
        self.dictionary = []
        self.SUPPORT = int()
        self.PAD_WIDTH = int()
        self.S1 = int()
        self.J = int()
        self.THRESH_COEF = 0.75
        self.TUNING_COEF = 0.659659659659660
        self.EPSILON = 1e-10
        self.swt_buffer_length = 0

    @staticmethod
    def get_ramp(t0=1000):
        t = np.arange(t0 * 2)
        return np.maximum(0, t - t0)

    @staticmethod
    def get_universal_threshold(cwt_1scale):
        return np.std(cwt_1scale) * np.sqrt(2 * np.log(cwt_1scale.size))

    @staticmethod
    def lh_2_array(locs_heights):
        return np.vstack((locs_heights[0], locs_heights[1]['peak_heights']))

    @staticmethod
    def sort_row2_dec(an_array):
        return an_array[:, (-an_array[1, :]).argsort()]

    @staticmethod
    def least_squares(a, b):
        return float(np.dot(a[None, :], b[:, None]) / np.dot(b[None, :], b[:, None]))

    @staticmethod
    def get_iswt_coeffs(approx, detail_bands):
        # Note: all but top-level approximation is ignored in inverse SWT.
        iswt_coeffs =[]
        num_bands = len(detail_bands)

        for i in range(num_bands):
            rev_index = num_bands - i - 1
            iswt_coeffs.append([approx, detail_bands[rev_index]])
        return iswt_coeffs

    def master_algorithm(self):
        self.set_cwt_()
        self.set_dictionary()
        self.set_nMAX()
        self.S1 = self.nMAX[0]

        # Gather "master atoms" locations and fitted amplitudes.
        S1_atom_locs, S1_atom_amps = self.get_atoms(self.S1)
        self.set_J()

        # Collect detail bands for 1, ..., J
        detail_bands = self.get_detail_bands(S1_atom_locs)
        approximation = self.get_approximation(self.x, self.J)

        iswt_coeffs = self.get_iswt_coeffs(approximation, detail_bands)
        self.y = pywt.iswt(iswt_coeffs, self.swt_wavelet_type)

        # Remove padding.
        self.y = self.y[self.PAD_WIDTH: -(self.PAD_WIDTH + self.swt_buffer_length)]

    def set_cwt_(self):
        self.set_wavelet_info()
        self.apply_pad_x()
        cwt_, _ = pywt.cwt(self.x, range(1, self.S + 1), self.wavelet_type)
        self.cwt_ = self.undo_boundary_effect(cwt_)

    def set_dictionary(self):
        t0 = int(self.PAD_WIDTH)
        ramp = self.get_ramp(t0)
        ramp = ramp/np.max(ramp) * self.TUNING_COEF
        ramp_cwt, _ = pywt.cwt(ramp, range(1, self.S + 1), self.wavelet_type)

        for s in range(self.S):
            width = int(np.floor(self.SUPPORT * (s + 1) / 2))
            self.dictionary.append(ramp_cwt[s, t0 - width + 1:t0 + width + 1])

    def set_nMAX(self):
        num_peaks_array = self.get_num_peaks()
        self.nMAX = np.argsort(num_peaks_array)

    def get_atoms(self, s, search_COIs=None):
        cwt_1scale = copy.deepcopy(self.cwt_[s, :])
        M_l, heights = self.get_sorted_peaks(cwt_1scale)
        N_l = M_l.size

        atom_locs = []
        atom_amps = []

        threshold = np.sqrt(np.dot(cwt_1scale, cwt_1scale)/(s + 1))

        for i in range(N_l):
            l, u = self.get_extract_bounds(s, M_l[i])
            extract = cwt_1scale[l:u]
            amplitude = self.least_squares(extract, self.dictionary[s])

            if search_COIs is None:
                is_in_COI = True
            else:
                is_in_COI = self.check_in_COI(s, M_l[i], search_COIs)

            if (np.abs(amplitude) > threshold) & is_in_COI:
                atom_locs.append(M_l[i])
                atom_amps.append(amplitude)
                cwt_1scale[l:u] -= extract

        return atom_locs, atom_amps

    def set_J(self):
        self.J = int(np.floor(np.log2(self.S) + 0.5))

    def get_detail_bands(self, S1_atom_locs):
        detail_bands = []

        for j in range(1, self.J + 1):
            Ij = self.get_Ij(j)

            sbar = self.get_scale_with_least_peaks(Ij)
            sbar_locs, sbar_amps = self.get_atoms(sbar, search_COIs=S1_atom_locs)
            adj_locs, adj_amps = self.get_atoms(sbar + 1, search_COIs=S1_atom_locs)
            gammas = self.get_gammas(sbar, sbar_amps, adj_amps)

            support = self.get_support(sbar, sbar_locs, sbar_amps, gammas)
            detail_bands.append(self.get_detail(support, j))
        return detail_bands


    def get_detail(self, support, j):
        detail = self.get_swt(support, j, 'detail')
        return detail

    def get_approximation(self, signal, level):
        approx = self.get_swt(signal, level, 'approx')
        return approx

    def get_prepad_length(self, support):
        return len(support) + 2 ** (self.J + 1) - (len(support) % 2 ** (self.J + 1))

    def get_swt(self, signal, level, band_choice):
        output = np.array([])
        prepad_length = self.get_prepad_length(signal)

        # Overpad to account for border effects.
        signal = self.pad_to_mult(signal, 2 ** (self.J + 2))
        approx, detail = pywt.swt(signal, self.swt_wavelet_type, 1, level)[0]

        # Remove pad and choose output.
        if band_choice == 'approx':
            output = approx[:prepad_length]
        elif band_choice == 'detail':
            output = detail[:prepad_length]

        return output

    def pad_to_mult(self, support, mult):
        if len(support) % mult == 0:
            return support
        else:
            self.swt_buffer_length = mult - (len(support) % mult)
            return pywt.pad(support, (0, self.swt_buffer_length), mode='smooth')
            # return np.concatenate((support, np.zeros(self.swt_buffer_length)))

    def set_wavelet_info(self):
        wavelet = pywt.ContinuousWavelet(self.wavelet_type)
        self.SUPPORT = wavelet.upper_bound - wavelet.lower_bound
        self.PAD_WIDTH = int(self.SUPPORT * (self.S + 1))

    def apply_pad_x(self):
        self.x = pywt.pad(self.x, self.PAD_WIDTH, 'antireflect')

    def undo_boundary_effect(self, cwt_):
        for s in range(self.S):
            boundary = int(self.SUPPORT * (s + 2))
            cwt_[s, :boundary] *= 0
            cwt_[s, -boundary:] *= 0
        return cwt_

    def get_peaks_locs(self, cwt_1scale):
        threshold = self.THRESH_COEF * self.get_universal_threshold(cwt_1scale)

        # if True:
        #     locs = scipy.signal.find_peaks(np.abs(cwt_1scale), height=threshold, distance=int(self.fs/100))[0]
        #     plt.plot(cwt_1scale)
        #     plt.scatter(locs, cwt_1scale[locs.astype('int')], c='m')
        #     plt.show()

        return scipy.signal.find_peaks(np.abs(cwt_1scale), height=threshold, distance=int(self.fs/100))

    def get_num_peaks_one(self, cwt_1scale):
        return self.get_peaks_locs(cwt_1scale)[0].size

    def get_num_peaks(self):
        num_peaks = []
        for s in range(self.S):
            num_peaks.append(self.get_num_peaks_one(self.cwt_[s, :]))
        return np.asarray(num_peaks)

    def get_sorted_peaks(self, cwt_1scale):
        locs_heights = self.get_peaks_locs(cwt_1scale)
        locs_heights = self.lh_2_array(locs_heights)
        locs_heights = self.sort_row2_dec(locs_heights)
        return locs_heights[0, :], locs_heights[1, :]

    def get_extract_bounds(self, s, t_i):
        l = int(t_i - np.floor(self.SUPPORT * (s + 1) / 2))
        u = int(t_i + np.floor(self.SUPPORT * (s + 1) / 2))
        return l, u

    def check_in_COI(self, s, query_location, S1_atom_locs):
        return any(np.abs(query_location - tk) < self.SUPPORT * (s + 1) for tk in S1_atom_locs)

    def get_Ij(self, j):
        Ij = []
        i = 1
        while i < self.S:
            condition = np.floor(np.log2(i) + 0.5)
            if condition == j:
                Ij.append(i - 1)  # Subtract one to convert to index number.
            elif condition > j:
                break
            i += 1
        return Ij

    def get_scale_with_least_peaks(self, Ij):
        lowest = np.inf
        for ij in Ij:
            query = np.where(self.nMAX==ij)[0][0]
            if query < lowest:
                lowest = int(query)
        return self.nMAX[lowest]

    def get_1gamma(self, s_bar, amp_s1, amp_s2):
        num = np.log2((np.abs(amp_s2) + self.EPSILON) / (np.abs(amp_s1) + self.EPSILON))
        denom = np.log2((s_bar + 2) / (s_bar + 1))
        return 1 + (num / denom)

    def get_gammas(self, sbar, sbar_amps, adj_amps):
        gammas = []
        for amp_s1, amp_s2 in zip(sbar_amps, adj_amps):
            gammas.append(self.get_1gamma(sbar, amp_s1, amp_s2))
        return gammas

    def get_support(self, s, locs, amps, gammas):
        cwt_length = self.cwt_.shape[1]
        support = np.zeros(cwt_length)
        t = np.arange(cwt_length)

        for l, a, g in zip(locs, amps, gammas):
            slope = a * ((s + 1) ** (g - 1))
            support += slope * np.maximum(0, t - l)

        # support /= (s+1)
        return support


# Testing and debugging
def get_test_signal(N=2000, enable_noise=False):
    t = np.arange(N)
    t0 = np.floor(N / 2)
    ramp = np.maximum(0, t - t0)
    if enable_noise:
        ramp += np.random.rand(N) * np.max(ramp) / 100
    return ramp / np.max(ramp)


def get_audio(filename):
    fs, x = scipy.io.wavfile.read(filename)
    norm_x = x/np.max(x)
    return norm_x, fs


def simple_ramp(t, tk):
    return (t - tk) * np.heaviside(t - tk, 1)


def notebook_test(noise=0, plot_x=False, exponential=False, freq=None, repeats=0):
    t = np.arange(2000)
    x = simple_ramp(t, 1900)
    x /= np.max(x)
    if exponential:
        x = np.concatenate((x, np.exp(-t/250)))
    else:
        x = np.concatenate((x, np.arange(1999, 0, -1)/2000, np.zeros(1900)))

    for i in range(repeats):
        x = np.concatenate((x, x))

    if freq is not None:
        x *= np.cos(2 * np.pi * freq * np.arange(len(x)))

    x += np.random.rand(len(x))*noise
    if plot_x:
        plt.plot(x)
        plt.title('input signal')
        plt.show()

    td = TransientDetector(x)
    td.master_algorithm()
    plt.plot(td.y)
    plt.title('Extracted transient')
    plt.xlabel('time index')
    plt.show()


def test(use_real_audio=False):
    if use_real_audio:
        x, _ = get_audio('glock_demo.wav')
    else:
        x = get_test_signal()
    td = TransientDetector(x)
    # td.TUNING_COEF=1
    td.master_algorithm()
    plt.plot(td.y)
    plt.title('Extracted transient')
    plt.xlabel('time index')
    plt.ylim([-5, 5])
    plt.show()



if __name__ == '__main__':
    test(use_real_audio=True)
    # notebook_test(noise=0.05, plot_x=True, repeats=4, exponential=True, freq=0.005)
