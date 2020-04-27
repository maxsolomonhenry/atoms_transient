import pywt
import copy
import numpy as np
import scipy.io.wavfile
import scipy.signal

from matplotlib import pyplot as plt


class TransientDetector:
    def __init__(self, x, num_scales=65, wavelet_type='gaus6'):
        self.x = x
        self.S = num_scales
        self.wavelet_type = wavelet_type

        self.cwt_ = np.array([])
        self.nMAX = np.array([])
        self.dictionary = []
        self.SUPPORT = int()
        self.PAD_WIDTH = int()
        self.S1 = int()
        self.TUNING_COEF = 0.659659659659660
        self.EPSILON = 1e-10

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

    def set_wavelet_info(self):
        wavelet = pywt.ContinuousWavelet(self.wavelet_type)
        self.SUPPORT = wavelet.upper_bound - wavelet.lower_bound
        self.PAD_WIDTH = self.SUPPORT * (self.S + 1)

    def apply_pad_x(self):
        self.x = pywt.pad(self.x, self.PAD_WIDTH, 'antireflect')

    def undo_boundary_effect(self, cwt_):
        for s in range(self.S):
            boundary = int(self.SUPPORT * (s + 1))
            cwt_[s, :boundary] *= 0
            cwt_[s, -boundary:] *= 0
        return cwt_

    def set_cwt_(self):
        self.set_wavelet_info()
        self.apply_pad_x()
        cwt_, _ = pywt.cwt(self.x, range(1, self.S + 1), self.wavelet_type)
        self.cwt_ = self.undo_boundary_effect(cwt_)

    def get_peaks_locs(self, cwt_1scale):
        threshold = self.get_universal_threshold(cwt_1scale)
        return scipy.signal.find_peaks(np.abs(cwt_1scale), height=threshold)

    def get_num_peaks_one(self, cwt_1scale):
        return self.get_peaks_locs(cwt_1scale)[0].size

    def get_num_peaks(self):
        num_peaks = []
        for s in range(self.S):
            num_peaks.append(self.get_num_peaks_one(self.cwt_[s, :]))
        return np.asarray(num_peaks)

    def set_nMAX(self):
        num_peaks_array = self.get_num_peaks()
        self.nMAX = np.argsort(num_peaks_array)

    def set_dictionary(self):
        t0 = int(self.PAD_WIDTH)
        ramp = self.get_ramp(t0)
        ramp = ramp/np.max(ramp) * self.TUNING_COEF
        ramp_cwt, _ = pywt.cwt(ramp, range(1, self.S + 1), self.wavelet_type)

        for s in range(self.S):
            width = int(np.floor(self.SUPPORT * (s + 1) / 2))
            self.dictionary.append(ramp_cwt[s, t0 - width + 1:t0 + width + 1])

    def get_sorted_peaks(self, cwt_1scale):
        locs_heights = self.get_peaks_locs(cwt_1scale)
        locs_heights = self.lh_2_array(locs_heights)
        locs_heights = self.sort_row2_dec(locs_heights)
        return locs_heights[0, :], locs_heights[1, :]

    def get_extract_bounds(self, s, t_i):
        l = int(t_i - np.floor(self.SUPPORT * (s + 1) / 2))
        u = int(t_i + np.floor(self.SUPPORT * (s + 1) / 2))
        return l, u

    def get_atoms(self, s):
        cwt_1scale = copy.deepcopy(self.cwt_[s, :])
        M_l, heights = self.get_sorted_peaks(cwt_1scale)
        N_l = M_l.size

        atom_locs = []
        atom_amps = []

        k = 0

        for i in range(N_l):
            l, u = self.get_extract_bounds(s, M_l[i])
            extract = cwt_1scale[l:u]
            amplitude = self.least_squares(extract, self.dictionary[s])

            # if True:
            #     plt.plot(extract)
            #     plt.plot(self.dictionary[s])
            #     plt.show()

            if amplitude > self.EPSILON:
                atom_locs.append(M_l[i])
                atom_amps.append(amplitude)
                cwt_1scale[l:u] -= extract

        return atom_locs, atom_amps

    def master_algorithm(self):
        self.set_cwt_()
        self.set_dictionary()
        self.set_nMAX()
        self.S1 = self.nMAX[0]

        S1_atom_locs, S1_atom_amps = self.get_atoms(self.S1)
        


# Testing and debugging
def get_test_signal(N=2000, enable_noise=False):
    t = np.arange(N)
    t0 = np.floor(N / 2)
    ramp = np.maximum(0, t - t0)
    if enable_noise:
        ramp += np.random.rand(N) * np.max(ramp) / 100
    return ramp / np.max(ramp)


def test():
    x = get_test_signal()
    td = TransientDetector(x)
    td.master_algorithm()
    # for s in range(5):
    #     plt.plot(td.dictionary[s])
    #     plt.plot(np.floor(len(td.dictionary[s])/2), 0, '*')
    #     plt.show()
    return


if __name__ == '__main__':
    test()
