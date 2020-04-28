#   Todo:   tune epsilon value.
#           fix this::::::       if True: #np.abs(estimated_slope) > threshold ? or not...
#           to construct support signal... sort tk by location???
#           prominence in peak finder -- and nMAX is all messed up too...
#           make J a global variable...
#           encapsulate::::::    find_transients, and one_atom_estimate

import numpy as np
from matplotlib import pyplot as plt
import pywt
from scipy import signal
import scipy.io.wavfile
import copy


class TransientDetector:
    def __init__(self, x, num_scales=65, ana_wavelet='gaus6'):
        self.x = x
        self.num_scales = num_scales
        self.ana_wavelet = ana_wavelet
        self.peak_locations = []
        self.nMAX = np.array([])
        self.cwt_extract = np.array([])
        self.dictionary = []
        self.l = int()
        self.u = int()
        self.approximation = np.array([])
        self.transients = np.array([])
        self.swt_buffer = 1

        wavelet = pywt.ContinuousWavelet(ana_wavelet)
        self.wavelet_support = wavelet.upper_bound - wavelet.lower_bound
        self.pad_width = int(self.wavelet_support * (self.num_scales + 1))

        self.x_cwt, _ = pywt.cwt(x, np.arange(1, num_scales + 1), ana_wavelet)
        self.x_cwt = self.pad_lr(self.x_cwt, self.pad_width)
        self.x = np.pad(self.x, self.pad_width)
        self.x_cwt_modulus = np.abs(self.x_cwt)
        self.x_cwt_estimate = np.zeros(self.x_cwt.shape)

    @staticmethod
    def pad_lr(x, pad_width, mode='edge'):
        return np.pad(x, ((0, 0), (pad_width, pad_width)), mode)

    @staticmethod
    def read_and_normalize_audio(filename, tukey=0.):
        fs, x = scipy.io.wavfile.read(filename)
        norm_x = x * signal.tukey(x.shape[0], alpha=tukey)  # fade in/out with Tukey window
        norm_x /= np.max(norm_x)
        return norm_x, fs

    @staticmethod
    def generate_ramp(alpha=0, alpha_1=10, buffer_length=2000, ramp_length=1000):
        ramp_signal = np.concatenate((alpha * np.ones(buffer_length),
                                      alpha + alpha_1 * np.arange(ramp_length)))
        return ramp_signal

    @staticmethod
    def least_squares(a, b):
        return float(np.dot(a[None, :], b[:, None]) / np.dot(b[None, :], b[:, None]))

    @staticmethod
    def count_sublists(list_with_sublists):
        sums = []
        for i in range(len(list_with_sublists)):
            sums.append(len(list_with_sublists[i]))
        return np.asarray(sums)

    @staticmethod
    def calculate_bounds(center, width):
        l = center - width
        u = center + width + 1
        return l, u

    @staticmethod
    def list_of_common_elements(array1, array2):
        return np.array(list(set(array1).intersection(array2)))

    @staticmethod
    def compute_gamma(s_bar, slope_s_bar, slope_s_bar_p1):
        eps = 1e-8
        numerator = np.log2((np.abs(slope_s_bar_p1) + eps) / (np.abs(slope_s_bar) + eps))
        denominator = np.log2((s_bar + 2) / (s_bar + 1))
        return (1 + numerator / denominator)[0]

    @staticmethod
    def arrange_by_loc(loc_peak_gamma):
        lpg = np.asarray(loc_peak_gamma)
        return lpg[lpg[:, 0].argsort()]

    @staticmethod
    def simple_ramp(t, tk):
        return (t - tk) * np.heaviside(t - tk, 1)

    def build_Ij(self, j):
        Ij = []
        i = 1
        while i < self.num_scales:
            inclusion_condition = np.floor(np.log2(i) + 0.5)
            if inclusion_condition == j:
                Ij.append(i - 1)  # Subtract one to convert to index number.
            elif inclusion_condition > j:
                break
            i += 1
        return Ij

    def get_modulus_peaks(self):
        for i in range(self.num_scales):
            universal_threshold = np.std(self.x_cwt[i, :]) * np.sqrt(2 * np.log(self.x_cwt.shape[1]))
            self.peak_locations.append(signal.find_peaks(np.abs(self.x_cwt)[i, :], height=universal_threshold)[0])

    def init_nMAX(self):
        num_peaks_per_scale = self.count_sublists(self.peak_locations)
        self.nMAX = np.vstack((np.arange(self.num_scales), num_peaks_per_scale)).T

    def sort_nMAX(self):
        # Sort scales by number of maxima (ascending). Return scales only.
        self.nMAX = self.nMAX[self.nMAX[:, 1].argsort()][:, 0]

    def make_nMAX(self):
        self.get_modulus_peaks()
        self.init_nMAX()
        self.sort_nMAX()

    def sort_peak_locations_by_amplitude(self):
        for s in range(self.num_scales):
            peak_amplitudes = np.abs(self.x_cwt)[s, self.peak_locations[s]]
            peak_locs_and_amps = np.vstack((self.peak_locations[s], peak_amplitudes)).T
            peak_locs_and_amps = peak_locs_and_amps[peak_locs_and_amps[:, 1].argsort()][::-1, :]  # sort by amplitude
            self.peak_locations[s] = peak_locs_and_amps[:, 0].astype(int)

    def half_scaled_wavelet_support(self, scale_index):
        return int(np.floor(self.wavelet_support * (scale_index + 1) / 2))

    def build_dictionary(self):
        singularity_loc = int(self.num_scales * self.wavelet_support)
        ramp = self.generate_ramp(0, 1, singularity_loc - 1, singularity_loc)
        ramp_cwt, _ = pywt.cwt(ramp, np.arange(1, self.num_scales + 1), self.ana_wavelet)

        for i in range(self.num_scales):
            width = self.half_scaled_wavelet_support(i)
            self.dictionary.append(ramp_cwt[i, singularity_loc - width:singularity_loc + width + 1])

    def get_cwt_extract(self, scale_index):
        self.cwt_extract = self.x_cwt[scale_index, self.l:self.u]

    def replace_cwt_extract(self, scale_index):
        self.x_cwt[scale_index, self.l:self.u] = self.cwt_extract

    def update_index_bounds(self, peak_loc, width):
        self.l, self.u = self.calculate_bounds(peak_loc, width)

    def calculate_threshold(self, scale_index):
        # return np.sqrt(np.dot(self.x_cwt[scale_index, :].T, self.x_cwt[scale_index, :]) / (scale_index + 1))
        return np.sqrt(np.dot(self.cwt_extract.T, self.cwt_extract)) / (scale_index + 1)

    def atoms_estimation_one_scale(self, s, debug=False, epsilon=1e-4, in_cone=None):
        preserved = copy.deepcopy(self.x_cwt[s, :])
        slopes_list = []
        locations_list = []
        width = self.half_scaled_wavelet_support(s)

        for peak_loc in iter(self.peak_locations[s]):

            #  Check if this peak is still significant after having removed other estimated atoms.
            if np.abs(self.x_cwt[s, peak_loc]) < epsilon:
                continue

            #  Check if this peak is in within any cone of influence of "in_cone" peaks.
            if in_cone is not None:
                if not any(np.abs(peak_loc - tk) < self.wavelet_support * (s + 1) for tk in in_cone):
                    continue

            self.update_index_bounds(peak_loc, width)
            self.get_cwt_extract(s)

            if debug:
                plt.plot(self.cwt_extract)
                plt.title('Extract before, loc = %d' % peak_loc)
                plt.show()

            estimated_slope = self.least_squares(self.cwt_extract, self.dictionary[s])
            estimated_atom = estimated_slope * self.dictionary[s]

            if debug:
                plt.plot(estimated_atom)
                plt.title('Estimated atom, loc = %d, slope = %f' % (peak_loc, estimated_slope))
                plt.show()

            threshold = self.calculate_threshold(s)

            if True: #np.abs(estimated_slope) > threshold:
                slopes_list.append(estimated_slope)
                locations_list.append(peak_loc)

            # Remove atom from global cwt
            self.cwt_extract -= estimated_atom
            self.replace_cwt_extract(s)

            if debug:
                plt.plot(self.cwt_extract)
                plt.title('Extract after, loc = %d' % peak_loc)
                plt.show()

            # Update estimate cwt
            self.x_cwt_estimate[s, self.l:self.u] += estimated_atom

        # Refresh cwt
        self.x_cwt[s, :] = preserved

        # Return list of atom locations and slopes.
        return np.vstack((locations_list, slopes_list)).T

    def calculate_J(self):
        return int(np.floor(np.log2(self.num_scales) + 0.5))

    def find_scale_least_maxima(self, Ij):
        lowest = np.inf
        for i in iter(Ij):
            check = np.where(self.nMAX == i)[0][0]
            if check < lowest:
                lowest = check
        return self.nMAX[lowest]

    def build_support_signal(self, loc_peak_gamma, s_bar):
        support_signal = np.zeros(len(self.x_cwt[0, :]))
        t = np.arange(len(self.x_cwt[0, :]))

        for lpg in iter(loc_peak_gamma):
            l, p, g = np.hsplit(lpg, 3)
            alpha_k = p * ((s_bar + 1) ** (g - 1))
            support_signal += alpha_k * self.simple_ramp(t, l)
        return support_signal

    def select_s1(self, lowest=10):
        for s in iter(self.nMAX):
            if s < lowest:
                continue
            tk_1 = self.atoms_estimation_one_scale(s)
            if len(tk_1) == 0:
                continue

            return s, tk_1

    def pad_to_mult(self, array_in, mult):
        if len(array_in) % mult is not 0:
            self.swt_buffer = mult - (len(array_in) % mult)
            return np.concatenate((array_in, np.zeros(self.swt_buffer)))
        else:
            return array_in

    def get_approximation(self, J):
        buffer = self.pad_to_mult(self.x, 2 ** (J + 1))
        self.approximation, _ = pywt.swt(buffer, 'bior2.4', 1, J)[0]

    def get_detail(self, support_signal, J, level):
        support_signal = self.pad_to_mult(support_signal, 2 ** J)
        _, detail = pywt.swt(support_signal, 'bior2.4', 1, level)[0]
        return detail

    def find_transients(self, debug=False):
        iswt_coeffs = []
        first_coeffs = True

        # Putting it all together.
        self.build_dictionary()
        self.make_nMAX()
        self.sort_peak_locations_by_amplitude()
        s1, tk_s1 = self.select_s1()

        J = self.calculate_J()
        details = []
        num_details = 0
        interim_approx = []

        self.get_approximation(J)

        for j in range(1, J + 1):
            loc_peak_gamma = []

            Ij = self.build_Ij(j)
            s_bar_j = self.find_scale_least_maxima(Ij)
            #  Find set of singularity times "t_k" at scale "s_bar," and adjacent scale "s_bar + 1"

            tk_s_bar_j = self.atoms_estimation_one_scale(s_bar_j, in_cone=tk_s1[:, 0])
            tk_s_bar_j_p1 = self.atoms_estimation_one_scale(s_bar_j + 1, in_cone=tk_s1[:, 0])

            tk_s_bar_locs = tk_s_bar_j[:, 0]
            tk_s_bar_p1_locs = tk_s_bar_j_p1[:, 0]

            common_maxima = self.list_of_common_elements(tk_s_bar_locs, tk_s_bar_p1_locs)

            if len(common_maxima) == 0:
                continue

            for peak_loc in iter(common_maxima):
                gamma_k = self.compute_gamma(s_bar_j,
                                             tk_s_bar_j[np.where(tk_s_bar_locs == peak_loc), 1],
                                             tk_s_bar_j_p1[np.where(tk_s_bar_p1_locs == peak_loc), 1])

                loc_peak_gamma.append(np.concatenate((np.array([peak_loc]),
                                                      tk_s_bar_j[np.where(tk_s_bar_locs == peak_loc), 1][0],
                                                      gamma_k)))

                if debug:
                    plt.plot(support_signal)
                    plt.show()

            support_signal = self.build_support_signal(loc_peak_gamma, s_bar_j)
            details.append(self.get_detail(support_signal, J, j))
            num_details += 1

        # Reconstruct interim approximations for reconstruction
        approx = self.approximation
        interim_approx.append(approx)
        for d in range(num_details):
            det = details[(num_details - 1) - d]
            interim_coeffs = (approx, det)
            approx = pywt.iswt(interim_coeffs, 'bior2.4')
            interim_approx.append(approx)  # rebuild with det and approx

        interim_approx.reverse()

        for i in range(num_details):
            iswt_coeffs.append((interim_approx[i], details[i]))

        self.transients = pywt.iswt(iswt_coeffs, 'bior2.4')
        self.transients = self.transients[self.pad_width: -(self.pad_width + self.swt_buffer)]


# Tests...
def test(noise=0.01, plot_x=False, exponential=False, freq=None, repeats=0):
    t = np.arange(2000)
    x = TransientDetector.simple_ramp(t, 1900)
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

    td = TransientDetector(x, num_scales=400)
    td.find_transients()
    return td


def test_real_audio(plot_x=True):
    x, fs = TransientDetector.read_and_normalize_audio('/content/drive/My Drive/glock_demo.wav', tukey=0)
    x = x[:2*fs]

    if plot_x:
        plt.plot(x)
        plt.title('input signal')
        plt.show()

    td = TransientDetector(x, num_scales=400)
    td.find_transients()
    return td


if __name__ == '__main__':
    td = test(noise=0.0, plot_x=True, exponential=False, freq=None, repeats=2)
    plt.plot(td.transients)
    plt.show()

    td = test(noise=0.0, plot_x=True, exponential=True, freq=None, repeats=2)
    plt.plot(td.transients)
    plt.show()

    td = test(noise=0.0, plot_x=True, exponential=True, freq=0.005, repeats=2)
    plt.plot(td.transients)
    plt.show()

    td = test(noise=0.01, plot_x=True, exponential=True, freq=0.005, repeats=2)
    plt.plot(td.transients)
    plt.show()

    td = test_real_audio()