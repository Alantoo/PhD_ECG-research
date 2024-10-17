from loguru import logger
import scipy.io

class ReadMatFile:

    def __init__(self, ecg_config):
        data_path = f'{ecg_config.getFileName()}.{ecg_config.getDataType()}'
        logger.info("Read Mat file")
        logger.info(data_path)

        data = scipy.io.loadmat(data_path)

        if "signal" in data:

            self.signals = data["signal"].transpose()

            result = [arr.item().strip() for arr in data["siginfo"]["Description"][0]]

            self.sampling_rate = int((data["Fs"])[0][0])
            logger.info(f'Fileds: {result}')
            logger.info(f'Sampling rate: {self.sampling_rate}')

        elif "OUTPUT" in data:
            output = self.get_nested(data, "OUTPUT", "ECGanalysis", 0, 0, "x", 0, 0)
            self.FourierSeries_FourierCoefficients = self.get_nested(output, "FourierSeries", 0, 0, "FourierCoefficients", 0, 0, 0) # Fourier coefficients
            self.FourierSeries_Harmonics = self.get_nested(output, "FourierSeries", 0, 0, "Harmonics", 0, 0, 0) # Harmonics
            self.cycRSCP1_Ryxa = self.get_nested(output, "cycRSCP1", 0, 0, "Ryxa", 0, 0, 0)  # cyclic autocorrelation of x(t)
            self.cycRSCP1_Cyxa = self.get_nested(output, "cycRSCP1", 0, 0, "Cyxa", 0, 0, 0)  # cyclic autocovaraince of x(t)
            self.cycRSCP1_Tau1 = self.get_nested(output, "cycRSCP1", 0, 0, "Tau1", 0, 0, 0)  # tau axis for cyclic autocorrelation and autocovariance
            self.cycRSCP1_Syxa = self.get_nested(output, "cycRSCP1", 0, 0, "Syxa", 0, 0, 0)  # cyclic spectrum
            self.cycRSCP1_Pyxa = self.get_nested(output, "cycRSCP1", 0, 0, "Pyxa", 0, 0, 0)  # 2nd-order cyclic polyspectrum
            self.cycRSCP1_F1 = self.get_nested(output, "cycRSCP1", 0, 0, "F1", 0, 0, 0)     # frequency axis for cyclic spectrum and 2nd-order cyclic polyspectrum

            self.stationary_cycRSCP1_Ryxa = self.get_nested(output, "stationary", 0, 0, "cycRSCP1", 0, 0, "Ryxa", 0, 0, 0)  # autocorrelation of x(t)
            self.stationary_cycRSCP1_Cyxa = self.get_nested(output, "stationary", 0, 0, "cycRSCP1", 0, 0, "Cyxa", 0, 0, 0)  # autocovariance of x(t)
            self.stationary_cycRSCP1_Tau1 = self.get_nested(output, "stationary", 0, 0, "cycRSCP1", 0, 0, "Tau1", 0, 0, 0)  # tau axis for autocorrelation and autocovariance
            self.stationary_cycRSCP1_Syxa = self.get_nested(output, "stationary", 0, 0, "cycRSCP1", 0, 0, "Syxa", 0, 0, 0)  # power spectral density
            self.stationary_cycRSCP1_Pyxa = self.get_nested(output, "stationary", 0, 0, "cycRSCP1", 0, 0, "Pyxa", 0, 0, 0)  # 2nd-order polyspectrum
            self.stationary_cycRSCP1_F1 = self.get_nested(output, "stationary", 0, 0, "cycRSCP1", 0, 0, "F1", 0, 0, 0)    # frequency axis for power spectral density and 2nd-order polyspectrum
        else:
            self.RD_Ryxa = self.get_nested(data, "OUTPUT_RD", 0, 0, "Ryxa", 0) # cyclic autocorrelation of x(t)
            self.RD_Tau1 = self.get_nested(data, "OUTPUT_RD", 0, 0, "Tau1", 0) # tau axis for cyclic autocorrelation
            self.RD_Pyxa = self.get_nested(data, "OUTPUT_RD", 0, 0, "Pyxa", 0) # 2nd-order cyclic polyspectrum
            self.RD_F1 = self.get_nested(data, "OUTPUT_RD", 0, 0, "F1", 0) # frequency axis for 2nd-order cyclic polyspectrum
            self.RD_FourierCoefficients = self.get_nested(data, "OUTPUT_RD", 0, 0, "FourierCoefficients", 0) # Fourier coefficients
            self.RD_Harmonics = self.get_nested(data, "OUTPUT_RD", 0, 0, "Harmonics", 0) # Harmonics


    def get_nested(self, data, *path):
        for key in path:
            try:
                data = data[key]
            except (KeyError, IndexError):
                return None
        return data