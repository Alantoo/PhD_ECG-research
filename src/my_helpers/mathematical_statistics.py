from my_helpers.mathematical_statistics_data import MathematicalStatisticsData
import numpy as np
from scipy.integrate import simps

class MathematicalStatistics(MathematicalStatisticsData):
    def __init__(self, data):
        data = np.transpose(data)
        #Mathematical expectation
        self.m_ = [np.mean(i) for i in data]
        #Initial moments of the second order
        self.m_2_ = [np.sum(np.array(i)**2) / len(i) for i in data]
        #Initial moments of the third order
        self.m_3_ = [np.sum(np.array(i)**3) / len(i) for i in data]
        #Initial moments of the fourth order
        self.m_4_ = [np.sum(np.array(i)**4) / len(i) for i in data]
        #Variance
        self.m__2 = [sum((data[i] - self.m_[i])**2) / len(data[i]) for i in range(len(self.m_))]
        #Central moment functions of the fourth order
        self.m__4 = [sum((data[i] - self.m_[i])**4) / len(data[i]) for i in range(len(self.m_))]

    def getMean(self, other):
        m_ = np.mean(np.abs(np.subtract(self.m_, other.m_)))
        m_2_ = np.mean(np.abs(np.subtract(self.m_2_, other.m_2_)))
        m_3_ = np.mean(np.abs(np.subtract(self.m_3_, other.m_3_)))
        m_4_ = np.mean(np.abs(np.subtract(self.m_4_, other.m_4_)))
        m__2 = np.mean(np.abs(np.subtract(self.m__2, other.m__2)))
        m__4 = np.mean(np.abs(np.subtract(self.m__4, other.m__4)))
        return MathematicalStatisticsData(m_ * 1000, m_2_ * 1000, m_3_ * 1000, m_4_ * 1000, m__2 * 1000, m__4 * 1000)
    
    def getFourierMean(self, other):
        m_f_a_s, m_f_b_s = self.getFourierSeries(self.m_)
        m_f_a_o, m_f_b_o = self.getFourierSeries(other.m_)
        m_2_f_a_s, m_2_f_b_s = self.getFourierSeries(self.m_2_)
        m_2_f_a_o, m_2_f_b_o = self.getFourierSeries(other.m_2_)
        m_3_f_a_s, m_3_f_b_s = self.getFourierSeries(self.m_3_)
        m_3_f_a_o, m_3_f_b_o = self.getFourierSeries(other.m_3_)
        m_4_f_a_s, m_4_f_b_s = self.getFourierSeries(self.m_4_)
        m_4_f_a_o, m_4_f_b_o = self.getFourierSeries(other.m_4_)
        m__2_f_a_s, m__2_f_b_s = self.getFourierSeries(self.m__2)
        m__2_f_a_o, m__2_f_b_o = self.getFourierSeries(other.m__2)
        m__4_f_a_s, m__4_f_b_s = self.getFourierSeries(self.m__4)
        m__4_f_a_o, m__4_f_b_o = self.getFourierSeries(other.m__4)
        m_f_a = np.mean(np.abs(np.subtract(m_f_a_s, m_f_a_o)))
        m_f_b = np.mean(np.abs(np.subtract(m_f_b_s, m_f_b_o)))
        m_2_f_a = np.mean(np.abs(np.subtract(m_2_f_a_s, m_2_f_a_o)))
        m_2_f_b = np.mean(np.abs(np.subtract(m_2_f_b_s, m_2_f_b_o)))
        m_3_f_a = np.mean(np.abs(np.subtract(m_3_f_a_s, m_3_f_a_o)))
        m_3_f_b = np.mean(np.abs(np.subtract(m_3_f_b_s, m_3_f_b_o)))
        m_4_f_a = np.mean(np.abs(np.subtract(m_4_f_a_s, m_4_f_a_o)))
        m_4_f_b = np.mean(np.abs(np.subtract(m_4_f_b_s, m_4_f_b_o)))
        m__2_f_a = np.mean(np.abs(np.subtract(m__2_f_a_s, m__2_f_a_o)))
        m__2_f_b = np.mean(np.abs(np.subtract(m__2_f_b_s, m__2_f_b_o)))
        m__4_f_a = np.mean(np.abs(np.subtract(m__4_f_a_s, m__4_f_a_o)))
        m__4_f_b = np.mean(np.abs(np.subtract(m__4_f_b_s, m__4_f_b_o)))
        m_f = (m_f_a + m_f_b) / 2
        m_2_f = (m_2_f_a + m_2_f_b) / 2
        m_3_f = (m_3_f_a + m_3_f_b) / 2
        m_4_f = (m_4_f_a + m_4_f_b) / 2
        m__2_f = (m__2_f_a + m__2_f_b) / 2
        m__4_f = (m__4_f_a + m__4_f_b) / 2
        return MathematicalStatisticsData(m_f * 1000, m_2_f * 1000, m_3_f * 1000, m_4_f * 1000, m__2_f * 1000, m__4_f * 1000)

    def getFourierSeries(self, y, terms = 30, L = 1):
        x = np.linspace(0, L, self.sampling_rate, endpoint=False)
        a0 = 2./L*simps(y,x)
        an = lambda n:2.0/L*simps(y*np.cos(2.*np.pi*n*x/L),x)
        bn = lambda n:2.0/L*simps(y*np.sin(2.*np.pi*n*x/L),x)
        list_a = np.abs([a0, *[an(k) for k in range(1, terms + 1)]])
        list_b = np.abs([bn(k) for k in range(1, terms + 1)])
        return list_a, list_b
    
    def setSamplingRate(self, sampling_rate):
        self.sampling_rate = sampling_rate
        