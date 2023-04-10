class MathematicalStatisticsData():
    
    def __init__(self, m_, m_2_, m_3_, m_4_, m__2, m__4):
        self.m_ = m_
        self.m_2_ = m_2_
        self.m_3_ = m_3_
        self.m_4_ = m_4_
        self.m__2 = m__2
        self.m__4 = m__4

    def getMathematicalExpectation(self):
        return self.m_

    def getInitialMomentsSecondOrder(self):
        return self.m_2_
    
    def getInitialMomentsThirdOrder(self):
        return self.m_3_
    
    def getInitialMomentsFourthOrder(self):
        return self.m_4_
    
    def getVariance(self):
        return self.m__2
    
    def getCentralMomentFunctionsFourthOrder(self):
        return self.m__4
    