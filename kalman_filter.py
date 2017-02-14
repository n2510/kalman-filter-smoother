import numpy

### Lineare Kalman Filter
class lineare_kalman_filter:

  def __init__(self,_M, _B, _H, _m, _C, _sigma, _gamma):
    self.M = _M                      # Zustandswandelmatrix
    self.B = _B                      # Kontrollmatrix
    self.H = _H                      # Beobachtungsmatrix
    self.m = _m                      # Anfangsschaetzung [Zustand]
    self.C = _C                      # Anfangsschaetzung [Fehler in m]
    self.sigma = _sigma              # geschaetzte Messfehler [im Verfahren]
    self.gamma = _gamma              # geschaetzte Messfehler [in Daten]

    self.filter_m_hut = []           # Liste - Vorhersage Zustand
    self.filter_C_hut = []           # Liste - Vorhersage Fehler
    self.filter_m = []               # Liste - Analyse Zustand
    self.filter_C = []               # Liste - Analyse Fehler

  def aktueller_zustand(self):
    return self.m

  def schritt(self, kontrollvektor, Y):

    ## 1. Vorhersage

    # Durchschnittswert (schaetzen, wo wir sein werden) :
    m_hut = self.M * self.m + self.B * kontrollvektor

    self.filter_m_hut.append(m_hut)  # Liste - Vorhersage Zustand

    # Kovarianzmatrix (schaetzen, wie viel Fehler es geben wird) :
    C_hut = (self.M * self.C) * numpy.transpose(self.M) + self.sigma

    self.filter_C_hut.append(C_hut)  # Liste - Vorhersage Fehler

    ## 2. Analyse

    # Kalman Gain Matrix (Schaetzungen maessigen) :
    K = C_hut * numpy.transpose(self.H) * numpy.linalg.inv(self.H * C_hut * numpy.transpose(self.H) + self.gamma)

    # aktuellste Schaetzung vom wahren Zustand (neue Schaetzung, von wo wir sind) :
    self.m = m_hut + K * (Y - self.H * m_hut)

    self.filter_m.append(self.m)  # Liste - Analyse Zustand

    # aktuellste Schaetzung vom durschnittlichen Fehler (neue Schaetzung vom Fehler) :
    self.C = (numpy.eye(self.C.shape[0]) - K * self.H) * C_hut

    self.filter_C.append(self.C) # Liste - Analyse Fehler