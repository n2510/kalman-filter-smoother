import numpy

### Lineare Kalman Smoother
class lineare_kalman_glaetter:
  def __init__(self, _M, _C, _C_hut, _m, _m_hut):
    self.M = _M          # Zustandswandelmatrix
    self.C = _C          # Analyse Fehler
    self.C_hut = _C_hut  # Vorhersage Fehler
    self.m = _m          # Analyse Zustand
    self.m_hut = _m_hut  # Vorhersage Zustand

  def schritt_glaetter(self, m_check_1, C_check_1):

    # Kalman Smoothing Gain
    L = self.C * numpy.transpose(self.M) * numpy.linalg.inv(self.C_hut)

    # geglaetteter Durchschnittswert
    self.m_check = self.m + L * (m_check_1 - self.m_hut)

    # geglaettete Kovarianz
    self.C_check = self.C + L * (C_check_1 - self.C_hut) * numpy.transpose(L)

  def geglaettet(self):
    return self.m_check

