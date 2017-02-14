# python 2.7
import pylab
import random
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

    self.filter_C.append(self.C)  # Liste - Analyse Fehler


### Lineare Kalman Glaetter
class lineare_kalman_glaetter:
  def __init__(self, _M, _C, _C_hut, _m, _m_hut):
    self.M = _M                 # Zustandswandelmatrix
    self.C = _C                 # Analyse Fehler
    self.C_hut = _C_hut         # Vorhersage Fehler
    self.m = _m                 # Analyse Zustand
    self.m_hut = _m_hut         # Vorhersage Zustand

  def schritt_glaetter(self, m_check_1, C_check_1):

    # Kalman Glaetten Gain
    L = self.C * numpy.transpose(self.M) * numpy.linalg.inv(self.C_hut)

    # geglaetteter Durchschnittswert
    self.m_check = self.m + L * (m_check_1 - self.m_hut)

    # geglaettete Kovarianz
    self.C_check = self.C + L * (C_check_1 - self.C_hut) * numpy.transpose(L)


  def geglaettet(self):
    return self.m_check


### Beispiel: Freier Fall
class fall:
  g = 1
  h = -(g/2)

  def __init__(self, _wahr_ort, _wahr_gkt, _rausch):
    self.wahr_ort = _wahr_ort       # wahrer Ort
    self.wahr_gkt = _wahr_gkt       # wahre Geschwindigkeit
    self.rausch = _rausch           # Rausch

  # gemessener Fall mit Rausch
  def ort_mit_rausch(self):
    # zufaellige Geschwindigkeit
    self.gkt = random.gauss(self.wahr_gkt, self.rausch)
    ort = self.wahr_ort
    ort += self.gkt + self.h
    return ort

  def gkt_mit_rausch(self):
    return self.gkt

  # wahrer Fall ohne Rausch
  def ort(self):
    # Geschwindigkeit wird mit jedem Schritt um 1 verringert
    self.wahr_gkt -= 1
    self.wahr_ort += self.wahr_gkt + self.h
    return self.wahr_ort


# Wir beobachten ein Gegenstand, der in einem konstanten Gravitationsfeld faellt.
# y(t) sei die Hoehe des Gegenstands.
# y(t) = y(t_0) + y'(t_0)(t - t_0) - (g/2)(t - t_0)^2
# Die Zeit ist diskret mit t - t_0 = 1
# Sei j := t_0 und j + 1 = t

# y(j+1) = y(j) + y'(j) - g/2
# Die Hoehe y(j+1) haengt von der vorherigen Geschwindigkeit y'(j) und die Hoehe zum Zeitpunkt j, y(j), ab.

# m(j+1) = Zustandswandelmatrix M * Anfangszustandschaetzung m + Kontrollmatrix B * Kontrollvektor
#            1, 1          0.5, 0              Ort
# m(j+1)  =  0, 1 m(j) +    0,  1   (-g)       Geschwindigkeit

M = numpy.matrix([[1,1], [0,1]])

# Schaetzung fuer den Anfangszustand
m = numpy.matrix([[100], [0]])

# Kontrollmatrix B
B = numpy.matrix([[0.5,0], [0,1]])

# Kontrollvektor
kontrollvektor = numpy.matrix([[-1], [-1]])

# Beobachtungsmatrix H
H = numpy.matrix([[1,0], [0,0]])

# Anfangsschaetzung C fuer Fehler in m
C = numpy.eye(2) * 0.01

# geschaetzte Messfehler sigma im Verfahren (0, weil wir Gleichungen direkt von Simulation entwickelt haben)
sigma = numpy.zeros(2)

# geschaetzte Messfehler in Daten (variabel)
gamma = numpy.eye(2)

# Klassen
kf = lineare_kalman_filter(M, B, H, m, C, sigma, gamma)
freier_fall = fall(100, 1, 5) # fall(wahr_ort, wahr_gkt, rausch)

# Arrays, um Datenpunkten, die wir am Ende plotten wollen, zu speichern
gemessen = []
wahr = []
filter = []
glaetter = []
iterationen = 11     # Anzahl der Iterationen fuer die Simulation

# Iteration fuer die Simulation
for i in range(iterationen):
  # gemessener Fall
  gem_fall = freier_fall.ort_mit_rausch()
  gemessen.append(gem_fall)       # Ort mit Rausch

  # wahrer Fall
  wahr.append(freier_fall.ort())  # wahrer Ort ohne Rausch

  # Kalman Filter
  kf.schritt(kontrollvektor,                   # schritt(kontrollvektor, Y)
             numpy.matrix([[gem_fall],
             [freier_fall.gkt_mit_rausch()]])) # (ort, geschwindigkeit)

  filter.append(kf.aktueller_zustand()[0,0])   # aktueller_zustand(m)

# Listen zur Kontrolle
print 'Vorhersage m_hut: \n', kf.filter_m_hut, '\n'
print 'Vorhersage C_hut: \n', kf.filter_C_hut, '\n'
print 'Analyse m: \n', kf.filter_m, '\n'
print 'Analyse C: \n', kf.filter_C, '\n'
print 'Fall mit Rausch: \n', gemessen, '\n'

# Liste fuer geglaetteter Durchschnittswert
m = []
m.append(kf.filter_m[-1])
glaetter.append(m[0][0]) # letzter geglaetteter Wert

# Liste fuer geglaettete Kovarianz
C = []
C.append(kf.filter_C[-1])

# Kalman Glaetter, Rueckwaertspass
for j in reversed(range(iterationen-1)):
  ks = lineare_kalman_glaetter(M, kf.filter_C[j],
                               kf.filter_C_hut[j + 1],
                               kf.filter_m[j],
                               kf.filter_m_hut[j + 1])
  ks.schritt_glaetter(m[-1], C[-1])            # schritt_glaetter(m_check_1, C_check_1)

  m.append(ks.m_check)
  C.append(ks.C_check)
  glaetter.append(ks.geglaettet()[0,0])        # geglaetteter Durchschnittswert

# Listen zur Kontrolle
print 'smooth m: \n', m , '\n'
print 'smooth C: \n', C, '\n'
print 'reversed glaetter: \n', glaetter[::-1]


# Alle Ergebnisse plotten
import seaborn as sns
sns.set()
pylab.plot(range(iterationen), gemessen, ':',
           range(iterationen), wahr,
           range(iterationen), filter, '--',
           range(iterationen), glaetter[::-1], '--')
pylab.xlabel('Zeit')
pylab.ylabel('Hoehe')
pylab.title('Freier Fall')
pylab.legend(('gemessen', 'wahr', 'Filter', 'Glaetter'))
pylab.show()