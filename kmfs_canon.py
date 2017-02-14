# Lineare Kalman Filter und Glaetter fuer mehrere Variablen
# http://greg.czerniak.info/guides/kalman1/

# Python 2.7
import pylab
import math
import random
import numpy

# Lineare Kalman Filter
from kalman_filter import lineare_kalman_filter

# Lineare Kalman Smoother
from kalman_smoother import lineare_kalman_glaetter

## Beispiel: Ball aus einer Kanone schiessen - parabelfoermige Laufbahn
class kanone:
  ## Variablen:
  winkel = 45                                                  # Der Winkel der Kanone vom Boden
  mu_gkt = 100                                                 # Muendungsgeschwindigkeit der Kanone
  schwerkraft = [0, -9.81]                                     # Ein Vektor mit Schwerkraftbeschleunigung
  geschwindigkeit = [mu_gkt * math.cos(winkel * math.pi/180),  # Anfangsgeschwindigkeit des Kanonenballs
                     mu_gkt * math.sin(winkel * math.pi/180)]
  ort = [0,0]                                                  # Anfangsort des Kanonenballs
  beschleunigung = [0,0]                                       # Anfangsbeschleunigung des Kanonenballs

  ## Methoden:
  def __init__(self, _zeitanteil, _rauschpegel):
    self.zeitanteil = _zeitanteil
    self.rauschpegel = _rauschpegel

  def add(self, x, y):
    return x + y

  def mult(self, x, y):
    return x * y

  def ort_x(self):
    return self.ort[0]

  def ort_y(self):
    return self.ort[1]

  def ort_x_mit_rausch(self):
    return random.gauss(self.ort_x(), self.rauschpegel)

  def ort_y_mit_rausch(self):
    return random.gauss(self.ort_y(), self.rauschpegel)

  def x_geschwindigkeit(self):
    return self.geschwindigkeit[0]

  def y_geschwindigkeit(self):
    return self.geschwindigkeit[1]

  ## Erhoehung durch den naechsten Zeitanteil der Simulation
  def kanone_schritt(self):

    # Mit dem Zeitanteilvektor werden wir alles in Zeitanteilen zuteilen:
    zeitanteilvek = [self.zeitanteil, self.zeitanteil]

    # Schwerkraft in ein kleinerer Zeitanteil zuteilen:
    # = [ 0 * zeitanteil, -9.81 * zeitanteil]
    teil_schwerkraft = map(self.mult, self.schwerkraft, zeitanteilvek)

    # Die einzige Kraft auf dem Kanonenball ist Schwerkraft:
    teil_beschleunigung = teil_schwerkraft

    # Beschleunigung auf Geschwindigkeit anwenden:
    # = [x geschwindigkeit, y geschwindigkeit - 9.81 * zeitanteil]
    self.geschwindigkeit = map(self.add, self.geschwindigkeit, teil_beschleunigung)

    # map( x * y, [x geschwindigkeit, y geschwindigkeit - 9.81 * zeitanteil], [zeitanteil, zeitanteil] )
    # = [ x geschwindigkeit * zeitanteil, (y geschwindigkeit - 9.81 * zeitanteil) * zeitanteil ]
    teil_geschwindigkeit = map(self.mult, self.geschwindigkeit, zeitanteilvek)

    # Geschwindigkeit auf Ort anwenden: {aktuellster Ort}
    # = [ ort_x + x geschwindigkeit * zeitanteil, ort_y + (y geschwindigkeit - 9.81 * zeitanteil) * zeitanteil]
    self.ort = map(self.add, self.ort, teil_geschwindigkeit)

    # Kanonenbaelle sollen nicht zum Boden gehen:
    if self.ort[1] < 0:
      self.ort[1] = 0


## Programm Start

# Physikalische Gleichungen eines Kanonenschusses:
# Geschwindigkeit: sin(45)*100 = 70.710 ; cos(45)*100 = 70.710

# Startgeschwindigkeit = Geschwindigkeit + Schwerkraft * Zeit
# 0 = 70.710 + (-9.81)t

# Zeit: t = 70.710/9.81 = 7.208 Sekunden fuer die Haelfte

# 14.416 Sekunden fuer die ganze Reise
# 0.1 s/Iteration * 145 Iterationen = 14.5

# Abstand: 70.710 m/s * 14.416 s = 1019.36796 m

zeitanteil = 0.1        # Sekunden pro Iteration
iterationen = 145       # Anzahl der Iterationen fuer die Simulation
rauschpegel = 30        # Rausch fuer rauschenthaltenden Messungen
mu_gkt = 100            # Wie schnell der Kanonenball herauskommt
winkel = 45             # Winkel vom Boden

# Arrays, um Datenpunkten, die wir am Ende plotten wollen, zu speichern
x = []
y = []
nx = []
ny = []
kx = []
ky = []
gx = []
gy = []

## Kanonesimulation
kan = kanone(zeitanteil, rauschpegel)

gkt_x = mu_gkt * math.cos(winkel * math.pi/180)
gkt_y = mu_gkt * math.sin(winkel * math.pi/180)

# Zustandswandelmatrix M
# 1, ts, 0,  0  =>  x(n+1) = x(n) + vx(n)
# 0,  1, 0,  0  => vx(n+1) =        vx(n)
# 0,  0, 1, ts  =>  y(n+1) =              y(n) + vy(n)
# 0,  0, 0,  1  => vy(n+1) =                     vy(n)
# Beschleunigung wird dazu addiert bei dem Kontrollvektor
M = numpy.matrix([[1,zeitanteil,0,0], [0,1,0,0], [0,0,1,zeitanteil], [0,0,0,1]])

B = numpy.matrix([[0,0,0,0], [0,0,0,0], [0,0,1,0], [0,0,0,1]])
# 0            =>  x(n+1) =  x(n+1)
# 0            => vx(n+1) = vx(n+1)
# -9.81 * ts^2 =>  y(n+1) =  y(n+1) + 0.5 * -9.81 * ts^2
# -9.81 * ts   => vy(n+1) = vy(n+1) + -9.81 * ts
kontroll_vektor = numpy.matrix([[0], [0], [0.5 * -9.81 * zeitanteil * zeitanteil], [-9.81 * zeitanteil]])

# Gleichungen fuer die Bewegung einer Parabel
#  x(n+1) = x(n) + vx(n)
# vx(n+1) = vx(n)
#  y(n+1) = y(n) + vy(n) - 0.5 * 9.81 * ts^2
# vy(n+1) = vy(n) + -9.81 * ts

# Beobachtungsmatrix H ist die Identitaetsmatrix, da wir alle Messungenwerte direkt bekommen koennen
H = numpy.eye(4)

# Schaetzung fuer den Anfangszustand
# auch wenn der y Wert (absichtlich) falsch gewaehlt ist, wird der Kalman Filter richtig simuliert
m = numpy.matrix([[0], [gkt_x], [0], [gkt_y]])

# Anfangsschaetzung C fuer Fehler in m
C = numpy.eye(4)

# geschaetzte Messfehler sigma im Verfahren (0, weil wir Gleichungen direkt von Simulation entwickelt haben)
sigma = numpy.zeros(4)

# geschaetzte Messfehler in Daten (variabel)
gamma = numpy.eye(4) * 0.01

kf = lineare_kalman_filter(M, B, H, m, C, sigma, gamma)

# Iteration fuer die Simulation
for i in range(iterationen):
    # wahre Laufbahn
    x.append(kan.ort_x())
    y.append(kan.ort_y())

    # gemessene, rauschenthaltende Werte
    neueste_x = kan.ort_x_mit_rausch()
    neueste_y = kan.ort_y_mit_rausch()
    nx.append(neueste_x)
    ny.append(neueste_y)

    # Iteriere die Kanonesimulation in den naechsten Zeitanteil
    kan.kanone_schritt()

    # Kalman Filter
    kx.append(kf.aktueller_zustand()[0,0])
    ky.append(kf.aktueller_zustand()[2,0])
    kf.schritt(kontroll_vektor, numpy.matrix([[neueste_x], [kan.x_geschwindigkeit()], [neueste_y], [kan.y_geschwindigkeit()]]))

# Listen zur Kontrolle
# print 'Vorhersage m_hut: \n', kf.filter_m_hut, '\n'
# print 'Vorhersage C_hut: \n', kf.filter_C_hut, '\n'
# print 'Analyse m: \n', kf.filter_m, '\n'
# print 'Analyse C: \n', kf.filter_C, '\n'

# Liste fuer geglaetteter Durchschnittswert
m = []
m.append(kf.filter_m[-1])

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
  gx.append(ks.geglaettet()[0,0])        # geglaetteter Durchschnittswert
  gy.append(ks.geglaettet()[2,0])

# Listen zur Kontrolle
# print 'smooth m: \n', m , '\n'
# print 'smooth C: \n', C, '\n'
print 'filter x-wert: \n', kx
print '\n filter y-wert: \n', ky
print '\n reversed x-glaetter: \n', gx[::-1]
print '\n reversed y-glaetter: \n', gy[::-1]


# Alle Ergebnisse plotten
import seaborn as sns
sns.set()

pylab.plot(nx,ny,':',x,y,'-',kx,ky,'--',gx[::-1],gy[::-1],'--')
pylab.xlabel('X Position')
pylab.ylabel('Y Position')
pylab.title('Messung eines Kanonenballs im Flug')
pylab.legend(('gemessen','wahr','Filter','Glaetter'))
pylab.show()