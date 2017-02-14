<h3>Linear Kalman Filter and Smoother in Python 2.7</h3>
<p>
The Kalman filter and smoother are algorithms that use Bayesian statistics
to produce accurate estimates of a system despite having noisy, inaccurate measurements.<br>
More information can be found <a href="https://en.wikipedia.org/wiki/Kalman_filter">here</a>.
</p>

<p>
The filter and smoother are stored in separate files for better organisation.<br>
The following examples will be used to demonstrate and compare the algorithms:
<ul>
<li><b>Free fall of an object:</b> The physics equations were taken from <a href="http://biorobotics.ri.cmu.edu/papers/sbp_papers/integrated3/kleeman_kalman_basics.pdf">this presentation</a>.</li>
<li><b>Firing a cannon ball:</b> This example was written by <a href="http://greg.czerniak.info/guides/kalman1/">Greg Czerniak</a>, and I added the smoothing algorithm.</li>
</ul>
<br>
Note: The code is written in German.
</p>
