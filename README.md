<pre>
Transient detection and extraction using matching pursuit in the wavelet domain.
    
This class extracts transient information of an audio signal via a reconstructed
"support signal," which attempts to isolate transient-only information from the
continuous wavelet transform. By finding wavelet-domain peaks, and tracing them
through scale values within their respective cones-of-influence, the algorithm
attempts to reconstruct a signal whose detail band is identical to a transient-only
signal (the support signal). This is then converted to the time domain via a
inverse nondecimated wavelet transform. Details of the algorithm can be seen in the
reference:

V. Bruni, S. Marconi, and D. Vitulano, “Time-scale atoms chains for transients detection 
    in audio signals,” IEEE Trans. Audio Speech Lang. Process., vol. 18, no. 3, pp. 420–433,
    Mar. 2010, doi: 10.1109/TASL.2009.2032623.
</pre>
