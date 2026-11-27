"""Background estimation for measured gamma spectra (SNIP peak clipping)."""

import numpy as np


def snipBackground(counts, maxHalfWindow, useLogLogSqrt=False):
    """Estimate the smooth background under a spectrum with the SNIP algorithm.

    SNIP (Sensitive Nonlinear Iterative Peak clipping; Ryan et al., NIM B 34 (1988) 396,
    as implemented by Morhac in ROOT's TSpectrum::Background). For half-windows
    p = 1, 2, ..., maxHalfWindow, every channel is replaced by the smaller of its own
    value and the mean of the channels p to the left and p to the right:

        v[i] <- min(v[i], (v[i - p] + v[i + p]) / 2)

    A peak sticks up above the straight line between its neighbours and is clipped
    away; a smooth (convex or linear) background is left unchanged. With
    useLogLogSqrt=False this reproduces TSpectrum::Background(..., numberIterations=
    maxHalfWindow, kBackIncreasingWindow, kBackOrder2, smoothing=false, compton=false).

    params:
        counts: 1D array of counts per channel (bin)
        maxHalfWindow: largest half-window in channels, either one int for the whole
            spectrum or an int array with one value per channel. Choose it around the
            width of the peaks at their base (roughly 1-2 FWHM); smaller values leave
            peak remnants, larger values cut into curved background. Since the LaBr3
            FWHM grows with energy, an array proportional to FWHM(E) / bin width
            works better than a single value. A channel with window w is only
            clipped in the passes p <= w.
        useLogLogSqrt: apply the LLS transform v = log(log(sqrt(y + 1) + 1) + 1)
            before clipping (Morhac & Matousek, Appl. Spectrosc. 62 (2008) 91). It
            compresses the large dynamic range of gamma spectra, so small peaks on a
            high background are clipped as well as large ones. TSpectrum does not
            do this, so leave it off when comparing with ROOT.

    returns:
        background: array with the same shape as counts. The first and last
            maxHalfWindow channels are not clipped (as in TSpectrum).

    Limitations:
        - Taking minima picks out downward Poisson fluctuations, so on low-count
          spectra the background comes out too low. Smooth the counts first
          (e.g. scipy.ndimage.gaussian_filter1d with sigma of a few channels).
        - Only the second-order clipping filter (TSpectrum's kBackOrder2).
        - The per-channel window is an extension; TSpectrum only has a single window.

    LLM usage: Level 2 (Snippet); an LLM wrote this function.
    """
    values = np.asarray(counts, dtype=float).copy()
    if useLogLogSqrt:
        values = np.log(np.log(np.sqrt(values + 1) + 1) + 1)

    numberOfChannels = values.size
    # One maximum half-window per channel (a single int gives the same window everywhere)
    halfWindowPerChannel = np.broadcast_to(np.asarray(maxHalfWindow, dtype=int), values.shape)
    for halfWindow in range(1, int(halfWindowPerChannel.max()) + 1):
        if 2 * halfWindow >= numberOfChannels:
            break
        inner = slice(halfWindow, numberOfChannels - halfWindow)
        neighbourMean = 0.5 * (values[:-2 * halfWindow] + values[2 * halfWindow:])
        clipped = np.minimum(values[inner], neighbourMean)
        # Channels whose own window is smaller than this pass keep their value
        stillClipping = halfWindow <= halfWindowPerChannel[inner]
        # Compute the whole pass from the previous values before overwriting, as TSpectrum does
        values[inner] = np.where(stillClipping, clipped, values[inner])

    if useLogLogSqrt:
        values = (np.exp(np.exp(values) - 1) - 1) ** 2 - 1
    return values
