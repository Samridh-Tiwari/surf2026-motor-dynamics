function state = initCausalFilterState(numChannels, fs, cutoffHz, order)
%INITCAUSALFILTERSTATE Initialize a causal Butterworth high-pass filter.
%
%   Phase 1's filtfilt() is zero-phase, which requires the whole signal
%   (it runs forward then backward) -- it cannot run online because the
%   backward pass needs samples that haven't been acquired yet.
%
%   This uses a single forward pass via filter(), which introduces a
%   small, fixed group delay instead (a few samples at 300 Hz / 20 kHz),
%   but only ever needs data up to the current sample.
%
%   state.zi carries filter memory across successive blocks so that
%   calling applyCausalFilter() on back-to-back chunks is equivalent to
%   filtering the concatenated signal in one shot -- no block-boundary
%   artifacts.

    [b, a] = butter(order, cutoffHz / (fs / 2), 'high');
    state.b = b;
    state.a = a;
    state.fs = fs;
    state.zi = zeros(max(length(a), length(b)) - 1, numChannels);
end
