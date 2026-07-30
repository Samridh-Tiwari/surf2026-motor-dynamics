function state = initAdaptiveBaselineState(fs, windowSec, refreshSec, numChannels, kMAD)
%INITADAPTIVEBASELINESTATE Trailing-window MAD threshold, causal only.
%
%   Your original threshold (-4 x MAD-sigma) was computed once from an
%   entire 60 s file -- fine offline, but a global-file statistic is
%   technically a peek at the future for samples early in that file.
%
%   This version keeps a rolling buffer of the last `windowSec` seconds
%   of FILTERED data and recomputes the threshold every `refreshSec`,
%   using only samples that have already occurred. That's the trade you're
%   making deliberately: slow drift gets tracked, but the threshold is a
%   few seconds "behind" a genuinely instantaneous noise-floor change.
%
%   No detections are allowed until the buffer has filled once
%   (state.threshold starts at -Inf), so the detector doesn't fire on a
%   threshold estimated from a half-empty window at startup.

    state.windowSamples   = round(windowSec * fs);
    state.refreshSamples  = round(refreshSec * fs);
    state.samplesSinceRefresh = state.refreshSamples; % force an estimate ASAP
    state.buffer       = zeros(state.windowSamples, numChannels);
    state.writePtr     = 1;   % ring-buffer write head (see updateAdaptiveBaseline)
    state.bufferFilled = 0;
    state.threshold    = -inf(1, numChannels);
    state.kMAD = kMAD;
end
