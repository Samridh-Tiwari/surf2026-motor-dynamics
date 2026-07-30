function [yFilt, state] = applyCausalFilter(xRaw, state)
%APPLYCAUSALFILTER Filter one block of raw samples, seamlessly across blocks.
%
%   xRaw  : samples x channels, raw (already common-median-referenced)
%   state : struct from initCausalFilterState, updated in place (state.zi)
%
%   Call this once per incoming block in the real-time loop, always
%   passing the SAME state struct back in so filter memory persists.

    % filter() processes each column independently along dim 1, carrying a
    % per-column initial/final condition in state.zi -- so the whole block
    % filters in one vectorized call, numerically identical to looping over
    % channels but far faster (the per-channel loop was a real-time
    % bottleneck).
    [yFilt, state.zi] = filter(state.b, state.a, xRaw, state.zi, 1);
end
