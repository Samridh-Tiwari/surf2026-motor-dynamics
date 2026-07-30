function state = updateAdaptiveBaseline(xFilt, state)
%UPDATEADAPTIVEBASELINE Push new filtered samples into the trailing buffer
%   and recompute the MAD threshold every refreshSamples worth of data.
%
%   xFilt: samples x channels, causally filtered block.

    n = size(xFilt, 1);
    W = state.windowSamples;

    % Ring buffer: overwrite the oldest samples in place rather than
    % rebuilding the whole [buffer(n+1:end,:); xFilt] array every block
    % (that copied the full ~5 s window per 20 ms block and was the
    % dominant real-time cost once filtering/detection were vectorized).
    % Only the n new rows are written. The threshold statistic below is a
    % median/MAD, which is permutation-invariant, so the ring's internal
    % ordering is irrelevant -- the result is identical to the old shift.
    if n >= W
        % Block bigger than the whole window (only with huge blocks / tiny
        % windows) -- just keep the tail, reset the write head.
        state.buffer = xFilt(end - W + 1:end, :);
        state.writePtr = 1;
    else
        p = state.writePtr;
        if p + n - 1 <= W
            state.buffer(p:p + n - 1, :) = xFilt;
        else
            firstPart = W - p + 1;
            state.buffer(p:W, :) = xFilt(1:firstPart, :);
            state.buffer(1:n - firstPart, :) = xFilt(firstPart + 1:end, :);
        end
        state.writePtr = mod(p - 1 + n, W) + 1;
    end
    state.bufferFilled = min(state.bufferFilled + n, W);
    state.samplesSinceRefresh = state.samplesSinceRefresh + n;

    readyToRefresh = state.samplesSinceRefresh >= state.refreshSamples;
    windowFull      = state.bufferFilled >= W;

    if readyToRefresh && windowFull
        validData = state.buffer;
        medAbs = median(abs(validData - median(validData, 1)), 1);
        sigma  = medAbs / 0.6745;
        state.threshold = -state.kMAD * sigma;
        state.samplesSinceRefresh = 0;
    end
end
