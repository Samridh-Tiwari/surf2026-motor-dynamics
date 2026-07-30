function [spikeFlags, state] = detectSpikesCausal(xFilt, state, lockoutSamples)
%DETECTSPIKESCAUSAL Threshold crossing with 1 ms-style lockout, block-safe.
%
%   Same logic as Phase 1's findpeaks-with-lockout, but implemented as an
%   explicit refractory counter per channel so it can be resumed exactly
%   where it left off at the start of the next block (findpeaks alone has
%   no notion of "state carried from the previous call").
%
%   state.threshold comes from updateAdaptiveBaseline -- pass the SAME
%   state struct through both functions each iteration.

    [n, numChannels] = size(xFilt);
    spikeFlags = false(n, numChannels);

    if ~isfield(state, 'refractoryCounter')
        state.refractoryCounter = zeros(1, numChannels);
    end

    % Vectorized per-channel scan: instead of visiting every sample (the
    % old n x numChannels double loop, which ran slower than real time),
    % jump directly from one threshold crossing to the end of its lockout.
    % Spikes are sparse relative to the 20 kHz sample rate, so this loops
    % only a handful of times per channel per block. Semantics are
    % identical: detect the FIRST sample that goes below threshold, then
    % suppress the next lockoutSamples samples, with the refractory count
    % carried across block boundaries.
    below = xFilt < state.threshold;   % n x numChannels logical (thr is 1 x numChannels)

    for ch = 1:numChannels
        if isinf(state.threshold(ch))
            state.refractoryCounter(ch) = 0;
            continue; % no baseline estimate yet -- don't detect on a bogus threshold
        end

        rc = state.refractoryCounter(ch);
        belowCh = below(:, ch);

        % Consume any refractory carried in from the previous block.
        pos = 1;
        if rc > 0
            skip = min(rc, n);
            pos = pos + skip;
            rc = rc - skip;   % >0 only if the whole block is still locked out
        end

        while pos <= n
            rel = find(belowCh(pos:n), 1);   % next crossing at or after pos
            if isempty(rel)
                rc = 0;
                break;
            end
            spikeIdx = pos + rel - 1;
            spikeFlags(spikeIdx, ch) = true;
            lastLocked = spikeIdx + lockoutSamples;   % samples spikeIdx+1..spikeIdx+lockout are suppressed
            pos = lastLocked + 1;
            rc = max(0, lastLocked - n);   % remaining lockout to carry into next block
        end

        state.refractoryCounter(ch) = rc;
    end
end
