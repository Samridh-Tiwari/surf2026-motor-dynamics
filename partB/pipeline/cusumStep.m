function [S, alarm] = cusumStep(x, S_prev, mu0, k, h)
%CUSUMSTEP One-sided CUSUM update for detecting a sustained upward ramp.
%
%   A flat threshold on "distance from baseline" fires late on a slow
%   ramp (it waits for one bin to cross the line) and can also false-fire
%   on a single noisy bin. CUSUM instead accumulates deviation above
%   baseline over time and only alarms once enough SUSTAINED evidence has
%   built up -- closer to how a ramp actually looks in the data.
%
%   x      : current statistic, e.g. norm(projectOnline(...) - calibScoreMean)
%   S_prev : accumulator from the previous step (initialize at 0)
%   mu0    : expected baseline value of x (0 if x is already a deviation,
%            otherwise use the calibration mean of x)
%   k      : slack/allowance -- deviations smaller than k don't accumulate.
%            Start around 0.5 * (baseline std of x).
%   h      : decision threshold -- alarm fires once S crosses this.
%            Start around 4-5 * (baseline std of x), then tune against
%            false-alarm rate on held-out calibration data.
%
%   Returns the updated accumulator S (feed back in as S_prev next call)
%   and a logical alarm flag. After an alarm, reset S to 0 externally if
%   you want the detector to look for the next ramp rather than staying
%   latched.

    S = max(0, S_prev + (x - mu0 - k));
    alarm = S > h;
end
