%% CLOSEDLOOP_REPLAY_REAL
% Real-data version of closedloop_demo.m: chains the same causal stages
% (commonMedianRef / applyCausalFilter, initAdaptiveBaselineState /
% updateAdaptiveBaseline, detectSpikesCausal, fitCalibrationPCA /
% projectOnline, cusumStep) over the actual 38-file session instead of
% synthetic data, with the PCA calibration window keyed to real pre-cue
% baseline periods (from behavioral_events.mat) instead of an arbitrary
% fixed duration.
%
% IMPORTANT: everything in this loop only looks backward in time, same
% invariant as closedloop_demo.m -- keep it that way if you modify it.
%
% Prerequisite: run extractBehavioralEvents.m first (needs cueTimes on the
% same global clock as this script produces).

clear; clc; close all;

%% ---- Config (mirrors closedloop_demo.m) ----
data_folder    = fullfile(dataRoot(), 'MouseM1', 'Day2', 'eOPN_M1_DualShank_Day2_260722_163423');
electrode_port = 'Port B';                 % M1 single shank (MouseM1 Day2)
n_channels_expected     = 64;              % Port B: B-000..B-063
raw_n_channels_expected = 160;             % Port A (64) + Port B (64) + Port C grid (32)

fs             = 20000;   % Hz
numChannels    = 64;      % Port B (M1 single shank), Day2
binSizeSec     = 0.02;    % 20 ms, matches Phase 2
blockSizeSec   = 0.02;    % process in 20 ms chunks -- must stay small relative
                           % to baselineRefreshSec, see note below
cutoffHz       = 300;
filtOrder      = 3;
lockoutMs      = 1;
baselineWindowSec  = 5;   % trailing window for adaptive threshold
baselineRefreshSec = 1;   % recompute threshold this often
kMAD           = 4;
nPCs           = 10;      % revisit with a scree plot once you've looked at
                           % the real calibration bins' variance structure

K = 8;                    % number of early trials to draw calibration bins from
preCueWindowSec = 1.0;    % window before each cue counted as "baseline"

lockoutSamples = round(lockoutMs / 1000 * fs);
binSamples     = round(binSizeSec * fs);
blockSamples   = round(blockSizeSec * fs);
assert(blockSamples == binSamples, ...
    'blockSizeSec and binSizeSec must match for this script''s bin-accumulation logic.');

%% ---- Load behavioral ground truth (must run extractBehavioralEvents.m first) ----
events_path = fullfile(data_folder, 'behavioral_events.mat');
assert(isfile(events_path), ...
    'behavioral_events.mat not found -- run extractBehavioralEvents.m first.');
ev = load(events_path, 'cueTimes', 'rewardTimes');
cueTimes    = ev.cueTimes;
rewardTimes = ev.rewardTimes;
assert(numel(cueTimes) > K, 'Need more than %d cue events for calibration, found %d.', K, numel(cueTimes));
calibCutoffTime = cueTimes(K);
fprintf('Calibration will collect pre-cue bins from the first %d trials, freezing PCA once t >= %.2f s.\n', ...
    K, calibCutoffTime);

%% ---- Get files in chronological order (same convention as phase1) ----
d = dir(fullfile(data_folder, '*.rhd'));
assert(~isempty(d), 'No .rhd files found in %s', data_folder);
[~, idx] = sort({d.name});
d = d(idx);
fprintf('Found %d .rhd files. First: %s | Last: %s\n', numel(d), d(1).name, d(end).name);

%% ---- Init causal-stage state ----
filtState = initCausalFilterState(numChannels, fs, cutoffHz, filtOrder);
baseState = initAdaptiveBaselineState(fs, baselineWindowSec, baselineRefreshSec, numChannels, kMAD);

currentBinCounts = zeros(numChannels, 1);
samplesIntoBin   = 0;

calibBins = [];   % channels x bins, accumulated during pre-cue windows
pcaModel  = [];   % set once the K-th cue's pre-cue window has passed

cusumS   = 0;
cusumMu0 = 0;
cusumK   = 0;
cusumH   = 0;

binIdx = 0;
alarmTimes  = [];
scoreTrace  = [];
scoreTimes  = [];

% Store the online-detected population count per bin for the whole session,
% so the (cheap) PCA + CUSUM tuning can be re-run offline from the saved
% matrix without repeating the (expensive) causal spike detection. See
% trajectory_cusum_offline.m.
maxBinsGuess = ceil(5000 / binSizeSec);      % > full-session bin count
online_binned  = zeros(numChannels, maxBinsGuess);
binCenterTimes = zeros(1, maxBinsGuess);

electrode_rows = [];

%% ---- Main causal loop over real files ----
for f = 1:numel(d)
    fpath = fullfile(d(f).folder, d(f).name);
    fprintf('[%d/%d] Processing %s ...\n', f, numel(d), d(f).name);

    [amp_data_raw, ~, fp, ~, ~, amp_channels] = read_rhd_programmatic(fpath);

    if f == 1
        assert(size(amp_data_raw, 1) == raw_n_channels_expected, ...
            'Expected %d raw amplifier channels, got %d.', raw_n_channels_expected, size(amp_data_raw, 1));
        assert(abs(fp.amplifier_sample_rate - fs) < 1e-6, ...
            'Config fs (%.1f) does not match file''s amplifier_sample_rate (%.1f).', fs, fp.amplifier_sample_rate);

        electrode_rows = find(strcmp({amp_channels.port_name}, electrode_port));
        assert(numel(electrode_rows) == n_channels_expected, ...
            'Expected %d channels on %s, found %d.', n_channels_expected, electrode_port, numel(electrode_rows));
        [~, chip_order] = sort([amp_channels(electrode_rows).chip_channel]);
        electrode_rows = electrode_rows(chip_order);
    end

    amp_data  = amp_data_raw(electrode_rows, :);   % [64 x n_samples_f]
    xRawFile  = amp_data';                          % samples x channels

    nSamplesFile = size(xRawFile, 1);

    for rawIdx = 1:blockSamples:nSamplesFile
        blockEnd = min(rawIdx + blockSamples - 1, nSamplesFile);
        xRaw = xRawFile(rawIdx:blockEnd, :);

        % --- Stage 1: causal preprocessing ---
        xRef = commonMedianRef(xRaw);
        [xFilt, filtState] = applyCausalFilter(xRef, filtState);

        % --- Stage 2: adaptive baseline + detection ---
        baseState = updateAdaptiveBaseline(xFilt, baseState);
        [spikeFlags, baseState] = detectSpikesCausal(xFilt, baseState, lockoutSamples);

        % --- Stage 3: accumulate into current 20 ms bin ---
        currentBinCounts = currentBinCounts + sum(spikeFlags, 1)';
        samplesIntoBin = samplesIntoBin + size(xRaw, 1);

        if samplesIntoBin >= binSamples
            binIdx = binIdx + 1;
            thisBin = currentBinCounts;
            currentBinCounts = zeros(numChannels, 1);
            samplesIntoBin = 0;

            tBinCenter = (binIdx - 0.5) * binSizeSec;
            online_binned(:, binIdx) = thisBin;
            binCenterTimes(binIdx)   = tBinCenter;

            if isempty(pcaModel)
                % --- Calibration-collection phase ---
                inPreCueWindow = any(tBinCenter >= (cueTimes(1:K) - preCueWindowSec) & tBinCenter < cueTimes(1:K));
                if inPreCueWindow
                    calibBins(:, end + 1) = thisBin; %#ok<SAGROW>
                end

                if tBinCenter >= calibCutoffTime
                    fprintf('Calibration done at t=%.2fs (%d pre-cue bins from %d trials). Freezing PCA basis.\n', ...
                        tBinCenter, size(calibBins, 2), K);
                    assert(size(calibBins, 2) >= nPCs, ...
                        'Only %d calibration bins collected, need at least nPCs=%d. Increase K or preCueWindowSec.', ...
                        size(calibBins, 2), nPCs);
                    pcaModel = fitCalibrationPCA(double(calibBins), nPCs);

                    cusumMu0 = 0;
                    distStd  = norm(pcaModel.calibScoreStd);
                    cusumK   = 0.5 * distStd;
                    cusumH   = 5   * distStd;
                    fprintf('CUSUM params: k=%.3f h=%.3f\n', cusumK, cusumH);
                end

            else
                % --- Stage 4: project + CUSUM (live phase) ---
                score = projectOnline(double(thisBin), pcaModel);
                dist  = norm(score - pcaModel.calibScoreMean);
                scoreTrace(end + 1) = dist;  %#ok<SAGROW>
                scoreTimes(end + 1) = tBinCenter; %#ok<SAGROW>

                [cusumS, alarm] = cusumStep(dist, cusumS, cusumMu0, cusumK, cusumH);
                if alarm
                    alarmTimes(end + 1) = tBinCenter; %#ok<SAGROW>
                    cusumS = 0; % reset so it can catch the next ramp too
                end
            end
        end
    end
end

%% ---- Report ----
fprintf('\nDone. %d bins processed after calibration. %d alarms fired.\n', ...
    numel(scoreTrace), numel(alarmTimes));

% Trim the preallocated online population matrix to the bins we actually
% filled, and save it separately so trajectory_cusum_offline.m can iterate
% on PCA/CUSUM without re-running causal detection.
online_binned  = online_binned(:, 1:binIdx);
binCenterTimes = binCenterTimes(1:binIdx);
binned_path = fullfile(data_folder, 'online_binned_data.mat');
save(binned_path, 'online_binned', 'binCenterTimes', 'binSizeSec', ...
    'calibCutoffTime', 'K', 'preCueWindowSec', '-v7.3');
fprintf('Saved online population matrix (%d x %d) to %s\n', ...
    size(online_binned,1), size(online_binned,2), binned_path);

results_path = fullfile(data_folder, 'closedloop_replay_real_results.mat');
save(results_path, 'alarmTimes', 'scoreTrace', 'scoreTimes', 'pcaModel', ...
    'cusumK', 'cusumH', 'calibCutoffTime', 'K', '-v7.3');
fprintf('Saved results to %s\n', results_path);

figure('Name', 'Real-data closed-loop trajectory monitoring');
plot(scoreTimes, scoreTrace, 'k'); hold on;
for i = 1:numel(alarmTimes)
    xline(alarmTimes(i), 'r--', 'LineWidth', 1.0);
end
liveCueTimes    = cueTimes(cueTimes >= calibCutoffTime);
liveRewardTimes = rewardTimes(rewardTimes >= calibCutoffTime);
for i = 1:numel(liveCueTimes)
    xline(liveCueTimes(i), 'b:', 'LineWidth', 0.5);
end
for i = 1:numel(liveRewardTimes)
    xline(liveRewardTimes(i), 'g:', 'LineWidth', 0.5);
end
xlabel('Time (s)'); ylabel('Distance from baseline (PCA space)');
title('Real-data closed-loop trajectory monitoring (post-calibration)');
legend({'distance', 'CUSUM alarm', 'cue', 'reward'}, 'Location', 'northwest');

%% ---- Lightweight validation: alarm hit rate & latency vs cue ----
% For every post-calibration trial, does an alarm fall inside
% [cue, cue + 2s] (the dat struct's known fixed trial window)?
trialWindowSec = 2.0;
hits = false(size(liveCueTimes));
latencies = nan(size(liveCueTimes));
for i = 1:numel(liveCueTimes)
    cand = alarmTimes(alarmTimes >= liveCueTimes(i) & alarmTimes < liveCueTimes(i) + trialWindowSec);
    if ~isempty(cand)
        hits(i) = true;
        latencies(i) = cand(1) - liveCueTimes(i);
    end
end
fprintf('\nValidation: %d/%d post-calibration trials had an alarm within %.1fs of cue (hit rate %.1f%%).\n', ...
    sum(hits), numel(liveCueTimes), trialWindowSec, 100 * mean(hits));
if any(hits)
    fprintf('Median latency (alarm - cue) on hits: %.3f s\n', median(latencies(hits)));
    figure('Name', 'Alarm latency relative to cue');
    histogram(latencies(hits), 20);
    xlabel('Alarm time - cue time (s)'); ylabel('Trial count');
    title('CUSUM alarm latency relative to cue onset');
end
