%% EXTRACT BEHAVIORAL EVENTS
% Walks the session .rhd files and pulls out
% the task's ground-truth event channels:
%   ANALOG-IN-2   -> lever trace (continuous)
%   DIGITAL-IN-04 -> cue onset
%   DIGITAL-IN-03 -> reward dispense
%
% Timestamps are placed on the SAME running clock (time_offset accumulated
% by n_samples/fs per file)
% so cue/reward/lever times are directly comparable to spike/bin times with
% no extra alignment step.

clear; clc;

%% ---- Config ----
data_folder       = fullfile(dataRoot(), 'MouseM1', 'Day2', 'eOPN_M1_DualShank_Day2_260722_163423');
analog_in_num     = 2;   % ANALOG-IN-2   = lever trace
cue_dig_in_num    = 4;   % DIGITAL-IN-04 = cue
reward_dig_in_num = 3;   % DIGITAL-IN-03 = reward

%% ---- Get files in chronological order (same convention as phase1) ----
d = dir(fullfile(data_folder, '*.rhd'));
assert(~isempty(d), 'No .rhd files found in %s', data_folder);
[~, idx] = sort({d.name});
d = d(idx);
fprintf('Found %d .rhd files. First: %s | Last: %s\n', numel(d), d(1).name, d(end).name);

%% ---- Walk files, extract events ----
cueTimes     = [];
rewardTimes  = [];
leverTrace   = [];
leverTimeVec = [];
fs = [];
time_offset = 0;
analog_row = []; cue_row = []; reward_row = [];

for f = 1:numel(d)
    fpath = fullfile(d(f).folder, d(f).name);
    fprintf('[%d/%d] Processing %s ...\n', f, numel(d), d(f).name);

    [~, ~, fp, board_dig_in_data, board_dig_in_channels, ~, board_adc_data, board_adc_channels] = ...
        read_rhd_programmatic(fpath);

    if isempty(fs)
        fs = fp.amplifier_sample_rate;
    end

    % Resolve channel rows once, from the first file, by the trailing
    % number in native_channel_name -- not by an exact zero-padding
    % assumption, since Intan's own convention can vary between
    % "ANALOG-IN-2" and "ANALOG-IN-02" depending on software version.
    if f == 1
        analog_row = find_channel_row(board_adc_channels, analog_in_num);
        cue_row    = find_channel_row(board_dig_in_channels, cue_dig_in_num);
        reward_row = find_channel_row(board_dig_in_channels, reward_dig_in_num);
        fprintf('Resolved rows -> lever (ADC): %d | cue (DIG): %d | reward (DIG): %d\n', ...
            analog_row, cue_row, reward_row);
    end

    n_samples_f = size(board_dig_in_data, 2);
    t_local  = (0:n_samples_f-1) / fs;
    t_global = t_local + time_offset;

    % --- Digital events: rising-edge detect ---
    cue_rises    = find(diff(board_dig_in_data(cue_row, :)) > 0) + 1;
    reward_rises = find(diff(board_dig_in_data(reward_row, :)) > 0) + 1;
    cueTimes    = [cueTimes,    t_global(cue_rises)];    %#ok<AGROW>
    rewardTimes = [rewardTimes, t_global(reward_rises)]; %#ok<AGROW>

    % --- Analog lever trace: keep the full trace + time vector; we don't
    % know the resting-vs-pulled voltage range yet, so no thresholding here.
    leverTrace   = [leverTrace,   board_adc_data(analog_row, :)]; %#ok<AGROW>
    leverTimeVec = [leverTimeVec, t_global]; %#ok<AGROW>

    time_offset = time_offset + n_samples_f / fs;
end

fprintf('\nDone. %d cue events, %d reward events, %.1f s of lever trace.\n', ...
    numel(cueTimes), numel(rewardTimes), leverTimeVec(end));

save(fullfile(data_folder, 'behavioral_events.mat'), ...
    'cueTimes', 'rewardTimes', 'leverTrace', 'leverTimeVec', 'fs', '-v7.3');
fprintf('Saved behavioral_events.mat\n');

%% ---- Sanity check: cue times vs known trial-start bin offsets ----
% BMI_Workspace.mat's `dat` struct trials were previously located at bins
% 778 -> 89828 of the 20 ms binned_data (matched byte-for-byte against
% dat(i).spikes in a prior session). Cross-check that cueTimes, converted
% to the same 20 ms bin grid, land in the same range with the same count
% (197 trials) -- don't assume agreement, report the actual discrepancy.
bin_width_s = 0.020;
cueBinIdx = round(cueTimes / bin_width_s);
fprintf('\nSanity check vs known trial offsets (bins 778 -> 89828, 197 trials):\n');
fprintf('  cue events found: %d (expected ~197)\n', numel(cueTimes));
if ~isempty(cueBinIdx)
    fprintf('  cue bin range: %d -> %d\n', min(cueBinIdx), max(cueBinIdx));
    fprintf('  first cue bin offset from 778: %d bins\n', cueBinIdx(1) - 778);
    fprintf('  last cue bin offset from 89828: %d bins\n', cueBinIdx(end) - 89828);
end

%% ======================================================================
function row = find_channel_row(channels, targetNum)
% FIND_CHANNEL_ROW  Locate the struct-array row whose native_channel_name
% ends in targetNum (handles both zero-padded and unpadded Intan naming,
% e.g. "ANALOG-IN-2" and "ANALOG-IN-02").
    names = {channels.native_channel_name};
    nums = nan(1, numel(names));
    for i = 1:numel(names)
        tok = regexp(names{i}, '-(\d+)$', 'tokens', 'once');
        if ~isempty(tok)
            nums(i) = str2double(tok{1});
        end
    end
    row = find(nums == targetNum, 1);
    assert(~isempty(row), 'No channel found matching number %d among: %s', ...
        targetNum, strjoin(names, ', '));
end
