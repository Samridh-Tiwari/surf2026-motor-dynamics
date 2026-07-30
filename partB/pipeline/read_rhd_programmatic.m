function [amplifier_data, t_amplifier, frequency_parameters, board_dig_in_data, board_dig_in_channels, amplifier_channels, board_adc_data, board_adc_channels] = read_rhd_programmatic(full_filepath)
% READ_RHD_PROGRAMMATIC  Thin wrapper so you can call the stock Intan
% read_Intan_RHD2000_file.m in a loop without the uigetfile dialog
% popping up for every single file.
%
% Requires: read_Intan_RHD2000_file.m on your MATLAB path (unmodified is fine).
%
% Trick: that script calls uigetfile internally to get 'path' and
% 'filename'. We can't easily strip that out without editing Intan's
% file line-by-line (fragile across versions), so instead we do the
% next best thing: cd into the folder and monkey-patch the two variables
% it expects by pre-seeding them in the base workspace, then eval the
% script body. This works for the common (2020+) versions of the file.
%
% If this errors on your version, the bulletproof fix is: open
% read_Intan_RHD2000_file.m, find the block with
%   [filename, path, filterindex] = uigetfile(...)
% and comment those lines out, replacing with the two lines commented
% below (search for "PATCH HERE" style edit instructions in the
% companion note at the bottom of this file).

[folder, fname, ext] = fileparts(full_filepath);
fname = [fname, ext];

% Seed the variables read_Intan_RHD2000_file.m expects to find already
% defined (once you've added the if-check patch), so it skips its
% uigetfile dialog. These must be LOCAL variables here, not
% base-workspace, because when we call the script by its bare name
% below, it executes in THIS function's workspace (that's how MATLAB
% scripts-called-from-functions behave).
%
% NOTE: this script's actual variable name is `file`, not `filename` --
% confirmed from the source: [file, path, filterindex] = uigetfile(...)
path = [folder, filesep];    %#ok<NASGU>
file = fname;                %#ok<NASGU>

% Make sure read_Intan_RHD2000_file.m stays on the path even after we cd
% into the data folder below (cd does NOT add to the search path).
this_file_dir = fileparts(mfilename('fullpath'));
addpath(this_file_dir);

old_dir = pwd;
cd(folder);
try
    read_Intan_RHD2000_file(file, path);   % runs as a script, dumps vars to base workspace
catch ME
    cd(old_dir);
    rethrow(ME);
end
cd(old_dir);

% Pull results back out of the base workspace into this function's output
amplifier_data       = evalin('base', 'amplifier_data');
t_amplifier           = evalin('base', 't_amplifier');
frequency_parameters   = evalin('base', 'frequency_parameters');
board_dig_in_data      = evalin('base', 'board_dig_in_data');
board_dig_in_channels  = evalin('base', 'board_dig_in_channels');
amplifier_channels     = evalin('base', 'amplifier_channels');

% board_adc_data/board_adc_channels only land in the base workspace if the
% file has num_board_adc_channels > 0 (see move_to_base_workspace calls in
% read_Intan_RHD2000_file.m) -- guard rather than assume every file has them.
if evalin('base', 'exist(''board_adc_data'', ''var'')')
    board_adc_data     = evalin('base', 'board_adc_data');
    board_adc_channels = evalin('base', 'board_adc_channels');
else
    board_adc_data     = [];
    board_adc_channels = [];
end
end

% ---------------------------------------------------------------------
% MANUAL PATCH (do this once, takes 30 seconds, and is the reliable fix
% if the wrapper above throws an error about uigetfile popping up):
%
% 1. Open read_Intan_RHD2000_file.m in the editor
% 2. Find this block near the top (approx line 40-60):
%
%       [filename, path, filterindex] = ...
%           uigetfile('*.rhd', 'Select an RHD2000 Data File', 'MultiSelect', 'off');
%
% 3. Replace with:
%
%       if ~exist('filename', 'var') || ~exist('path', 'var')
%           [filename, path, filterindex] = ...
%               uigetfile('*.rhd', 'Select an RHD2000 Data File', 'MultiSelect', 'off');
%       end
%
% 4. Save it as read_Intan_RHD2000_file.m (overwrite). Now it behaves
%    exactly as before when run interactively, but if 'filename' and
%    'path' already exist in the base workspace (as this wrapper sets
%    them), it skips the dialog entirely.
% ---------------------------------------------------------------------