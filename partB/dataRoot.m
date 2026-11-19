function root = dataRoot()
%DATAROOT Folder that holds the Part B recordings (Data_Realtime).
%   Set it once per MATLAB session before running any Part B script:
%       setenv('SURF_DATA_REALTIME', '/path/to/Data_Realtime')
%   The folder is expected to contain MouseM1/Day1, MouseM1/Day2, ...
root = getenv('SURF_DATA_REALTIME');
if isempty(root)
    error('dataRoot:unset', ...
        'Set SURF_DATA_REALTIME to the Data_Realtime folder (see paths.env).');
end
end
