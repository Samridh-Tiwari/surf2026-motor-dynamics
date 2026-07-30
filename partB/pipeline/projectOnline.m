function score = projectOnline(newBin, pcaModel)
%PROJECTONLINE Project one new bin (channels x 1) onto the frozen basis.
%
%   Uses the calibration mu/sigma and coeff from fitCalibrationPCA -- this
%   is the only PCA-related computation that runs during the closed loop
%   itself, and it's a single matrix multiply, so it's cheap enough to
%   call every bin.

    z = (newBin - pcaModel.mu) ./ pcaModel.sigma;
    score = pcaModel.coeff' * z; % nComponents x 1
end
