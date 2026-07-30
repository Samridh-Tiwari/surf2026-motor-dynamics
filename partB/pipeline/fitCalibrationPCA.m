function pcaModel = fitCalibrationPCA(calibBinned, nComponents)
%FITCALIBRATIONPCA Fit PCA ONCE on a calibration block, then freeze it.
%
%   calibBinned : channels x bins, smoothed spike counts from a known
%                 calibration period (e.g. an initial rest/hold block,
%                 or the pre-cue baseline of early trials).
%   nComponents : how many PCs to keep (start with Phase 3's scree plot;
%                 for trial-averaged, condition-free calibration data you
%                 should land closer to the paper's 10-15 than the 31
%                 you saw on continuous, un-averaged data).
%
%   The basis (pcaModel.coeff) is NOT refit online -- see projectOnline.m.
%   If you let the axes keep moving as new data comes in, "distance from
%   baseline" is measured against a target that's also drifting, which
%   defeats the point of a fixed detector. Recalibrate deliberately
%   (e.g. once per session) rather than continuously.

    mu    = mean(calibBinned, 2);
    sigma = std(calibBinned, 0, 2);
    sigma(sigma == 0) = 1; % guard dead/flat channels

    z = (calibBinned - mu) ./ sigma;
    [coeff, score] = pca(z');

    pcaModel.mu    = mu;
    pcaModel.sigma = sigma;
    pcaModel.coeff = coeff(:, 1:nComponents);

    % Baseline statistics of the projected score itself, used downstream
    % to set CUSUM's mu0/k/h in the same coordinate system.
    pcaModel.calibScoreMean = mean(score(:, 1:nComponents), 1)';
    pcaModel.calibScoreStd  = std(score(:, 1:nComponents), 0, 1)';
end
