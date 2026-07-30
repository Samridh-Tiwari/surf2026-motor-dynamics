function [onset, thrV, amp] = genuine_pulls(sessionDir, minGapSec)
%GENUINE_PULLS  Lever-pull onsets using the ACTUAL hardware reward threshold.
%   The reward is dispensed when the lever crosses a fixed voltage, so that
%   voltage is recoverable from the data: the lower edge of the peak lever
%   voltage observed just before each reward. Measured at 2.8197 V (Day1) and
%   2.8188 V (Day2) -- a 0.9 mV difference, i.e. genuinely unchanging hardware.
%
%   This replaces MAD- or amplitude-percentile thresholds, which drift between
%   sessions and let micro-movements in (Day2 previously gave 2702 "onsets").
%
%   [onset, thrV, amp] = genuine_pulls(sessionDir)
    if nargin<2, minGapSec=1.0; end
    ev=load(fullfile(sessionDir,'behavioral_events.mat'));
    lv=double(ev.leverTrace(:)'); lt=double(ev.leverTimeVec(:)'); fs=ev.fs;
    rew=[]; if isfield(ev,'rewardTimes'), rew=double(ev.rewardTimes(:)'); end

    if ~isempty(rew)
        p=nan(1,numel(rew));
        for k=1:numel(rew)
            i1=max(1,round((rew(k)-1-lt(1))*fs)+1); i2=min(numel(lv),round((rew(k)-lt(1))*fs)+1);
            if i2>i1, p(k)=max(lv(i1:i2)); end
        end
        thrV=prctile(p(~isnan(p)),5);        % hardware threshold
    else
        thrV=median(lv)+0.33*(prctile(lv,99)-median(lv));
    end

    ab=lv>thrV; ri=find(diff(ab)>0)+1; onset=[]; amp=[]; last=-inf;
    backTo=thrV-0.02;                        % walk back to just above baseline
    for r=ri
        tc=lt(r); if tc-last<minGapSec, continue; end
        i=r; lim=r-round(0.6*fs);
        while i>1 && i>lim && lv(i)>backTo, i=i-1; end
        j=min(numel(lv),r+round(0.4*fs));
        onset(end+1)=lt(i); amp(end+1)=max(lv(r:j)); %#ok<AGROW>
        last=tc;
    end
end
