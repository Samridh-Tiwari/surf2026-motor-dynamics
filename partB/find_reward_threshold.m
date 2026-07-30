function find_reward_threshold
%FIND_REWARD_THRESHOLD  The reward is dispensed by hardware when the lever
%   crosses a FIXED voltage, so that voltage must be the same on both days.
%   Recover it empirically: look at the lever trace right before each reward.
%   The lower edge of that distribution IS the hardware threshold.
%   Then check whether one common absolute threshold works for both sessions.
    root='C:\Users\tiwar133\gpfa_v0203\Data_Realtime';
    S={'Day1',fullfile(root,'MouseM1\Day1\eOPN_M1_DualShank_Day1_260721_195215');
       'Day2',fullfile(root,'MouseM1\Day2\eOPN_M1_DualShank_Day2_260722_163423')};
    pk=cell(1,2);
    for s=1:2
        ev=load(fullfile(S{s,2},'behavioral_events.mat'));
        lv=double(ev.leverTrace(:)'); lt=double(ev.leverTimeVec(:)'); fs=ev.fs;
        rew=double(ev.rewardTimes(:)');
        fprintf('\n########## %s ##########\n',S{s,1});
        fprintf('lever trace (ABSOLUTE volts): min %.4f  p1 %.4f  median %.4f  p99 %.4f  max %.4f\n', ...
            min(lv),prctile(lv,1),median(lv),prctile(lv,99),max(lv));
        % peak lever voltage in the 1 s before each reward
        p=nan(1,numel(rew));
        for k=1:numel(rew)
            i1=round((rew(k)-1-lt(1))*fs)+1; i2=round((rew(k)-lt(1))*fs)+1;
            i1=max(1,i1); i2=min(numel(lv),i2);
            if i2>i1, p(k)=max(lv(i1:i2)); end
        end
        p=p(~isnan(p)); pk{s}=p;
        fprintf('peak lever voltage in 1 s BEFORE reward (n=%d):\n',numel(p));
        fprintf('   min %.4f | p5 %.4f | p25 %.4f | median %.4f | max %.4f\n', ...
            min(p),prctile(p,5),prctile(p,25),median(p),max(p));
        fprintf('   -> implied hardware threshold ~ %.4f V (5th pct of reward-triggering peaks)\n',prctile(p,5));
    end
    thr1=prctile(pk{1},5); thr2=prctile(pk{2},5);
    fprintf('\n================ COMPARISON ================\n');
    fprintf('Day1 implied threshold %.4f V | Day2 %.4f V | difference %.4f V\n',thr1,thr2,abs(thr1-thr2));
    common=min(thr1,thr2);
    fprintf('COMMON absolute threshold to use on BOTH days: %.4f V\n',common);

    % how many genuine pulls does the common absolute threshold give per day?
    for s=1:2
        ev=load(fullfile(S{s,2},'behavioral_events.mat'));
        lv=double(ev.leverTrace(:)'); lt=double(ev.leverTimeVec(:)'); fs=ev.fs;
        rew=double(ev.rewardTimes(:)'); cue=double(ev.cueTimes(:)');
        ab=lv>common; ri=find(diff(ab)>0)+1; on=[]; last=-inf;
        for r=ri, tc=lt(r); if tc-last<1, continue; end
            i=r; lim=r-round(0.6*fs);
            while i>1 && i>lim && lv(i)>common-0.02, i=i-1; end
            on(end+1)=lt(i); last=tc; end %#ok<AGROW>
        nRew=sum(arrayfun(@(o) any(rew>=o & rew<=o+2), on));
        nCued=sum(arrayfun(@(o) any(cue<=o & cue>=o-2), on));
        fprintf('%s: %d genuine pulls @ %.4f V (%.1f/min) | %d rewarded | %d cue-preceded | %d rewards total\n', ...
            S{s,1},numel(on),common,numel(on)/(lt(end)/60),nRew,nCued,numel(rew));
    end
end
