function verify_m1m2(day)
%VERIFY_M1M2  Robustness check on the "M2 helps decode MI vs MIFA" claim.
%   Re-runs the decode from the SAVED trial tensors (128ch x 41bins) with three
%   fixes the original lacked:
%     (1) DIMENSION-MATCHED comparisons  - M1(64) vs M2(64) vs M1+M2(128) plus
%         random-64 subsets, because 128 features beat 64 almost regardless of
%         content. M2-alone vs M1-alone is the only fair "does M2 carry more?" test.
%     (2) TIME-BLOCKED CV - contiguous folds, so temporally adjacent trials can't
%         leak between train and test (random KFold inflates accuracy).
%     (3) WINDOW SPLIT - pre-onset vs post-onset. MI trials have a cue shortly
%         before onset, so a post-onset window contains the cue-evoked auditory
%         response; if M2's edge lives only there it is sensory, not "intention".
%   Everything reported as BALANCED accuracy with a label-shuffle null.
    if nargin<1, day=2; end
    root='C:\Users\tiwar133\gpfa_v0203\Data_Realtime';
    if day==2
        f=fullfile(root,'MouseM1\Day2\eOPN_M1_DualShank_Day2_260722_163423\quick_M1_vs_M1M2_results.mat');
    else
        f=fullfile(root,'MouseM1\Day1\eOPN_M1_DualShank_Day1_260721_195215\quick_M1_vs_M1M2_results.mat');
    end
    S=load(f); rng(7);
    TMI=S.pcaM1M2.trialsMI; TFA=S.pcaM1M2.trialsMIFA;
    tMI=S.onsetMI(:); tFA=S.onsetMIFA(:);
    nCh=size(TMI{1},1); nB=size(TMI{1},2); onsetBin=round(nB/2);   % 41 bins -> onset at 21
    dt=0.02;
    fprintf('\n############ DAY %d  (%d MI, %d MIFA, %d ch, %d bins) ############\n',day,numel(TMI),numel(TFA),nCh,nB);

    wins={'PRE-onset  [-400,-20]ms', 1:onsetBin-1;
          'POST-onset [0,+300]ms',   onsetBin:min(nB,onsetBin+15)};
    sets={'M1 only            (64ch)', 1:64;
          'M2 only            (64ch)', 65:128;
          'M1+M2            (128ch)', 1:128};

    for w=1:size(wins,1)
        bins=wins{w,2};
        fprintf('\n===== window: %s =====\n',wins{w,1});
        fprintf('%-28s %-16s %-14s %s\n','feature set','balanced acc','shuffle null','above null');
        for s=1:size(sets,1)
            ch=sets{s,2};
            [X,y,t]=buildXY(TMI,TFA,tMI,tFA,ch,bins);
            a=cvBal(X,y,t);
            an=zeros(20,1); for i=1:20, an(i)=cvBal(X,y(randperm(numel(y))),t); end
            fprintf('%-28s %5.1f%%           %5.1f%% +/-%.1f    %s\n',sets{s,1},100*a,100*mean(an),100*std(an), ...
                ternary(a>mean(an)+3*std(an),'YES','no'));
        end
        % --- dimension-matched: 64 features drawn from M1, M2, or mixed ---
        nRep=15; aM1=zeros(nRep,1); aM2=zeros(nRep,1); aMix=zeros(nRep,1);
        for i=1:nRep
            p1=randperm(64,32); p2=64+randperm(64,32);
            [Xa,y,t]=buildXY(TMI,TFA,tMI,tFA,randperm(64,64),bins);        aM1(i)=cvBal(Xa,y,t);
            [Xb,~,~]=buildXY(TMI,TFA,tMI,tFA,64+randperm(64,64),bins);     aM2(i)=cvBal(Xb,y,t);
            [Xc,~,~]=buildXY(TMI,TFA,tMI,tFA,[p1 p2],bins);                aMix(i)=cvBal(Xc,y,t);
        end
        fprintf('  -- dimension-matched (all 64 features) --\n');
        fprintf('   64 M1 ch          : %5.1f%% +/-%.1f\n',100*mean(aM1),100*std(aM1));
        fprintf('   64 M2 ch          : %5.1f%% +/-%.1f\n',100*mean(aM2),100*std(aM2));
        fprintf('   32 M1 + 32 M2     : %5.1f%% +/-%.1f\n',100*mean(aMix),100*std(aMix));
    end
    fprintf('\n(If "M1+M2 (128)" beats M1(64) but M2-alone(64) ~= M1-alone(64) and the\n');
    fprintf(' 32+32 mix is no better than 64 M1, the gain was DIMENSIONALITY, not M2 content.)\n');
end

function [X,y,t]=buildXY(TMI,TFA,tMI,tFA,ch,bins)
    nMI=numel(TMI); nFA=numel(TFA); X=zeros(nMI+nFA,numel(ch));
    for k=1:nMI, X(k,:)=mean(TMI{k}(ch,bins),2).'; end
    for k=1:nFA, X(nMI+k,:)=mean(TFA{k}(ch,bins),2).'; end
    y=[ones(nMI,1); zeros(nFA,1)]; t=[tMI(:); tFA(:)];
    bad=any(~isfinite(X),2); X=X(~bad,:); y=y(~bad); t=t(~bad);
end

function a=cvBal(X,y,t)
% time-blocked 5-fold CV, balanced accuracy (mean of per-class recall)
    [~,ord]=sort(t); fold=zeros(size(y)); n=numel(y); k=5;
    edges=round(linspace(0,n,k+1));
    for i=1:k, fold(ord(edges(i)+1:edges(i+1)))=i; end
    yp=nan(size(y));
    for i=1:k
        tr=fold~=i; te=fold==i;
        if numel(unique(y(tr)))<2, continue; end
        Xp=X(tr&y==1,:); Xn=X(tr&y==0,:);
        if size(Xp,1)<3||size(Xn,1)<3, continue; end
        m1=mean(Xp,1); m0=mean(Xn,1);
        Sw=((size(Xp,1)-1)*cov(Xp)+(size(Xn,1)-1)*cov(Xn))/(size(Xp,1)+size(Xn,1)-2);
        Sw=Sw+1e-2*mean(diag(Sw))*eye(size(Sw,1));
        wv=Sw\(m1-m0)'; b=-wv'*((m1+m0)'/2);
        yp(te)=double((X(te,:)*wv+b)>0);
    end
    ok=~isnan(yp);
    r1=mean(yp(ok&y==1)==1); r0=mean(yp(ok&y==0)==0);
    a=nanmean([r1 r0]); %#ok<NANMEAN>
end
function o=ternary(c,a,b), if c, o=a; else, o=b; end, end
