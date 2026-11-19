%% TRAIN_BEST_MODEL
% Pushes the learned detector as far as a real-time-capable model allows:
% compares feature sets (PC scores vs raw channels, short vs long temporal
% window) under the same 5-fold time-blocked CV, picks the best, generates
% cue-agnostic alarm times for the whole session from the cross-validated
% scores, and saves them so the trial clips can be re-rendered with the
% learned detector for a side-by-side vs CUSUM.
clear; clc; close all;
root=dataRoot();
data=fullfile(root,'61199SomDay4Recording1_241213_181033');

S =load(fullfile(data,'online_binned_data.mat'),'online_binned','binCenterTimes','binSizeSec','calibCutoffTime');
ev=load(fullfile(data,'behavioral_events.mat'),'leverTrace','leverTimeVec','fs','cueTimes','rewardTimes');
R =load(fullfile(data,'trajectory_cusum_offline_results.mat'),'pcaModel','alarmTimes');
X=double(S.online_binned); binT=S.binCenterTimes; dt=S.binSizeSec; pm=R.pcaModel;
cue=ev.cueTimes(:)'; lt=ev.leverTimeVec(:)'; lv=ev.leverTrace(:)'; fs=ev.fs; N=size(X,2);

% signals
z=(X-pm.mu)./pm.sigma; SC=pm.coeff'*z;                 % 10 x N  (frozen PC scores)
Xz=(X-mean(X,2))./(std(X,0,2)+eps);                    % 64 x N  z-scored raw counts

% movement onsets
thr=load(fullfile(data,'lever_validation_results.mat'),'thr'); thr=thr.thr;
rest=median(lv); madR=median(abs(lv-rest)); onsetM=rest+5*madR;
above=lv>thr; rise=find(diff(above)>0)+1; onset=[]; last=-inf;
for r=rise; tc=lt(r); if tc-last<1; continue; end
    i=r; lim=r-round(0.6*fs); while i>1&&i>lim&&lv(i)>onsetM; i=i-1; end; onset(end+1)=lt(i); last=tc; end %#ok<SAGROW>

% labels + folds
live=binT>=S.calibCutoffTime;
posMask=false(1,N); for o=onset; posMask=posMask|(binT>=o-0.18 & binT<=o); end
nearCue=false(1,N); for c=cue; nearCue=nearCue|(abs(binT-c)<=2); end
nearMv =false(1,N); for o=onset; nearMv=nearMv|(abs(binT-o)<=1); end
restMask=~nearCue & ~nearMv & live; posMask=posMask & live;
liveIdx=find(live); edges=linspace(binT(liveIdx(1)),binT(liveIdx(end))+1e-6,6); fold=discretize(binT,edges);
fprintf('%d onsets | pre-move bins %d | rest bins %d\n',numel(onset),sum(posMask),sum(restMask));

% feature sets
sets={ '10PC x3',  lagFeat(SC,3); ...
       '10PC x10', lagFeat(SC,10); ...
       'raw64 x8', lagFeat(Xz,8); ...
       'raw64 x12',lagFeat(Xz,12) };
fprintf('\n feature set     dim    CV-AUC\n');
best=-1; bestScore=[]; bestName='';
for s=1:size(sets,1)
    Fm=sets{s,2}; [auc,sc]=cvAUC(Fm,posMask,restMask,fold,live);
    fprintf('  %-10s   %4d   %.3f\n', sets{s,1}, size(Fm,2), auc);
    if auc>best; best=auc; bestScore=sc; bestName=sets{s,1}; end
end
fprintf('BEST: %s  (AUC %.3f)\n', bestName, best);

% ---- generate cue-agnostic alarms from the best cross-validated score ----
restMin=sum(restMask)*dt/60; refrB=round(2.0/dt); dwell=2;
thrs=linspace(prctile(bestScore(live),50),prctile(bestScore(live),99.8),50);
rec=zeros(size(thrs)); fa=zeros(size(thrs));
for ii=1:numel(thrs)
    al=binAlarms(bestScore,binT,thrs(ii),dwell,refrB,live);
    rec(ii)=mean(arrayfun(@(o) any(al>=o-0.3 & al<=o+0.1),onset));
    fa(ii)=sum(arrayfun(@(a) ~any(abs(cue-a)<=2)&~any(abs(onset-a)<=1),al))/restMin;
end
% operating point = knee (closest to ideal corner: recall 1, fa 0)
faN=fa/max(fa); [~,kn]=min((1-rec).^2 + faN.^2); opThr=thrs(kn);
alarmTimes=binAlarms(bestScore,binT,opThr,dwell,refrB,live);

% CUSUM baseline on same metrics
alC=R.alarmTimes(:)';
recCU=mean(arrayfun(@(o) any(alC>=o-0.3&alC<=o+0.1),onset));
faCU =sum(arrayfun(@(a) ~any(abs(cue-a)<=2)&~any(abs(onset-a)<=1),alC))/restMin;
fprintf('\n=== operating point (knee) ===\n');
fprintf('LEARNED : recall %.0f%%  rest FA %.1f/min  (%d alarms)\n',100*rec(kn),fa(kn),numel(alarmTimes));
fprintf('CUSUM   : recall %.0f%%  rest FA %.1f/min  (%d alarms)\n',100*recCU,faCU,numel(alC));
fprintf('  -> %.0f%% fewer rest false alarms at %+.0f pts recall\n',100*(1-fa(kn)/faCU),100*(rec(kn)-recCU));

% ---- save for clip regeneration (format precompute_session_trajectory expects) ----
pcaModel=pm; K=8; %#ok<NASGU>
save(fullfile(data,'classifier_results.mat'),'alarmTimes','pcaModel','K','bestScore','binT','opThr','bestName','best');

% ---- figure ----
figure('Color','w','Position',[80 80 780 560]);
plot(100*rec,fa,'-','LineWidth',2.5,'Color',[.10 .55 .60]); hold on; grid on;
plot(100*recCU,faCU,'p','MarkerSize',18,'MarkerFaceColor',[.84 .22 .19],'MarkerEdgeColor','k');
plot(100*rec(kn),fa(kn),'o','MarkerSize',12,'MarkerFaceColor',[.10 .55 .60],'MarkerEdgeColor','k');
xlabel('movement recall (%)'); ylabel('false alarms per minute of REST');
title(sprintf('Best learned detector: %s  (CV AUC %.2f)',bestName,best),'FontSize',13);
legend({'learned (operating curve)','current CUSUM','chosen operating point'},'Location','northwest','Box','off');
saveas(gcf,fullfile(data,'best_model_vs_cusum.png'));
fprintf('Saved classifier_results.mat + best_model_vs_cusum.png\nDONE\n');

%% helpers
function Fm=lagFeat(D,W)
    [d,N]=size(D); Fm=zeros(N,d*W);
    for w=0:W-1; Fm(:,w*d+(1:d))=[zeros(w,d); D(:,1:N-w)']; end
end
function [auc,score]=cvAUC(Fm,posMask,restMask,fold,live)
    score=nan(1,size(Fm,1));
    for k=1:5
        trP=find(posMask&fold~=k); trR=find(restMask&fold~=k);
        if numel(trR)>5*numel(trP); trR=trR(randperm(numel(trR),5*numel(trP))); end
        w=ldaTrain(Fm(trP,:),Fm(trR,:));
        te=find(fold==k & live); score(te)=[Fm(te,:) ones(numel(te),1)]*w;
    end
    auc=aucOf(score(posMask),score(restMask));
end
function w=ldaTrain(Xp,Xn)
    mu1=mean(Xp,1); mu0=mean(Xn,1);
    Sw=((size(Xp,1)-1)*cov(Xp)+(size(Xn,1)-1)*cov(Xn))/(size(Xp,1)+size(Xn,1)-2);
    Sw=Sw+1e-2*mean(diag(Sw))*eye(size(Sw,1));
    a=Sw\(mu1-mu0)'; b=-a'*((mu1+mu0)'/2); w=[a;b];
end
function a=aucOf(sp,sn)
    sp=sp(~isnan(sp)); sn=sn(~isnan(sn));
    allv=[sp(:);sn(:)]; lab=[ones(numel(sp),1);zeros(numel(sn),1)];
    [~,ord]=sort(allv,'descend'); lab=lab(ord);
    tpr=cumsum(lab)/sum(lab); fpr=cumsum(1-lab)/sum(1-lab); a=trapz(fpr,tpr);
end
function al=binAlarms(score,binT,thrv,dwell,refrB,live)
    pred=(score>thrv)&live; al=[]; run=0; last=-inf;
    for j=1:numel(score)
        if pred(j),run=run+1; else,run=0; end
        if run>=dwell && (j-last)>=refrB; al(end+1)=binT(j); last=j; end %#ok<SAGROW>
    end
end
