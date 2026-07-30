function export_live_model(trainSession, faTarget)
%EXPORT_LIVE_MODEL  Train ONE deployable Fisher-LDA movement-intention
%   detector and freeze everything the live GUI needs into live_model.mat.
%
%   train_best_model.m compares feature sets with 5-fold CV but saves only the
%   cross-validated SCORES (no single weight vector). For deployment we need a
%   single frozen model: the weight vector w, the bias b, the per-channel
%   normalisation, the 12-lag feature spec, and an operating threshold.
%
%   export_live_model                       % default session, ~5 FA/min
%   export_live_model('61198SOMDay3...', 5) % pick training session + FA budget
%
%   The live scorer (closedloop_live_gui) does, per 20 ms bin, exactly:
%       z    = (count64 - chMean) ./ chStd
%       feat = [z_t; z_{t-1}; ...; z_{t-11}]        (768-dim, 12 lags)
%       score= w' * feat + b
%       alarm when score > theta for `dwell` bins, then `refractorySec` lockout

    root='C:\Users\tiwar133\gpfa_v0203\Data_Realtime';
    if nargin<1 || isempty(trainSession), trainSession='61198SOMDay2Recording1_241214_121351'; end
    if nargin<2, faTarget=5.0; end
    nLag=12; dwell=2; refractorySec=2.0; posPreSec=0.18;
    data=fullfile(root,trainSession);

    S =load(fullfile(data,'online_binned_data.mat'),'online_binned','binCenterTimes','binSizeSec','calibCutoffTime');
    ev=load(fullfile(data,'behavioral_events.mat'),'leverTrace','leverTimeVec','fs','cueTimes');
    tv=load(fullfile(data,'lever_validation_results.mat'),'thr');
    X=double(S.online_binned); binT=S.binCenterTimes; dt=S.binSizeSec; nChan=size(X,1);
    lt=ev.leverTimeVec(:)'; lv=ev.leverTrace(:)'; fs=ev.fs; cue=ev.cueTimes(:)'; thr=tv.thr;

    % movement onsets (backtracked to true departure from rest)
    rest=median(lv); madR=median(abs(lv-rest)); onsetM=rest+5*madR;
    ab=lv>thr; ri=find(diff(ab)>0)+1; onset=[]; last=-inf;
    for r=ri; tc=lt(r); if tc-last<1; continue; end
        i=r; lim=r-round(0.6*fs); while i>1&&i>lim&&lv(i)>onsetM; i=i-1; end; onset(end+1)=lt(i); last=tc; end %#ok<AGROW>

    % frozen per-channel normalisation (used identically online)
    chMean=mean(X,2); chStd=std(X,0,2)+eps;
    Xz=(X-chMean)./chStd;
    F=lagFeat(Xz,nLag);   % N x (nChan*nLag)

    live=binT>=S.calibCutoffTime; N=size(X,2);
    pos=false(1,N); for o=onset; pos=pos|(binT>=o-posPreSec & binT<=o); end
    nc=false(1,N); for c=cue; nc=nc|(abs(binT-c)<=2); end
    nm=false(1,N); for o=onset; nm=nm|(abs(binT-o)<=1); end
    rest_m=~nc&~nm&live; pos=pos&live;

    % one final LDA on ALL data (no CV -- this is the deployed model)
    wfull=ldaTrain(F(pos,:),F(rest_m,:));   % [(nChan*nLag)+1] x 1  (last elem = bias)
    w=wfull(1:end-1); b=wfull(end);

    % pick theta at the requested rest-FA budget on this session
    sc=(F*w + b).';
    refrB=round(refractorySec/dt); restMin=sum(rest_m)*dt/60;
    ths=linspace(prctile(sc(live),50),prctile(sc(live),99.9),60);
    rec=zeros(size(ths)); fa=zeros(size(ths));
    for i=1:numel(ths)
        al=binAlarms(sc,binT,ths(i),dwell,refrB,live);
        rec(i)=mean(arrayfun(@(o) any(al>=o-0.3&al<=o+0.1),onset));
        fa(i)=sum(arrayfun(@(a) ~any(abs(cue-a)<=2)&~any(abs(onset-a)<=1),al))/restMin;
    end
    [~,j]=min(abs(fa-faTarget)); thetaRaw=ths(j);
    al=binAlarms(sc,binT,thetaRaw,dwell,refrB,live);
    recAt=mean(arrayfun(@(o) any(al>=o-0.3&al<=o+0.1),onset));

    % standardise scores to the rest distribution -> theta in "SD above rest"
    % units, so online retraining (which rescales w) keeps a stable threshold.
    restMean=mean(sc(rest_m)); restStd=std(sc(rest_m))+eps;
    theta=(thetaRaw-restMean)/restStd;

    model=struct('w',w,'b',b,'chMean',chMean,'chStd',chStd,'nLag',nLag,'nChan',nChan, ...
        'theta',theta,'thetaRaw',thetaRaw,'restMean',restMean,'restStd',restStd, ...
        'dwell',dwell,'refractorySec',refractorySec,'binSizeSec',dt, ...
        'trainSession',trainSession,'faTargetPerMin',faTarget,'note', ...
        'score=(w''*feat+b - restMean)/restStd ; feat=[z_t..z_{t-11}] ; z=(count-chMean)./chStd ; alarm if score>theta(SD) for dwell bins');
    save(fullfile(root,'live_model.mat'),'-struct','model');

    fprintf('Trained live LDA on %s\n', trainSession);
    fprintf('  features %d (%d ch x %d lags) | theta %.2f SD-above-rest @ %.1f FA/min | recall %.0f%% | dwell %d, refr %.1fs\n', ...
        numel(w), nChan, nLag, theta, faTarget, 100*recAt, dwell, refractorySec);
    fprintf('Saved live_model.mat\n');
end

function Fm=lagFeat(D,W)
    [d,N]=size(D); Fm=zeros(N,d*W);
    for w=0:W-1; Fm(:,w*d+(1:d))=[zeros(w,d); D(:,1:N-w)']; end
end
function w=ldaTrain(Xp,Xn)
    mu1=mean(Xp,1); mu0=mean(Xn,1);
    Sw=((size(Xp,1)-1)*cov(Xp)+(size(Xn,1)-1)*cov(Xn))/(size(Xp,1)+size(Xn,1)-2);
    Sw=Sw+1e-2*mean(diag(Sw))*eye(size(Sw,1));
    a=Sw\(mu1-mu0)'; bb=-a'*((mu1+mu0)'/2); w=[a;bb];
end
function al=binAlarms(sc,binT,thv,dwell,refrB,live)
    pred=(sc>thv)&live; al=[]; run=0; last=-inf;
    for j=1:numel(sc)
        if pred(j),run=run+1; else,run=0; end
        if run>=dwell && (j-last)>=refrB; al(end+1)=binT(j); last=j; end %#ok<AGROW>
    end
end
