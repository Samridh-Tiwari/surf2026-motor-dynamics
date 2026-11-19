function m1_crossday
%M1_CROSSDAY  Honest cross-day test: train Fisher-LDA on Day1, predict Day2.
%   Uses 64 channels on both (Port B, B-000..B-063), cue-aware rest labels,
%   Day1-frozen normalization applied to Day2 (exactly how the GUI deploys it).
    root=dataRoot(); rng(7); nLag=12; nCh=64;
    d1=fullfile(root,'MouseM1', 'Day1', 'eOPN_M1_DualShank_Day1_260721_195215');
    d2=fullfile(root,'MouseM1', 'Day2', 'eOPN_M1_DualShank_Day2_260722_163423');
    [X1,bT1,dt,on1,cue1,lv1]=ld(d1,nCh);
    [X2,bT2,~ ,on2,cue2,lv2]=ld(d2,nCh); %#ok<ASGLU>
    [p1,r1]=lab(bT1,on1,cue1); [p2,r2]=lab(bT2,on2,cue2);
    fprintf('Day1: %d ch x %d bins | %d pulls %d cues | pos %d rest %d\n',size(X1,1),size(X1,2),numel(on1),numel(cue1),sum(p1),sum(r1));
    fprintf('Day2: %d ch x %d bins | %d pulls %d cues | pos %d rest %d\n',size(X2,1),size(X2,2),numel(on2),numel(cue2),sum(p2),sum(r2));

    % Day1-frozen per-channel normalization (used for BOTH days, as the GUI does)
    cM=mean(X1,2); cS=std(X1,0,2)+eps;
    F1=lf((X1-cM)./cS,nLag); F2=lf((X2-cM)./cS,nLag);
    w=lda(F1(p1,:),F1(r1,:)); a=w(1:end-1); b=w(end);
    sc1=(F1*a+b).'; rm=mean(sc1(r1)); rs=std(sc1(r1))+eps;
    sc1s=(sc1-rm)/rs; sc2s=((F2*a+b).'-rm)/rs;

    aucD1=auc(sc1s(p1),sc1s(r1));                 % Day1 in-sample (train)
    aucC =auc(sc2s(p2),sc2s(r2));                 % Day1 model on Day2 (CROSS-DAY)
    % within-Day2 reference (Day2 own normalization + CV)
    cM2=mean(X2,2); cS2=std(X2,0,2)+eps; G2=lf((X2-cM2)./cS2,nLag);
    aucD2=cvauc(G2,p2,r2,bT2);

    % operating point: Day1 theta @ ~5 FA/min, applied to Day2
    dt2=median(diff(bT2)); refr=round(2/dt); refr2=round(2/dt2);
    thD1=pickTheta(sc1s,bT1,on1,cue1,r1,dt,refr);
    [recD2,faD2]=opAt(sc2s,bT2,on2,cue2,r2,dt2,refr2,thD1);
    [recD1,faD1]=opAt(sc1s,bT1,on1,cue1,r1,dt ,refr ,thD1);

    fprintf('\n================ RESULTS ================\n');
    fprintf('within-Day1 (train, in-sample)   AUC  %.3f\n',aucD1);
    fprintf('within-Day2 (own model, 5-fold CV) AUC %.3f\n',aucD2);
    fprintf('Day1 model -> Day2 (CROSS-DAY)   AUC  %.3f\n',aucC);
    fprintf('-----------------------------------------\n');
    fprintf('Day1 theta=%.2f SD @ %.1f FA/min -> on Day1: %.0f%% recall\n',thD1,faD1,100*recD1);
    fprintf('                                    on Day2: %.0f%% recall, %.1f FA/min\n',100*recD2,faD2);
    fprintf('=========================================\n');
end

function [X,binT,dt,onset,cue,lv]=ld(session,nCh)
    S=load(fullfile(session,'online_binned_data.mat'),'online_binned','binCenterTimes','binSizeSec','calibCutoffTime');
    ev=load(fullfile(session,'behavioral_events.mat'),'leverTrace','leverTimeVec','fs','cueTimes');
    X=double(S.online_binned); X=X(1:min(nCh,size(X,1)),:); binT=S.binCenterTimes; dt=S.binSizeSec;
    lt=ev.leverTimeVec(:)'; lv=ev.leverTrace(:)'; fs=ev.fs; cue=ev.cueTimes(:)';
    pc=prctile(lv,[50 99]); rest=pc(1); thr=rest+0.33*(pc(2)-rest); mR=median(abs(lv-rest)); oM=rest+5*mR;
    ab=lv>thr; ri=find(diff(ab)>0)+1; onset=[]; last=-inf;
    for r=ri, tc=lt(r); if tc-last<1, continue; end
        i=r; lim=r-round(0.6*fs); while i>1&&i>lim&&lv(i)>oM, i=i-1; end; onset(end+1)=lt(i); last=tc; end %#ok<AGROW>
end

function [pos,rest]=lab(binT,onset,cue)
    N=numel(binT); pos=false(1,N); for o=onset, pos=pos|(binT>=o-0.18&binT<=o); end
    nc=false(1,N); for c=cue, nc=nc|(abs(binT-c)<=2); end
    nm=false(1,N); for o=onset, nm=nm|(abs(binT-o)<=1); end
    live=binT>binT(1)+2; rest=~nc&~nm&live; pos=pos&live;
end

function F=lf(D,W), [d,N]=size(D); F=zeros(N,d*W); for w=0:W-1, F(:,w*d+(1:d))=[zeros(w,d); D(:,1:N-w)']; end, end
function w=lda(Xp,Xn)
    m1=mean(Xp,1); m0=mean(Xn,1); Sw=((size(Xp,1)-1)*cov(Xp)+(size(Xn,1)-1)*cov(Xn))/(size(Xp,1)+size(Xn,1)-2);
    Sw=Sw+1e-2*mean(diag(Sw))*eye(size(Sw,1)); aa=Sw\(m1-m0)'; bb=-aa'*((m1+m0)'/2); w=[aa;bb];
end
function a=auc(sp,sn)
    sp=sp(~isnan(sp)); sn=sn(~isnan(sn)); v=[sp(:);sn(:)]; l=[ones(numel(sp),1);zeros(numel(sn),1)];
    [~,o]=sort(v,'descend'); l=l(o); a=trapz(cumsum(1-l)/sum(1-l),cumsum(l)/sum(l));
end
function a=cvauc(F,pos,rest,binT)
    N=size(F,1); li=find(binT>binT(1)+2); ed=linspace(binT(li(1)),binT(li(end))+1e-6,6); fold=discretize(binT,ed);
    sc=nan(1,N);
    for k=1:5
        tp=find(pos&fold~=k); tr=find(rest&fold~=k); if numel(tr)>5*numel(tp), tr=tr(randperm(numel(tr),5*numel(tp))); end
        w=lda(F(tp,:),F(tr,:)); te=find(fold==k); sc(te)=(F(te,:)*w(1:end-1)+w(end)).';
    end
    a=auc(sc(pos),sc(rest));
end
function th=pickTheta(sc,binT,onset,cue,rest,dt,refr)
    restMin=sum(rest)*dt/60; ths=linspace(0,prctile(sc,99.8),60); fa=zeros(size(ths));
    for i=1:numel(ths), al=binAl(sc,binT,ths(i),2,refr); fa(i)=sum(arrayfun(@(x)~any(abs(cue-x)<=2)&~any(abs(onset-x)<=1),al))/max(restMin,eps); end
    [~,j]=min(abs(fa-5)); th=ths(j);
end
function [rec,fa]=opAt(sc,binT,onset,cue,rest,dt,refr,th)
    al=binAl(sc,binT,th,2,refr); restMin=sum(rest)*dt/60;
    rec=mean(arrayfun(@(o)any(al>=o-0.3&al<=o+0.1),onset));
    fa=sum(arrayfun(@(x)~any(abs(cue-x)<=2)&~any(abs(onset-x)<=1),al))/max(restMin,eps);
end
function al=binAl(sc,binT,th,dwell,refr)
    pred=sc>th; al=[]; run=0; last=-inf;
    for j=1:numel(sc), if pred(j),run=run+1; else,run=0; end
        if run>=dwell&&(j-last)>=refr, al(end+1)=binT(j); last=j; end, end %#ok<AGROW>
end
