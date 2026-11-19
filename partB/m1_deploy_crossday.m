function m1_deploy_crossday
%M1_DEPLOY_CROSSDAY  Build a "Day1-trained, Day2-aligned" model the GUI can run
%   causally on unseen Day2 data. Trains LDA in a K-dim Day1 latent, aligns to
%   Day2 using ONLY the first 40% of Day2 (session-start calibration), then
%   COLLAPSES the (PCA -> rotate -> LDA) chain into one 64ch x 12lag weight
%   vector -> saves ClosedLoopGUI_standalone/live_model.mat. Held-out Day2 (last
%   60%) is used only to verify, never for building.
    root=dataRoot(); rng(7); nLag=12; nCh=64; K=15; calFrac=0.40;
    d1=fullfile(root,'MouseM1', 'Day1', 'eOPN_M1_DualShank_Day1_260721_195215');
    d2=fullfile(root,'MouseM1', 'Day2', 'eOPN_M1_DualShank_Day2_260722_163423');
    outPath=fullfile(root,'ClosedLoopGUI_standalone','live_model.mat');

    % ---- Day1 reference: latent LDA + pull-anchor ----
    [X1,bT1,dt,on1,cue1]=ld(d1,nCh); [p1,r1]=lab(bT1,on1,cue1);
    Z1=(X1-mean(X1,2))./(std(X1,0,2)+eps); W1=pcaBasis(Z1,K); L1=W1'*Z1;
    F1=lf(L1,nLag); w=lda(F1(p1,:),F1(r1,:)); latW=w(1:end-1); b1=w(end);
    A1=pullAvg(L1,bT1,on1);

    % ---- Day2 session-start calibration (first calFrac only) ----
    [X2,bT2,~,on2,cue2]=ld(d2,nCh);
    tCut=bT2(1)+calFrac*(bT2(end)-bT2(1)); cal=bT2<=tCut; te=bT2>tCut;
    Xc=X2(:,cal); cM2=mean(Xc,2); cS2=std(Xc,0,2)+eps;
    Z2=(X2-cM2)./cS2; Z2c=Z2(:,cal);
    W2=pcaBasis(Z2c,K); L2c=W2'*Z2c;
    onc=on2(on2<=tCut); A2=pullAvg(L2c,bT2(cal),onc);
    [U,~,V]=svd(A1*A2'); R=U*V';                          % align Day2 latent -> Day1 latent

    % ---- collapse (PCA W2 -> rotate R -> latent LDA) into one 64ch x 12lag weight ----
    rawW=zeros(nCh*nLag,1);
    for l=0:nLag-1, rawW(l*nCh+(1:nCh))=W2*R'*latW(l*K+(1:K)); end
    b=b1;

    % ---- threshold from calibration rest ----
    [pc,rc]=lab(bT2(cal),onc,cue2(cue2<=tCut)); %#ok<ASGLU>
    Fc=lf(Z2c,nLag); scC=(Fc*rawW+b).'; restMean=mean(scC(rc)); restStd=std(scC(rc))+eps;
    scCs=(scC-restMean)/restStd; refr=round(2/dt);
    ths=linspace(0,prctile(scCs,99.8),60); fa=zeros(size(ths));
    for i=1:numel(ths), al=binAl(scCs,bT2(cal),ths(i),2,refr,true(1,sum(cal)));
        fa(i)=sum(arrayfun(@(x)~any(abs(cue2-x)<=2)&~any(abs(on2-x)<=1),al))/max(sum(rc)*dt/60,eps); end
    [~,j]=min(abs(fa-5)); theta=ths(j);

    % ---- save GUI model (standard raw-feature format) ----
    mdl=struct('w',rawW,'b',b,'chMean',cM2,'chStd',cS2,'nLag',nLag,'nChan',nCh, ...
        'theta',theta,'thetaRaw',theta*restStd+restMean,'restMean',restMean,'restStd',restStd, ...
        'dwell',2,'refractorySec',2.0,'binSizeSec',dt, ...
        'trainSession','Day1-trained, Day2-aligned','trainName','Day1->Day2 ALIGNED','trainDate',datestr(now,'yyyy-mm-dd HH:MM'), ...
        'nPulls',numel(on1),'cvAUC',NaN,'note','cross-day aligned via 40% Day2 calibration');
    save(outPath,'-struct','mdl');

    % ---- verify on HELD-OUT Day2 (last 60%, never used to build) ----
    [pT,rT]=lab(bT2,on2,cue2); pT=pT&te; rT=rT&te;
    F2=lf(Z2,nLag); sc2=((F2*rawW+b).'-restMean)/restStd;
    aucHeld=auc(sc2(pT),sc2(rT));
    onT=on2(on2>tCut); restMin=sum(rT)*dt/60;
    al=binAl(sc2,bT2,theta,2,refr,te);
    recHeld=mean(arrayfun(@(o)any(al>=o-0.3&al<=o+0.1),onT));
    faHeld=sum(arrayfun(@(x)~any(abs(cue2-x)<=2)&~any(abs(on2-x)<=1),al))/max(restMin,eps);
    lds=[]; for o=onT, c=al(al>=o-0.5&al<=o+0.15); if ~isempty(c), lds(end+1)=1000*(o-c(end)); end, end %#ok<AGROW>
    fprintf('\n=== SAVED aligned model -> %s ===\n',outPath);
    fprintf('theta %.2f SD | built from Day1 + first %.0f%% of Day2\n',theta,100*calFrac);
    fprintf('HELD-OUT Day2 (last 60%%, unseen): AUC %.3f | %.0f%% recall @ %.1f FA/min | median lead %+.0f ms\n', ...
        aucHeld,100*recHeld,faHeld,median(lds));
    fprintf('Load this in the GUI, Browse the LAST ~30 Day2 files, START.\n');
end

function A=pullAvg(Z,bT,onset)
    tg=-0.3:0.02:0.1; A=zeros(size(Z,1),numel(tg)); n=0;
    for o=onset, s=interp1(bT,Z',o+tg,'linear',NaN)'; if ~any(isnan(s(:))), A=A+s; n=n+1; end, end
    A=A/max(n,1);
end
function [X,binT,dt,onset,cue]=ld(session,nCh)
    S=load(fullfile(session,'online_binned_data.mat'),'online_binned','binCenterTimes','binSizeSec');
    ev=load(fullfile(session,'behavioral_events.mat'),'leverTrace','leverTimeVec','fs','cueTimes');
    X=double(S.online_binned); X=X(1:min(nCh,size(X,1)),:); binT=S.binCenterTimes(:)'; dt=S.binSizeSec;
    lt=ev.leverTimeVec(:)'; lv=ev.leverTrace(:)'; fs=ev.fs; cue=ev.cueTimes(:)';
    pc=prctile(lv,[50 99]); rr=pc(1); thr=rr+0.33*(pc(2)-rr); mR=median(abs(lv-rr)); oM=rr+5*mR;
    ab=lv>thr; ri=find(diff(ab)>0)+1; onset=[]; last=-inf;
    for r=ri, tc=lt(r); if tc-last<1, continue; end
        i=r; lim=r-round(0.6*fs); while i>1&&i>lim&&lv(i)>oM, i=i-1; end; onset(end+1)=lt(i); last=tc; end %#ok<AGROW>
end
function [pos,rest]=lab(bT,onset,cue)
    N=numel(bT); pos=false(1,N); for o=onset, pos=pos|(bT>=o-0.18&bT<=o); end
    nc=false(1,N); for c=cue, nc=nc|(abs(bT-c)<=2); end
    nm=false(1,N); for o=onset, nm=nm|(abs(bT-o)<=1); end
    live=bT>bT(1)+2; rest=~nc&~nm&live; pos=pos&live;
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
function W=pcaBasis(Z,K), C=cov(Z'); [V,D]=eig((C+C')/2); [~,o]=sort(diag(D),'descend'); W=V(:,o(1:K)); end
function al=binAl(sc,bT,th,dwell,refr,mask)
    pred=(sc>th)&mask; al=[]; run=0; last=-inf;
    for j=1:numel(sc), if pred(j),run=run+1; else,run=0; end
        if run>=dwell&&(j-last)>=refr, al(end+1)=bT(j); last=j; end, end %#ok<AGROW>
end
