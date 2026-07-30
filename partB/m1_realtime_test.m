function m1_realtime_test
%M1_REALTIME_TEST  Two things that matter for deployment:
%  (1) HELD-OUT within a session: train on the first 60%, detect on the unseen
%      last 40% with a threshold calibrated on train only -> the honest
%      "calibrate at session start, then run live on new data" scenario.
%  (2) CROSS-DAY: Day1 model on Day2, un-aligned vs Procrustes-aligned, to show
%      the manifold is preserved (coordinates just rotate).
    root='C:\Users\tiwar133\gpfa_v0203\Data_Realtime'; rng(7); nLag=12; nCh=64; refrS=2; dwell=2;
    d1=fullfile(root,'MouseM1\Day1\eOPN_M1_DualShank_Day1_260721_195215');
    d2=fullfile(root,'MouseM1\Day2\eOPN_M1_DualShank_Day2_260722_163423');

    % =============== (1) within-session held-out on Day2 ===============
    [X,bT,dt,on,cue]=ld(d2,nCh); [pos,rest]=lab(bT,on,cue); refr=round(refrS/dt);
    cM=mean(X,2); cS=std(X,0,2)+eps; F=lf((X-cM)./cS,nLag);
    t0=bT(1); t1=bT(end); split=t0+0.6*(t1-t0);
    tr=bT<=split; te=bT>split;
    w=lda(F(pos&tr,:),F(rest&tr,:)); a=w(1:end-1); b=w(end);
    sc=(F*a+b).'; rm=mean(sc(rest&tr)); rs=std(sc(rest&tr))+eps; scs=(sc-rm)/rs;
    th=pickTh(scs,bT,on,cue,rest,dt,refr,dwell,tr,5);
    [rTe,faTe,ldTe,pbTe]=evalM(scs,bT,on,cue,rest,dt,refr,dwell,th,te);
    [rTr,faTr]=evalM(scs,bT,on,cue,rest,dt,refr,dwell,th,tr);
    fprintf('\n===== (1) Day2 SAME-SESSION, held-out (train first 60%%, test last 40%%) =====\n');
    fprintf('  theta %.2f SD (calibrated on train)\n',th);
    fprintf('  TRAIN part : %.0f%% recall, %.1f FA/min\n',100*rTr,faTr);
    fprintf('  UNSEEN part: %.0f%% recall, %.1f FA/min, median lead %+.0f ms, %.0f%% precede\n',100*rTe,faTe,ldTe,100*pbTe);

    % =============== (2) cross-day Day1 -> Day2 in K-dim PCA latent ===============
    [X1,bT1,dt1,on1,cue1]=ld(d1,nCh); [p1,r1]=lab(bT1,on1,cue1); K=15; %#ok<ASGLU>
    Z1=(X1-mean(X1,2))./(std(X1,0,2)+eps); Z2=(X-mean(X,2))./(std(X,0,2)+eps);
    W1=pcaBasis(Z1,K); L1=W1'*Z1; W2=pcaBasis(Z2,K); L2=W2'*Z2;   % each day its own K-dim latent
    F1=lf(L1,nLag); w1=lda(F1(p1,:),F1(r1,:)); a1=w1(1:end-1); b1=w1(end);
    s1=(F1*a1+b1).'; rm1=mean(s1(r1)); rs1=std(s1(r1))+eps;
    F2u=lf(L2,nLag); s2u=((F2u*a1+b1).'-rm1)/rs1; aucU=auc(s2u(pos),s2u(rest));      % un-aligned
    A1=pullAvg(L1,bT1,on1); A2=pullAvg(L2,bT,on);                                    % K x T anchors
    [U,~,V]=svd(A1*A2'); R=U*V';                                                     % K x K rotation
    L2a=R*L2; F2a=lf(L2a,nLag); s2a=((F2a*a1+b1).'-rm1)/rs1; aucA=auc(s2a(pos),s2a(rest));
    aucD2=cvauc(F2u_base(L2,nLag),pos,rest,bT);                                      % within-Day2 latent CV
    fprintf('\n===== (2) CROSS-DAY  Day1 model -> Day2 (K=%d latent) =====\n',K);
    fprintf('  un-aligned (zero-shot)      AUC  %.3f\n',aucU);
    fprintf('  Procrustes-aligned          AUC  %.3f\n',aucA);
    fprintf('  within-Day2 latent (CV)     AUC  %.3f\n',aucD2);
    fprintf('=============================================\n');
end

function F=F2u_base(L,nLag), F=lf(L,nLag); end
function W=pcaBasis(Z,K)
    C=cov(Z'); [V,D]=eig((C+C')/2); [~,o]=sort(diag(D),'descend'); W=V(:,o(1:K));
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
function a=cvauc(F,pos,rest,bT)
    N=size(F,1); li=find(bT>bT(1)+2); ed=linspace(bT(li(1)),bT(li(end))+1e-6,6); fold=discretize(bT,ed); sc=nan(1,N);
    for k=1:5, tp=find(pos&fold~=k); tr=find(rest&fold~=k); if numel(tr)>5*numel(tp), tr=tr(randperm(numel(tr),5*numel(tp))); end
        w=lda(F(tp,:),F(tr,:)); te=find(fold==k); sc(te)=(F(te,:)*w(1:end-1)+w(end)).'; end
    a=auc(sc(pos),sc(rest));
end
function th=pickTh(sc,bT,on,cue,rest,dt,refr,dwell,mask,faT)
    onM=on(on>=min(bT(mask))&on<=max(bT(mask))); restMin=sum(rest&mask)*dt/60;
    ths=linspace(0,prctile(sc(mask),99.8),60); fa=zeros(size(ths));
    for i=1:numel(ths), al=binAl(sc,bT,ths(i),dwell,refr,mask);
        fa(i)=sum(arrayfun(@(x)~any(abs(cue-x)<=2)&~any(abs(on-x)<=1),al))/max(restMin,eps); end
    [~,j]=min(abs(fa-faT)); th=ths(j); %#ok<NASGU>
    if isempty(onM), end
end
function [rec,fa,medlead,pbefore]=evalM(sc,bT,on,cue,rest,dt,refr,dwell,th,mask)
    al=binAl(sc,bT,th,dwell,refr,mask); onM=on(on>=min(bT(mask))&on<=max(bT(mask)));
    restMin=sum(rest&mask)*dt/60;
    rec=mean(arrayfun(@(o)any(al>=o-0.3&al<=o+0.1),onM));
    fa=sum(arrayfun(@(x)~any(abs(cue-x)<=2)&~any(abs(on-x)<=1),al))/max(restMin,eps);
    lead=[]; for o=onM, c=al(al>=o-0.5&al<=o+0.15); if ~isempty(c), lead(end+1)=1000*(o-c(end)); end, end %#ok<AGROW>
    medlead=median(lead); if isnan(medlead), medlead=0; end; pbefore=mean(lead>0); if isnan(pbefore), pbefore=0; end
end
function al=binAl(sc,bT,th,dwell,refr,mask)
    pred=(sc>th)&mask; al=[]; run=0; last=-inf;
    for j=1:numel(sc), if pred(j),run=run+1; else,run=0; end
        if run>=dwell&&(j-last)>=refr, al(end+1)=bT(j); last=j; end, end %#ok<AGROW>
end
