function m1m2_timing(sessionDir, tag)
%M1M2_TIMING  Does M2 (premotor) LEAD M1 (motor) before movement?
%   Accuracy comparisons are confounded by channel count and by the cue-evoked
%   response; TIMING is not. Three independent measures, all aligned to genuine
%   (reward-magnitude) lever-pull onsets:
%
%     (A) RAMP LATENCY   - population activity normalised to its own peak, so a
%                          higher-SNR area cannot appear to "lead" trivially.
%                          Time at which each area reaches 25% / 50% of peak.
%     (B) DECODE LATENCY - per time-offset diagonal decoding (movement vs matched
%                          rest). The earliest offset where an area beats chance
%                          is the earliest a BMI could possibly detect from it.
%     (C) CROSS-CORRELATION - peak lag between M1 and M2 population signals
%                          (inherently normalised; sign gives the leader).
%
%   Also reports M1+M2 combined: if it detects earlier than M1 alone, adding the
%   second probe would buy real lead time in the closed-loop rig.
    if nargin<1, sessionDir='C:\Users\tiwar133\gpfa_v0203\Data_Realtime\MouseM1\Day2\eOPN_M1_DualShank_Day2_260722_163423'; end
    if nargin<2, tag='Day2'; end
    rng(7);
    B=load(fullfile(sessionDir,'binned_M1_M2.mat'),'binM1','binM2','binT','binSec');
    ev=load(fullfile(sessionDir,'behavioral_events.mat'));
    dt=B.binSec; bT=B.binT(:).'; N=numel(bT);
    M1=zs(double(B.binM1)); M2=zs(double(B.binM2));
    lv=double(ev.leverTrace(:)'); lt=double(ev.leverTimeVec(:)'); fs=ev.fs;
    rew=[]; if isfield(ev,'rewardTimes'), rew=double(ev.rewardTimes(:)'); end

    % ---- genuine pulls at the ACTUAL hardware reward threshold (day-invariant) ----
    [on,ampThr]=genuine_pulls(sessionDir,1.5);
    on=on(on>2 & on<bT(end)-2);
    fprintf('\n############ %s  M2-vs-M1 TIMING  (%d genuine pulls, hardware thr %.4f V) ############\n',tag,numel(on),ampThr);
    if numel(on)<40, fprintf('!! too few pulls for reliable timing\n'); end

    tau=-0.8:dt:0.3; nT=numel(tau);            % offsets relative to onset
    oi=round(on/dt);                            % onset bin indices

    % ================= (A) normalised population ramp =================
    P1=zeros(numel(oi),nT); P2=zeros(numel(oi),nT); keep=true(1,numel(oi));
    p1=mean(M1,1); p2=mean(M2,1);               % population mean signals
    for k=1:numel(oi)
        idx=oi(k)+round(tau/dt);
        if idx(1)<1||idx(end)>N, keep(k)=false; continue; end
        P1(k,:)=p1(idx); P2(k,:)=p2(idx);
    end
    P1=P1(keep,:); P2=P2(keep,:);
    r1=mean(P1,1); r2=mean(P2,1);
    b1=mean(r1(tau<-0.6)); b2=mean(r2(tau<-0.6));
    n1=(r1-b1)/max(r1-b1); n2=(r2-b2)/max(r2-b2);   % normalise to own peak
    [t25a,t50a]=cross(tau,n1); [t25b,t50b]=cross(tau,n2);
    fprintf('\n(A) RAMP LATENCY (normalised to each area''s own peak)\n');
    fprintf('    M1: 25%% at %+.0f ms, 50%% at %+.0f ms\n',1000*t25a,1000*t50a);
    fprintf('    M2: 25%% at %+.0f ms, 50%% at %+.0f ms\n',1000*t25b,1000*t50b);
    fprintf('    -> M2 leads M1 by %+.0f ms (25%%), %+.0f ms (50%%)\n',1000*(t25a-t25b),1000*(t50a-t50b));

    % ================= (B) diagonal decode latency =================
    % matched rest samples: random times >3 s from any pull
    nr=numel(on); rt=zeros(1,nr); c=0; guard=3;
    while c<nr
        t=rand*(bT(end)-6)+3;
        if all(abs(on-t)>guard), c=c+1; rt(c)=t; end
    end
    ri2=round(rt/dt);
    sets={'M1',M1; 'M2',M2; 'M1+M2',[M1;M2]};
    A=zeros(numel(sets),nT); LAT=nan(1,numel(sets));
    for s=1:size(sets,1)
        D=sets{s,2};
        for j=1:nT
            sh=round(tau(j)/dt);
            Xp=grab(D,oi+sh,N); Xn=grab(D,ri2+sh,N);
            ok=~any(isnan(Xp),2)&~any(isnan(Xn),2); Xp=Xp(ok,:); Xn=Xn(ok,:);
            A(s,j)=cvAUC(Xp,Xn,on(ok).');
        end
        sig=0.5+3*0.5/sqrt(size(Xp,1));         % ~3 SE above chance
        % SUSTAINED crossing: 3 consecutive bins above threshold (a single-bin
        % rule picks up noise blips in the pre-movement band)
        ok3=A(s,:)>sig; run=0; LAT(s)=NaN;
        for j=1:nT
            if tau(j)>0.05, break; end
            if ok3(j), run=run+1; else, run=0; end
            if run>=3, LAT(s)=tau(j-2); break; end
        end
    end
    fprintf('\n(B) DECODE LATENCY (earliest offset beating chance, AUC>%.3f)\n',sig);
    for s=1:size(sets,1)
        fprintf('    %-6s : %+.0f ms   (peak AUC %.3f)\n',sets{s,1},1000*LAT(s),max(A(s,:)));
    end
    fprintf('    -> M2 earlier than M1 by %+.0f ms | M1+M2 earlier than M1 by %+.0f ms\n', ...
        1000*(LAT(1)-LAT(2)),1000*(LAT(1)-LAT(3)));

    % ---- windowed pre-movement AUC with bootstrap CI (robust to blips) ----
    fprintf('\n(B2) PRE-MOVEMENT AUC by window  [M2 - M1, bootstrap 95%% CI]\n');
    wl={'[-300,-200]',[-0.30 -0.20]; '[-200,-100]',[-0.20 -0.10]; '[-100,0]',[-0.10 0]};
    for q=1:size(wl,1)
        sel=tau>=wl{q,2}(1) & tau<=wl{q,2}(2);
        d=zeros(200,1);
        for bIt=1:200
            ii=randi(numel(on),numel(on),1);
            a1=winAUC(M1,oi(ii),ri2(ii),tau(sel),dt,N,on(ii).');
            a2=winAUC(M2,oi(ii),ri2(ii),tau(sel),dt,N,on(ii).');
            d(bIt)=a2-a1;
        end
        ci=prctile(d,[2.5 97.5]);
        fprintf('    %-12s ms : M1 %.3f  M2 %.3f  |  diff %+.3f  CI [%+.3f %+.3f]  %s\n', ...
            wl{q,1},mean(A(1,sel)),mean(A(2,sel)),mean(d),ci(1),ci(2), ...
            ternary(ci(1)>0||ci(2)<0,'SIGNIFICANT','n.s.'));
    end

    % ================= (C) cross-correlation lag =================
    mv=tau>=-0.5 & tau<=0.2;
    x1=r1(mv)-mean(r1(mv)); x2=r2(mv)-mean(r2(mv));
    maxlag=round(0.3/dt); [cc,lg]=xcorr(x1,x2,maxlag,'coeff');
    [~,im]=max(cc); lagS=lg(im)*dt;
    fprintf('\n(C) CROSS-CORRELATION  peak r=%.2f at lag %+.0f ms\n',cc(im),1000*lagS);
    fprintf('    (positive lag = M2 leads M1)\n');

    % ================= figure =================
    f=figure('Color','w','Position',[60 60 1300 430]);
    tiledlayout(f,1,3,'Padding','compact','TileSpacing','compact');
    BL=[.20 .35 .60]; OR=[.85 .45 .10]; GN=[.13 .55 .29];
    ax=nexttile; hold(ax,'on'); plot(ax,1000*tau,n1,'-','Color',BL,'LineWidth',2.2);
    plot(ax,1000*tau,n2,'-','Color',OR,'LineWidth',2.2);
    xline(ax,0,'k-','movement','LineWidth',1.5); yline(ax,0.25,':','25%','Color',[.5 .5 .5]);
    xlabel(ax,'time from movement onset (ms)'); ylabel(ax,'population activity (norm. to own peak)');
    legend(ax,{'M1','M2'},'Location','northwest','Box','off'); grid(ax,'on');
    title(ax,sprintf('(A) ramp: M2 leads by %+.0f ms',1000*(t25a-t25b)),'FontWeight','bold');

    ax=nexttile; hold(ax,'on');
    plot(ax,1000*tau,A(1,:),'-','Color',BL,'LineWidth',2.2);
    plot(ax,1000*tau,A(2,:),'-','Color',OR,'LineWidth',2.2);
    plot(ax,1000*tau,A(3,:),'-','Color',GN,'LineWidth',1.8,'LineStyle','--');
    yline(ax,0.5,':','chance','Color',[.5 .5 .5]); xline(ax,0,'k-','LineWidth',1.5);
    xlabel(ax,'time from movement onset (ms)'); ylabel(ax,'decoding AUC (movement vs rest)');
    legend(ax,{'M1','M2','M1+M2'},'Location','northwest','Box','off'); grid(ax,'on');
    title(ax,sprintf('(B) decode latency  M1 %+.0f, M2 %+.0f ms',1000*LAT(1),1000*LAT(2)),'FontWeight','bold');

    ax=nexttile; plot(ax,1000*lg*dt,cc,'-','Color',[.35 .35 .45],'LineWidth',2); hold(ax,'on');
    xline(ax,1000*lagS,'-','Color',[.75 .2 .17],'LineWidth',2,'Label',sprintf('%+.0f ms',1000*lagS));
    xline(ax,0,'k:'); xlabel(ax,'lag (ms)   [+ = M2 leads M1]'); ylabel(ax,'correlation'); grid(ax,'on');
    title(ax,'(C) M1-M2 cross-correlation','FontWeight','bold');
    outDir=fullfile('C:\Users\tiwar133\gpfa_v0203\Data_Realtime','m1m2_audit');
    if ~exist(outDir,'dir'), mkdir(outDir); end
    exportgraphics(f,fullfile(outDir,sprintf('m1m2_timing_%s.png',tag)),'Resolution',200); close(f);
    fprintf('\nSAVED %s\n',fullfile(outDir,sprintf('m1m2_timing_%s.png',tag)));
end

function Z=zs(X), Z=(X-mean(X,2))./(std(X,0,2)+eps); end
function o=ternary(c,a,b), if c, o=a; else, o=b; end, end
function a=winAUC(D,oi,ri2,tw,dt,N,tp)
% mean activity over a pre-movement window -> single AUC (movement vs rest)
    sh=round(tw/dt); Xp=zeros(numel(oi),size(D,1)); Xn=Xp;
    for k=1:numel(oi)
        i1=oi(k)+sh; i2=ri2(k)+sh;
        i1=i1(i1>=1&i1<=N); i2=i2(i2>=1&i2<=N);
        if isempty(i1)||isempty(i2), Xp(k,:)=NaN; Xn(k,:)=NaN; continue; end
        Xp(k,:)=mean(D(:,i1),2).'; Xn(k,:)=mean(D(:,i2),2).';
    end
    ok=~any(isnan(Xp),2)&~any(isnan(Xn),2);
    a=cvAUC(Xp(ok,:),Xn(ok,:),tp(ok));
end
function X=grab(D,idx,N)
    X=nan(numel(idx),size(D,1)); ok=idx>=1&idx<=N;
    X(ok,:)=D(:,idx(ok)).';
end
function [t25,t50]=cross(tau,n)
    t25=firstX(tau,n,0.25); t50=firstX(tau,n,0.50);
end
function t=firstX(tau,n,lev)
    k=find(n>=lev & tau<0.15,1); if isempty(k), t=NaN; else, t=tau(k); end
end
function a=cvAUC(Xp,Xn,tp)
    X=[Xp;Xn]; y=[ones(size(Xp,1),1);zeros(size(Xn,1),1)]; t=[tp(:);tp(:)];
    [~,ord]=sort(t); fold=zeros(size(y)); n=numel(y); k=5; ed=round(linspace(0,n,k+1));
    for i=1:k, fold(ord(ed(i)+1:ed(i+1)))=i; end
    sc=nan(size(y));
    for i=1:k
        tr=fold~=i; te=fold==i; A=X(tr&y==1,:); Bb=X(tr&y==0,:);
        if size(A,1)<5||size(Bb,1)<5, continue; end
        m1=mean(A,1); m0=mean(Bb,1);
        Sw=((size(A,1)-1)*cov(A)+(size(Bb,1)-1)*cov(Bb))/(size(A,1)+size(Bb,1)-2);
        Sw=Sw+1e-2*mean(diag(Sw))*eye(size(Sw,1));
        w=Sw\(m1-m0)'; sc(te)=X(te,:)*w;
    end
    o=~isnan(sc); sp=sc(o&y==1); sn=sc(o&y==0);
    if isempty(sp)||isempty(sn), a=0.5; return; end
    v=[sp;sn]; l=[ones(numel(sp),1);zeros(numel(sn),1)];
    [~,q]=sort(v,'descend'); l=l(q); a=trapz(cumsum(1-l)/sum(1-l),cumsum(l)/sum(l));
end
