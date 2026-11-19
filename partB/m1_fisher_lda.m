function m1_fisher_lda(session)
%M1_FISHER_LDA  Fisher-LDA movement-intention detector on one session (no CUSUM).
%   Uses only online_binned_data.mat + behavioral_events.mat. Computes the lever
%   threshold + movement onsets from the lever trace, trains a ridge Fisher-LDA
%   (raw ch x 12 lags) with 5-fold time-blocked CV, and saves 3 figures:
%     m1_lda_separation.png    - score distributions + ROC (AUC)
%     m1_alarm_before_move.png - lead-time histogram (alarm vs pull onset)
%     m1_operating.png         - recall vs false-alarms/min operating curve
    if nargin<1, session=fullfile(dataRoot(), 'MouseM1', 'Day1', 'eOPN_M1_DualShank_Day1_260721_195215'); end
    rng(7); nLag=12;
    S =load(fullfile(session,'online_binned_data.mat'),'online_binned','binCenterTimes','binSizeSec','calibCutoffTime');
    ev=load(fullfile(session,'behavioral_events.mat'),'leverTrace','leverTimeVec','fs','cueTimes');
    X=double(S.online_binned); binT=S.binCenterTimes; dt=S.binSizeSec; N=size(X,2);
    lt=ev.leverTimeVec(:)'; lv=ev.leverTrace(:)'; fs=ev.fs; cue=ev.cueTimes(:)';
    calib=S.calibCutoffTime; if isempty(calib)||isnan(calib), calib=binT(1); end

    % lever threshold + movement onsets (from the lever trace only)
    pc=prctile(lv,[50 99]); rest=pc(1); thr=rest+0.33*(pc(2)-rest);
    mR=median(abs(lv-rest)); oM=rest+5*mR; ab=lv>thr; ri=find(diff(ab)>0)+1; onset=[]; last=-inf;
    for r=ri, tc=lt(r); if tc-last<1, continue; end
        i=r; lim=r-round(0.6*fs); while i>1&&i>lim&&lv(i)>oM, i=i-1; end; onset(end+1)=lt(i); last=tc; end %#ok<AGROW>
    fprintf('%s\n  %d cues | %d lever-pull onsets | thr %.3f V\n',session,numel(cue),numel(onset),thr);

    % features + labels
    Xz=(X-mean(X,2))./(std(X,0,2)+eps); dch=size(X,1); F=zeros(N,dch*nLag);
    for w=0:nLag-1, F(:,w*dch+(1:dch))=[zeros(w,dch); Xz(:,1:N-w)']; end
    live=binT>=calib;
    pos=false(1,N); for o=onset, pos=pos|(binT>=o-0.18&binT<=o); end
    nc=false(1,N);  for c=cue,   nc=nc|(abs(binT-c)<=2); end
    nm=false(1,N);  for o=onset, nm=nm|(abs(binT-o)<=1); end
    rest_m=~nc&~nm&live; pos=pos&live;
    if sum(pos)<10 || sum(rest_m)<40, error('too few labeled bins (pos=%d rest=%d)',sum(pos),sum(rest_m)); end

    % 5-fold time-blocked CV
    li=find(live); edges=linspace(binT(li(1)),binT(li(end))+1e-6,6); fold=discretize(binT,edges);
    score=nan(1,N);
    for k=1:5
        trP=find(pos&fold~=k); trR=find(rest_m&fold~=k);
        if numel(trR)>5*numel(trP), trR=trR(randperm(numel(trR),5*numel(trP))); end
        w=lda(F(trP,:),F(trR,:)); te=find(fold==k&live); score(te)=F(te,:)*w(1:end-1)+w(end);
    end
    rs=score(rest_m); mu=mean(rs,'omitnan'); sd=std(rs,'omitnan')+eps; score=(score-mu)/sd;
    sp=score(pos); sn=score(rest_m); sp=sp(~isnan(sp)); sn=sn(~isnan(sn));
    au=auc(sp,sn); fprintf('  Fisher-LDA CV AUC = %.3f\n',au);

    % ---- FIG 1: separation + ROC ----
    v=[sp(:);sn(:)]; lab=[ones(numel(sp),1);zeros(numel(sn),1)]; [~,o]=sort(v,'descend'); lab=lab(o);
    tpr=cumsum(lab)/sum(lab); fpr=cumsum(1-lab)/sum(1-lab);
    f=figure('Color','w','Position',[80 80 1000 420]); tiledlayout(f,1,2,'Padding','compact');
    nexttile; hold on; histogram(sn,60,'Normalization','pdf','FaceColor',[.4 .5 .6],'EdgeColor','none','FaceAlpha',.65);
    histogram(sp,40,'Normalization','pdf','FaceColor',[.85 .35 .1],'EdgeColor','none','FaceAlpha',.65);
    legend({'rest','pre-movement'},'Box','off'); xlabel('LDA score (SD above rest)'); ylabel('pdf'); grid on;
    title('M1 (Port B): LDA separates pre-movement from rest');
    nexttile; plot(fpr,tpr,'LineWidth',2.4,'Color',[.1 .55 .6]); hold on; plot([0 1],[0 1],'k:'); axis square; grid on;
    xlabel('false-positive rate'); ylabel('true-positive rate'); title(sprintf('ROC — AUC = %.3f',au));
    exportgraphics(f,fullfile(session,'m1_lda_separation.png'),'Resolution',200);

    % ---- alarms + operating point ----
    refrB=round(2/dt); dwell=2; restMin=sum(rest_m)*dt/60;
    ths=linspace(prctile(score(live),50),prctile(score(live),99.8),60); rec=zeros(size(ths)); fa=zeros(size(ths));
    for i=1:numel(ths)
        al=binAlarms(score,binT,ths(i),dwell,refrB,live);
        rec(i)=mean(arrayfun(@(o) any(al>=o-0.3&al<=o+0.1),onset));
        fa(i) =sum(arrayfun(@(a) ~any(abs(cue-a)<=2)&~any(abs(onset-a)<=1),al))/max(restMin,eps);
    end
    [~,j]=min(abs(fa-5)); alOp=binAlarms(score,binT,ths(j),dwell,refrB,live);

    % ---- FIG 2: alarm before movement ----
    lead=nan(1,numel(onset));
    for i=1:numel(onset), o=onset(i); c=alOp(alOp>=o-0.5&alOp<=o+0.15); if ~isempty(c), lead(i)=1000*(o-c(end)); end, end
    lead=lead(~isnan(lead)); pctB=100*mean(lead>0); med=median(lead);
    f=figure('Color','w','Position',[80 80 900 460]); ax=axes(f); hold(ax,'on'); ed=-200:20:400;
    histogram(ax,lead(lead>=0),ed(ed>=0),'FaceColor',[.13 .55 .29],'EdgeColor','w'); histogram(ax,lead(lead<0),ed(ed<=0),'FaceColor',[.75 .2 .17],'EdgeColor','w');
    xline(ax,0,'-','Color',[.15 .15 .15],'LineWidth',2.5,'Label','movement onset'); xline(ax,med,'--','Color',[.1 .4 .55],'LineWidth',2,'Label',sprintf('median %+.0f ms',med));
    text(ax,150,0.9*max(ylim(ax)),sprintf('%.0f%% of alarms fire BEFORE the pull\n(n=%d)',pctB,numel(lead)),'FontWeight','bold','Color',[.13 .45 .29],'VerticalAlignment','top');
    xlabel(ax,'onset - alarm (ms)   [positive = alarm precedes]'); ylabel('lever pulls'); grid on;
    title(ax,sprintf('M1 (Port B): alarm vs lever pull  (median %+.0f ms, %.0f%% precede)',med,pctB));
    exportgraphics(f,fullfile(session,'m1_alarm_before_move.png'),'Resolution',200);

    % ---- FIG 3: operating curve ----
    f=figure('Color','w','Position',[80 80 820 520]); ax=axes(f); hold(ax,'on');
    plot(ax,fa,100*rec,'-','LineWidth',2.6,'Color',[.1 .55 .6]); plot(ax,fa(j),100*rec(j),'o','MarkerSize',13,'MarkerFaceColor',[.85 .35 .1],'MarkerEdgeColor','k');
    text(ax,fa(j)+0.3,100*rec(j)+12,sprintf('%.0f%% recall @ %.1f FA/min',100*rec(j),fa(j)),'FontWeight','bold','Color',[.6 .25 .05]);
    xlabel(ax,'false alarms / min of rest'); ylabel(ax,'movement recall (%)'); grid on; xlim(ax,[0 10]); ylim(ax,[0 100]);
    title(ax,sprintf('M1 (Port B) Fisher-LDA  |  CV AUC = %.3f',au));
    exportgraphics(f,fullfile(session,'m1_operating.png'),'Resolution',200);
    fprintf('  SAVED m1_lda_separation / m1_alarm_before_move / m1_operating .png  | op %.0f%% @ %.1f FA/min\n',100*rec(j),fa(j));
end
function w=lda(Xp,Xn)
    m1=mean(Xp,1); m0=mean(Xn,1); Sw=((size(Xp,1)-1)*cov(Xp)+(size(Xn,1)-1)*cov(Xn))/(size(Xp,1)+size(Xn,1)-2);
    Sw=Sw+1e-2*mean(diag(Sw))*eye(size(Sw,1)); a=Sw\(m1-m0)'; b=-a'*((m1+m0)'/2); w=[a;b];
end
function a=auc(sp,sn)
    v=[sp(:);sn(:)]; lab=[ones(numel(sp),1);zeros(numel(sn),1)]; [~,o]=sort(v,'descend'); lab=lab(o); a=trapz(cumsum(1-lab)/sum(1-lab),cumsum(lab)/sum(lab));
end
function al=binAlarms(sc,binT,thv,dwell,refrB,live)
    pred=(sc>thv)&live; al=[]; run=0; last=-inf;
    for j=1:numel(sc), if pred(j),run=run+1; else,run=0; end
        if run>=dwell&&(j-last)>=refrB, al(end+1)=binT(j); last=j; end %#ok<AGROW>
    end
end
