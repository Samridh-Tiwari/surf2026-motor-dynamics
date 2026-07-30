function m1m2_cuefree_test
%M1M2_CUEFREE_TEST  Day1 shows a significant M2 pre-movement advantage, Day2
%   does not. Day1 has 12.2 cues/min vs Day2's 5.7 -- and M2's edge has tracked
%   the cue response in every earlier test. So: does Day1's M2 advantage survive
%   when we use ONLY pulls with no cue in the preceding 3 s?
%   If it vanishes -> the advantage is cue-evoked, and the days actually agree.
%   If it holds  -> M2 genuinely carries more movement-preparation signal.
    root='C:\Users\tiwar133\gpfa_v0203\Data_Realtime'; rng(7);
    S={'Day1',fullfile(root,'MouseM1\Day1\eOPN_M1_DualShank_Day1_260721_195215');
       'Day2',fullfile(root,'MouseM1\Day2\eOPN_M1_DualShank_Day2_260722_163423')};
    for s=1:2
        d=S{s,2};
        B=load(fullfile(d,'binned_M1_M2.mat'),'binM1','binM2','binT','binSec');
        ev=load(fullfile(d,'behavioral_events.mat'),'cueTimes');
        cue=double(ev.cueTimes(:)'); dt=B.binSec; bT=B.binT(:).'; N=numel(bT);
        M1=zs(double(B.binM1)); M2=zs(double(B.binM2));
        on=genuine_pulls(d,1.5); on=on(on>3 & on<bT(end)-2);
        cueFree=arrayfun(@(o) ~any(cue>o-3 & cue<=o+0.1), on);
        fprintf('\n########## %s : %d pulls (%d cue-preceded, %d CUE-FREE) ##########\n', ...
            S{s,1},numel(on),sum(~cueFree),sum(cueFree));
        subsets={'ALL pulls',true(size(on)); 'cue-preceded',~cueFree; 'CUE-FREE (>3s no cue)',cueFree};
        wl={'[-200,-100]',[-0.20 -0.10]; '[-100,0]',[-0.10 0]};
        for q=1:size(wl,1)
            fprintf('  window %s ms\n',wl{q,1});
            for u=1:3
                sel=subsets{u,2};
                if sum(sel)<40, fprintf('    %-24s n=%d  (too few)\n',subsets{u,1},sum(sel)); continue; end
                ons=on(sel);
                [a1,a2,ci]=cmp(M1,M2,ons,wl{q,2},dt,N,bT);
                fprintf('    %-24s n=%-4d M1 %.3f  M2 %.3f  diff %+.3f CI [%+.3f %+.3f]  %s\n', ...
                    subsets{u,1},numel(ons),a1,a2,a2-a1,ci(1),ci(2), ...
                    tern(ci(1)>0||ci(2)<0,'SIGNIFICANT','n.s.'));
            end
        end
    end
    fprintf('\nIf the M2 advantage is present for cue-preceded pulls but GONE for cue-free\n');
    fprintf('pulls, then M2''s edge is the cue-evoked response and the two days agree.\n');
end

function [a1,a2,ci]=cmp(M1,M2,on,win,dt,N,bT)
    oi=round(on/dt);
    nr=numel(on); rt=zeros(1,nr); c=0;
    while c<nr
        t=rand*(bT(end)-8)+4;
        if all(abs(on-t)>3), c=c+1; rt(c)=t; end
    end
    ri=round(rt/dt); sh=round(win(1)/dt):round(win(2)/dt);
    a1=wAUC(M1,oi,ri,sh,N,on); a2=wAUC(M2,oi,ri,sh,N,on);
    d=zeros(200,1);
    for b=1:200
        ii=randi(nr,nr,1);
        d(b)=wAUC(M2,oi(ii),ri(ii),sh,N,on(ii))-wAUC(M1,oi(ii),ri(ii),sh,N,on(ii));
    end
    ci=prctile(d,[2.5 97.5]);
end
function a=wAUC(D,oi,ri,sh,N,tp)
    Xp=zeros(numel(oi),size(D,1)); Xn=Xp;
    for k=1:numel(oi)
        i1=oi(k)+sh; i2=ri(k)+sh; i1=i1(i1>=1&i1<=N); i2=i2(i2>=1&i2<=N);
        if isempty(i1)||isempty(i2), Xp(k,:)=NaN; Xn(k,:)=NaN; continue; end
        Xp(k,:)=mean(D(:,i1),2).'; Xn(k,:)=mean(D(:,i2),2).';
    end
    ok=~any(isnan(Xp),2)&~any(isnan(Xn),2);
    a=cvAUC(Xp(ok,:),Xn(ok,:),tp(ok));
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
function Z=zs(X), Z=(X-mean(X,2))./(std(X,0,2)+eps); end
function o=tern(c,a,b), if c, o=a; else, o=b; end, end
