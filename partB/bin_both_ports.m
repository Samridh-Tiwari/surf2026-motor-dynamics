function bin_both_ports(sessionDir, nChanPort)
%BIN_BOTH_PORTS  Stream every .rhd in a session and bin BOTH probes through the
%   same causal front-end used online (CMR -> 300Hz HP -> adaptive MAD -> spikes
%   -> 20 ms bins), carrying filter/baseline state across file boundaries.
%   Port B = M1 (single shank), Port A = M2 (dual shank).
%   Saves <sessionDir>/binned_M1_M2.mat so later analyses never re-read raw data.
    if nargin<1, sessionDir='C:\Users\tiwar133\gpfa_v0203\Data_Realtime\MouseM1\Day2\eOPN_M1_DualShank_Day2_260722_163423'; end
    if nargin<2, nChanPort=64; end
    root='C:\Users\tiwar133\gpfa_v0203\Data_Realtime'; addpath(root);
    d=dir(fullfile(sessionDir,'*.rhd')); [~,ix]=sort({d.name}); d=d(ix);
    assert(~isempty(d),'no .rhd in %s',sessionDir);
    fprintf('BIN BOTH PORTS: %d files from %s\n',numel(d),sessionDir);

    binSec=0.02; fs=[]; f1=[]; f2=[]; binSamp=[];
    B1=cell(1,numel(d)); B2=cell(1,numel(d)); nb=zeros(1,numel(d));
    for k=1:numel(d)
        fp=fullfile(d(k).folder,d(k).name);
        [amp,~,fpar,~,~,ampCh]=read_rhd_programmatic(fp);
        if isempty(fs)
            fs=fpar.amplifier_sample_rate; binSamp=round(binSec*fs);
            f1=initFE(nChanPort,fs); f2=initFE(nChanPort,fs);
        end
        rows1=pickPort(ampCh,'B',nChanPort); rows2=pickPort(ampCh,'A',nChanPort);
        x1=double(amp(rows1,:)).'; x2=double(amp(rows2,:)).';   % samples x ch
        n=floor(size(x1,1)/binSamp); c1=zeros(nChanPort,n); c2=zeros(nChanPort,n);
        for b=1:n
            idx=(b-1)*binSamp+(1:binSamp);
            [f1,c1(:,b)]=feBlock(f1,x1(idx,:));
            [f2,c2(:,b)]=feBlock(f2,x2(idx,:));
        end
        B1{k}=c1; B2{k}=c2; nb(k)=n;
        fprintf('  file %2d/%d  %s  -> %d bins\n',k,numel(d),d(k).name,n);
        clear amp x1 x2
    end
    binM1=[B1{:}]; binM2=[B2{:}]; binT=(1:size(binM1,2))*binSec;
    out=fullfile(sessionDir,'binned_M1_M2.mat');
    save(out,'binM1','binM2','binT','binSec','fs','-v7.3');
    fprintf('SAVED %s  (M1 %dx%d, M2 %dx%d, %.1f min)\n',out,size(binM1,1),size(binM1,2),size(binM2,1),size(binM2,2),binT(end)/60);
end

function rows=pickPort(ampCh,letter,nCh)
    pn={ampCh.port_name}; hit=find(contains(pn,letter));
    if isempty(hit)
        pf={ampCh.port_prefix}; hit=find(strcmp(pf,letter));
    end
    [~,so]=sort([ampCh(hit).chip_channel]); rows=hit(so); rows=rows(1:min(nCh,numel(rows)));
end
function f=initFE(nCh,fs)
    f.fs=fs; f.filt=initCausalFilterState(nCh,fs,300,3);
    f.base=initAdaptiveBaselineState(fs,5,1,nCh,4); f.lockout=round(1e-3*fs);
end
function [f,cnt]=feBlock(f,xRaw)
    xRef=commonMedianRef(xRaw);
    [xF,f.filt]=applyCausalFilter(xRef,f.filt);
    f.base=updateAdaptiveBaseline(xF,f.base);
    [sp,f.base]=detectSpikesCausal(xF,f.base,f.lockout);
    cnt=sum(sp,1).';
end
