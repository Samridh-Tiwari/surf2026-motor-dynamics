function closedloop_live_gui(mode, arg)
%CLOSEDLOOP_LIVE_GUI  Live movement-intention detector + LED trigger GUI.
%
%   closedloop_live_gui            % open the GUI (default: RHD file mode)
%   closedloop_live_gui('selftest')% headless: verify live scorer == offline
%
%   Three data sources:
%     * RHD FILES : load raw .rhd files directly — full causal pipeline
%                   runs on the raw data live. No pre-processing needed.
%     * RHX LIVE  : tcpclient to Intan RHX waveform port; real-time stream.
%     * REPLAY    : streams a recorded session's online_binned_data.mat at
%                   realtime (requires pre-computed data from offline pipeline).
%
%   Pipeline (Fisher-LDA):
%     per 20 ms bin:  z=(count-chMean)./chStd ; feat=[z_t..z_{t-11}] ;
%                     score = w'*feat + b ; alarm if score>theta for `dwell`
%                     bins, then refractory. On alarm -> LED + lamp.

    root = fileparts(mfilename('fullpath'));
    if nargin<2, arg=''; end
    if nargin>=1 && strcmpi(mode,'selftest')
        selftest(root, ~isempty(arg), arg); return;
    end
    buildGUI(root);
end

%% ===================== shared scoring core =====================
function st = initLiveState(model, theta)
    st.ring = zeros(model.nChan, model.nLag);   % col 1 = newest z-scored bin
    st.run = 0; st.binIdx = 0; st.lastAlarmBin = -inf;
    st.theta = theta;
    st.refrBins = round(model.refractorySec / model.binSizeSec);
end

function [st, score, alarm, feat] = liveScoreStep(model, st, count64)
    z = (count64(:) - model.chMean) ./ model.chStd;
    st.ring = [z, st.ring(:, 1:end-1)];         % shift in newest bin
    feat = st.ring(:);                          % [z_t;z_{t-1};...;z_{t-11}]
    raw = model.w.' * feat + model.b;
    % standardise to the rest distribution so the threshold keeps a fixed
    % meaning ("SD above rest") even after online retraining rescales w.
    rm=0; rs=1;
    if isfield(model,'restMean'), rm=model.restMean; end
    if isfield(model,'restStd') && model.restStd>0, rs=model.restStd; end
    score = (raw - rm) / rs;
    st.binIdx = st.binIdx + 1;
    if score > st.theta, st.run = st.run + 1; else, st.run = 0; end
    alarm = st.run >= model.dwell && (st.binIdx - st.lastAlarmBin) >= st.refrBins;
    if alarm, st.lastAlarmBin = st.binIdx; end
end

%% ===================== headless verification =====================
function selftest(root, hasSession, session)
    if ~hasSession, session = '61198SOMDay2Recording1_241214_121351'; end
    m = load(fullfile(root,'live_model.mat'));
    S = load(fullfile(root,session,'online_binned_data.mat'),'online_binned','binCenterTimes');
    X = double(S.online_binned); binT = S.binCenterTimes; N = size(X,2);

    chMean=m.chMean; chStd=m.chStd; Xz=(X-chMean)./chStd;
    nLag=m.nLag; d=m.nChan; F=zeros(N,d*nLag);
    for w=0:nLag-1, F(:,w*d+(1:d))=[zeros(w,d); Xz(:,1:N-w)']; end
    raw=(F*m.w + m.b).'; rm=0; rs=1;
    if isfield(m,'restMean'), rm=m.restMean; end
    if isfield(m,'restStd')&&m.restStd>0, rs=m.restStd; end
    sc=(raw-rm)/rs; refrB=round(m.refractorySec/m.binSizeSec);
    refAl = binAlarmsRef(sc,binT,m.theta,m.dwell,refrB);

    st = initLiveState(m, m.theta); liveAl=[]; liveSc=zeros(1,N);
    for j=1:N
        [st,liveSc(j),al] = liveScoreStep(m, st, X(:,j));
        if al, liveAl(end+1)=binT(j); end %#ok<AGROW>
    end

    fprintf('SELFTEST on %s\n', session);
    fprintf('  max |liveScore - offlineScore| = %.3e  (should be ~0)\n', max(abs(liveSc-sc)));
    fprintf('  offline alarms %d | live alarms %d | identical times: %d\n', ...
        numel(refAl), numel(liveAl), isequal(round(refAl,6),round(liveAl,6)));
    fprintf('  theta %.3f | dwell %d | refractory %.1fs\n', m.theta, m.dwell, m.refractorySec);
    disp('SELFTEST DONE');
end
function al=binAlarmsRef(sc,binT,thv,dwell,refrB)
    pred=sc>thv; al=[]; run=0; last=-inf;
    for j=1:numel(sc)
        if pred(j),run=run+1; else,run=0; end
        if run>=dwell && (j-last)>=refrB; al(end+1)=binT(j); last=j; end %#ok<AGROW>
    end
end

%% ===================== GUI =====================
function buildGUI(root)
    % --- try to load existing model ---
    modelPath = fullfile(root,'live_model.mat');
    if isfile(modelPath)
        model = load(modelPath);
        hasModel = true;
    else
        % create placeholder model so GUI can build without crashing
        model = struct('w',zeros(768,1),'b',0,'chMean',zeros(64,1),'chStd',ones(64,1), ...
            'nLag',12,'nChan',64,'theta',2.0,'restMean',0,'restStd',1, ...
            'dwell',2,'refractorySec',2.0,'binSizeSec',0.02);
        hasModel = false;
    end
    sess = listSessions(root);

    fig = uifigure('Name','Closed-Loop Movement-Intention Detector','Position',[60 60 1260 780]);
    fig.CloseRequestFcn = @(o,e) onClose(fig);
    g = uigridlayout(fig,[3 1]); g.RowHeight={155,'1x',70}; g.Padding=[8 8 8 8];

    % ---- Setup panel (4 rows x 10 cols) ----
    cp = uipanel(g,'Title','Setup'); cp.Layout.Row=1;
    cg = uigridlayout(cp,[4 10]); cg.RowHeight={26,26,26,26};
    cg.ColumnWidth = {60,140,90,90,55,80,55,80,85,'1x'};

    % Row 1: Source + file selection + speed + start/stop
    uilabel(cg,'Text','Source:');
    ddSrc = uidropdown(cg,'Items',{'RHD files (raw)','RHX live (TCP)','Replay (recorded)'},...
        'Value','RHD files (raw)');
    btnBrowse = uibutton(cg,'Text','Browse Files','FontSize',11);
    btnFolder = uibutton(cg,'Text','Browse Folder','FontSize',11);
    ddSess = uidropdown(cg,'Items',sess,'Value',sess{1},'Visible','off');
    ddSess.Layout.Row=1; ddSess.Layout.Column=3;   % overlaps btnBrowse position
    uilabel(cg,'Text','Speed:');
    ddSpd = uidropdown(cg,'Items',{'realtime','2x','5x','max'});
    % columns 7-8 free
    uilabel(cg,'Text',''); uilabel(cg,'Text','');
    btnStart = uibutton(cg,'Text','START','BackgroundColor',[.2 .7 .3],'FontWeight','bold');
    btnStop = uibutton(cg,'Text','STOP','BackgroundColor',[.85 .3 .3],'FontWeight','bold','Enable','off');

    % Row 2: Port/channel config + detector params + model status
    uilabel(cg,'Text','Port:');
    ddPort = uidropdown(cg,'Items',{'B (auto)'},'Value','B (auto)');
    lblChan = uilabel(cg,'Text','64 ch @ 20 kHz');
    lblFiles = uilabel(cg,'Text','no files loaded');
    uilabel(cg,'Text','theta:');
    efTh = uieditfield(cg,'numeric','Value',model.theta,'Limits',[-50 50]);
    uilabel(cg,'Text','dwell:');
    efDwell = uieditfield(cg,'numeric','Value',model.dwell,'Limits',[1 10]);
    btnTrain = uibutton(cg,'Text','Train Model','FontWeight','bold','Enable','off');
    if hasModel
        lblModel = uilabel(cg,'Text','model: loaded','FontColor',[.1 .6 .2]);
    else
        lblModel = uilabel(cg,'Text','model: NONE','FontColor',[.8 .2 .1]);
    end

    % Row 3: Serial/LED + Adapt
    uilabel(cg,'Text','COM:');
    ddCom = uidropdown(cg,'Items',[{'(none)'} rowcell(serialportlist('available'))]);
    uilabel(cg,'Text','baud:');
    efBaud = uieditfield(cg,'numeric','Value',115200);
    btnCon = uibutton(cg,'Text','Connect LED');
    cbLED = uicheckbox(cg,'Text','LED out','Value',false);
    uilabel(cg,'Text','Adapt:');
    cbAdapt = uicheckbox(cg,'Text','online','Value',false);
    uilabel(cg,'Text','every s:');
    efUpd = uieditfield(cg,'numeric','Value',60,'Limits',[10 600]);

    % Row 4: fast-mode, refr, status
    uilabel(cg,'Text','refr(s):');
    efRefr = uieditfield(cg,'numeric','Value',model.refractorySec,'Limits',[0.5 10]);
    cbFast = uicheckbox(cg,'Text','Fast (no plots)','Value',false);
    lblProgress = uilabel(cg,'Text',''); lblProgress.Layout.Column=[4 10];

    % ---- plots + lamp ----
    mp = uigridlayout(g,[3 2]); mp.Layout.Row=2; mp.ColumnWidth={'3x','1x'}; mp.RowHeight={'1x','1x','1x'};
    ax1 = uiaxes(mp); ax1.Layout.Row=1; ax1.Layout.Column=1;
    title(ax1,'LDA score (movement-intention)'); ylabel(ax1,'score'); grid(ax1,'on');
    ax2 = uiaxes(mp); ax2.Layout.Row=2; ax2.Layout.Column=1;
    title(ax2,'population activity'); ylabel(ax2,'spikes/bin'); grid(ax2,'on');
    ax3 = uiaxes(mp); ax3.Layout.Row=3; ax3.Layout.Column=1;
    title(ax3,'lever (actual movement)'); ylabel(ax3,'lever (V)'); xlabel(ax3,'time (s)'); grid(ax3,'on');
    lp = uipanel(mp,'Title','LED / prediction'); lp.Layout.Row=[1 3]; lp.Layout.Column=2;
    lg = uigridlayout(lp,[3 1]); lg.RowHeight={'1x',40,'1x'};
    lamp = uilamp(lg,'Color',[.3 .3 .3]); lamp.Layout.Row=[1 3];
    lblPred = uilabel(lg,'Text','—','FontSize',20,'FontWeight','bold','HorizontalAlignment','center');

    % ---- status bar ----
    sp = uipanel(g,'Title','Status'); sp.Layout.Row=3;
    sg = uigridlayout(sp,[1 7]);
    lblElapsed=uilabel(sg,'Text','elapsed: 0.0 s');
    lblAlarms =uilabel(sg,'Text','alarms: 0');
    lblRate   =uilabel(sg,'Text','alarms/min: 0');
    lblScore  =uilabel(sg,'Text','score: -');
    lblLead   =uilabel(sg,'Text','last lead: —','FontWeight','bold','FontColor',[.1 .5 .55]);
    lblLED    =uilabel(sg,'Text','LED: not connected');
    lblSrc    =uilabel(sg,'Text','stopped');

    % ---- persistent handles/state ----
    h = struct('fig',fig,'ax1',ax1,'ax2',ax2,'ax3',ax3,'lamp',lamp,'lblPred',lblPred, ...
        'lblElapsed',lblElapsed,'lblAlarms',lblAlarms,'lblRate',lblRate,'lblScore',lblScore, ...
        'lblLead',lblLead,'lblLED',lblLED,'lblSrc',lblSrc, ...
        'efTh',efTh,'efDwell',efDwell,'efRefr',efRefr, ...
        'ddSrc',ddSrc,'ddSess',ddSess,'ddSpd',ddSpd,'ddPort',ddPort,'ddCom',ddCom, ...
        'efBaud',efBaud,'btnStart',btnStart,'btnStop',btnStop,'btnBrowse',btnBrowse, ...
        'btnFolder',btnFolder,'btnTrain',btnTrain,'btnCon',btnCon, ...
        'cbAdapt',cbAdapt,'cbLED',cbLED,'cbFast',cbFast,'efUpd',efUpd, ...
        'lblChan',lblChan,'lblFiles',lblFiles,'lblModel',lblModel,'lblProgress',lblProgress, ...
        'root',root,'model',model,'hasModel',hasModel,'modelPath',modelPath, ...
        'sp',[],'timer',[],'run',[], ...
        'rhdFiles',{{}},'rhdInfo',struct(),'rhdPortName','B');
    fig.UserData = h;

    % rolling plot lines
    h.lineScore = plot(ax1,nan,nan,'-','Color',[.1 .55 .6],'LineWidth',1.3); hold(ax1,'on');
    h.lineTheta = yline(ax1,efTh.Value,'--','Color',[.8 .2 .2]);
    h.lineAlarm = plot(ax1,nan,nan,'v','MarkerFaceColor',[.85 .1 .1],'MarkerEdgeColor','k','LineStyle','none');
    h.lineAct   = plot(ax2,nan,nan,'-','Color',[.35 .35 .4],'LineWidth',1);
    h.lineLever = plot(ax3,nan,nan,'-','Color',[.10 .10 .45],'LineWidth',1.1); hold(ax3,'on');
    h.lineLeverThr = yline(ax3,nan,':','pull thr','Color',[.6 .35 .1]);
    h.lineLeverAlarm = plot(ax3,nan,nan,'v','MarkerFaceColor',[.85 .1 .1],'MarkerEdgeColor','k','LineStyle','none','MarkerSize',5);
    h.lineCue = plot(ax3,nan,nan,'d','MarkerFaceColor',[.11 .63 .47],'MarkerEdgeColor','k','LineStyle','none','MarkerSize',6);
    h.lineRew = plot(ax3,nan,nan,'s','MarkerFaceColor',[.2 .5 .9],'MarkerEdgeColor','k','LineStyle','none','MarkerSize',6);
    fig.UserData = h;

    % callbacks
    ddSrc.ValueChangedFcn        = @(o,e) onSourceChanged(fig);
    btnBrowse.ButtonPushedFcn    = @(o,e) onBrowseFiles(fig);
    btnFolder.ButtonPushedFcn    = @(o,e) onBrowseFolder(fig);
    ddPort.ValueChangedFcn       = @(o,e) onPortChanged(fig);
    btnTrain.ButtonPushedFcn     = @(o,e) onTrainModel(fig);
    btnCon.ButtonPushedFcn       = @(o,e) onConnect(fig);
    btnStart.ButtonPushedFcn     = @(o,e) onStart(fig);
    btnStop.ButtonPushedFcn      = @(o,e) onStop(fig);
    efTh.ValueChangedFcn         = @(o,e) set(getf(fig).lineTheta,'Value',o.Value);
end

%% ---- Source mode switch ----
function onSourceChanged(fig)
    h=getf(fig); src=h.ddSrc.Value;
    isRHD    = startsWith(src,'RHD');
    isReplay = startsWith(src,'Replay');
    h.btnBrowse.Visible = ternary(isRHD,'on','off');
    h.btnFolder.Visible = ternary(isRHD,'on','off');
    h.ddSess.Visible    = ternary(isReplay,'on','off');
    h.ddPort.Enable     = ternary(isRHD,'on','off');
    h.btnTrain.Enable   = ternary(isRHD && ~isempty(h.rhdFiles) && ~h.hasModel,'on','off');
    fig.UserData = h;
end

%% ---- Browse for RHD files ----
function onBrowseFiles(fig)
    h=getf(fig);
    [fn,fp] = uigetfile({'*.rhd','Intan RHD (*.rhd)'}, ...
        'Select .rhd files — multi-select allowed', h.root, 'MultiSelect','on');
    if isequal(fn,0), return; end
    if ischar(fn), fn={fn}; end
    fn = sort(fn);
    h.rhdFiles = cellfun(@(n)fullfile(fp,n), fn, 'uni',0);
    h.lblFiles.Text = sprintf('%d file(s) from %s', numel(fn), fp);
    fig.UserData = h;
    probeRhdFiles(fig);
end

function onBrowseFolder(fig)
    h=getf(fig);
    fp = uigetdir(h.root, 'Select folder containing .rhd files');
    if isequal(fp,0), return; end
    d = dir(fullfile(fp,'*.rhd'));
    if isempty(d), uialert(fig,'No .rhd files found in this folder.','RHD'); return; end
    fn = sort({d.name});
    h.rhdFiles = cellfun(@(n)fullfile(fp,n), fn, 'uni',0);
    h.lblFiles.Text = sprintf('%d file(s) from %s', numel(fn), fp);
    fig.UserData = h;
    probeRhdFiles(fig);
end

function probeRhdFiles(fig)
    h=getf(fig);
    if isempty(h.rhdFiles), return; end
    h.lblProgress.Text = 'probing header...';  drawnow;
    try
        info = readRhdHeader(h.rhdFiles{1});
    catch err
        uialert(fig, sprintf('Failed to read RHD header:\n%s',err.message), 'RHD');
        h.lblProgress.Text = ''; fig.UserData=h; return;
    end
    h.rhdInfo = info;
    % build port list from amplifier channels
    if ~isempty(info.amplifier_channels)
        ports = {info.amplifier_channels.port_name};
        uports = unique(ports,'stable');
        counts = cellfun(@(p) sum(strcmp(ports,p)), uports);
        items = arrayfun(@(i) sprintf('%s (%d ch)', uports{i}, counts(i)), 1:numel(uports), 'uni',0);
        h.ddPort.Items = items;
        % auto-select Port B if present (our 64-ch M1 probe)
        bIdx = find(startsWith(uports,'Port B') | strcmp(uports,'B'), 1);
        if isempty(bIdx)
            % fallback: pick the port with 64 channels
            bIdx = find(counts==64, 1);
        end
        if isempty(bIdx), bIdx=1; end
        h.ddPort.Value = items{bIdx};
        h.rhdPortName = uports{bIdx};
        h.lblChan.Text = sprintf('%d ch @ %.0f kHz', counts(bIdx), info.sample_rate/1000);
    else
        h.ddPort.Items = {'(none)'}; h.rhdPortName='';
        h.lblChan.Text = 'no amplifier channels';
    end
    % enable Train if no model
    if ~h.hasModel
        h.btnTrain.Enable = 'on';
        h.lblModel.Text = 'model: NONE — click Train';
        h.lblModel.FontColor = [.8 .2 .1];
    end
    h.lblProgress.Text = sprintf('ready (%.1f s/file @ %.0f kHz)', ...
        info.record_time, info.sample_rate/1000);
    fig.UserData = h;
end

%% ---- Port changed ----
function onPortChanged(fig)
    h=getf(fig);
    if isempty(fieldnames(h.rhdInfo)), return; end
    sel = h.ddPort.Value;
    portName = regexp(sel, '^(\S+)', 'tokens','once');
    if isempty(portName), return; end
    portName = portName{1};
    % try to match "Port B" style or just "B"
    ports = {h.rhdInfo.amplifier_channels.port_name};
    uports = unique(ports,'stable');
    match = find(startsWith(uports, portName), 1);
    if isempty(match), match = 1; end
    h.rhdPortName = uports{match};
    nCh = sum(strcmp(ports, h.rhdPortName));
    h.lblChan.Text = sprintf('%d ch @ %.0f kHz', nCh, h.rhdInfo.sample_rate/1000);
    fig.UserData = h;
end

%% ---- Auto-train Fisher LDA from RHD files ----
function onTrainModel(fig)
    h=getf(fig);
    if isempty(h.rhdFiles)
        uialert(fig, 'Load RHD files first (Browse Files or Browse Folder).', 'Train');
        return;
    end
    h.lblProgress.Text = 'Training model from RHD files...'; drawnow;
    h.btnTrain.Enable = 'off'; drawnow;
    try
        nFiles = min(5, numel(h.rhdFiles));   % use first 5 files (~5 min)
        allBins = []; allLever = []; fs = 20000;
        portName = h.rhdPortName;
        nChan = 64;
        front = initFrontEnd(nChan, fs);
        binSamp = front.binSamp;

        for fi = 1:nFiles
            h.lblProgress.Text = sprintf('Training: reading file %d/%d...', fi, nFiles); drawnow;
            [ampData, adcData, ~, ampCh, ~, ~, filefs] = readRhdDirect(h.rhdFiles{fi});
            if fi==1, fs=filefs; front=initFrontEnd(nChan,fs); binSamp=front.binSamp; end

            % select Port B channels (64 ch, sorted by chip_channel)
            rows = find(strcmp({ampCh.port_name}, portName));
            [~,so] = sort([ampCh(rows).chip_channel]); rows=rows(so);
            if numel(rows) < nChan
                error('Port %s has only %d channels, need %d', portName, numel(rows), nChan);
            end
            rows = rows(1:nChan);
            raw = double(ampData(rows,:))';   % samples x nChan

            % lever from ADC (look for ANALOG-IN-2 or first ADC)
            lev = nan(1, size(raw,1));
            if ~isempty(adcData), lev = adcData(1,:); end

            % process through front-end in 20ms blocks
            nBins = floor(size(raw,1)/binSamp);
            bins = zeros(nChan, nBins);
            leverBin = nan(1, nBins);
            for b = 1:nBins
                seg = raw((b-1)*binSamp+1 : b*binSamp, :);
                [front, cnt] = frontEndBlock(front, seg);
                bins(:,b) = cnt;
                leverBin(b) = median(lev((b-1)*binSamp+1 : b*binSamp));
            end
            allBins = [allBins, bins]; %#ok<AGROW>
            allLever = [allLever, leverBin]; %#ok<AGROW>
        end

        h.lblProgress.Text = 'Training: fitting Fisher LDA...'; drawnow;
        dt = binSamp/fs;   % 0.02s
        N = size(allBins,2);
        binT = (1:N)*dt;

        % normalisation
        chMean = mean(allBins,2); chStd = std(allBins,0,2)+eps;
        Xz = (allBins - chMean) ./ chStd;

        % lag features
        nLag = 12;
        F = lagFeat(Xz, nLag);

        % auto-label from lever
        lv = allLever;
        rest = median(lv,'omitnan'); madR = median(abs(lv-rest),'omitnan');
        thr = rest + 0.33*(prctile(lv,99) - rest);
        if isnan(thr) || thr <= rest + eps
            uialert(fig, 'Cannot auto-detect lever pulls — lever trace is flat or missing.', 'Train');
            h.btnTrain.Enable='on'; h.lblProgress.Text='training failed'; fig.UserData=h; return;
        end

        % detect pull onsets (threshold crossings, 1s refractory)
        onsetM = rest + 5*madR;
        ab = lv > thr; ri = find(diff(ab)>0)+1;
        onset = []; last = -inf;
        for idx = 1:numel(ri)
            r = ri(idx); tc = binT(r);
            if tc - last < 1, continue; end
            i = r; lim = max(1, r - round(0.6/dt));
            while i > 1 && i > lim && lv(i) > onsetM, i = i-1; end
            onset(end+1) = binT(i); last = tc; %#ok<AGROW>
        end
        if numel(onset) < 5
            uialert(fig, sprintf('Only %d lever pulls found — need at least 5 to train.', numel(onset)), 'Train');
            h.btnTrain.Enable='on'; h.lblProgress.Text='training failed'; fig.UserData=h; return;
        end

        % label: positive = 180ms before onset, rest = away from pulls
        posPreSec = 0.18;
        pos = false(1,N);
        for o = onset, pos = pos | (binT >= o-posPreSec & binT <= o); end
        nm = false(1,N);
        for o = onset, nm = nm | (abs(binT-o) <= 1); end
        live = binT > 10*dt;   % skip first 10 bins (baseline warmup)
        rest_m = ~nm & live;
        pos = pos & live;

        % train LDA
        wfull = ldaFit(F(pos,:), F(rest_m,:));
        w = wfull(1:end-1); b = wfull(end);

        % pick theta at ~5 FA/min
        sc = (F*w + b).';
        dwell = 2; refractorySec = 2.0;
        refrB = round(refractorySec/dt); restMin = sum(rest_m)*dt/60;
        faTarget = 5.0;
        ths = linspace(prctile(sc(live),50), prctile(sc(live),99.9), 60);
        fa = zeros(size(ths));
        for i = 1:numel(ths)
            al = binAlarmsAll(sc, binT, ths(i), dwell, refrB, live);
            fa(i) = sum(arrayfun(@(a) ~any(abs(onset-a)<=1), al)) / max(restMin, 0.01);
        end
        [~,j] = min(abs(fa - faTarget)); thetaRaw = ths(j);

        % standardise
        restMean = mean(sc(rest_m)); restStd = std(sc(rest_m))+eps;
        theta = (thetaRaw - restMean) / restStd;

        % save model
        mdl = struct('w',w,'b',b,'chMean',chMean,'chStd',chStd,'nLag',nLag,'nChan',nChan, ...
            'theta',theta,'thetaRaw',thetaRaw,'restMean',restMean,'restStd',restStd, ...
            'dwell',dwell,'refractorySec',refractorySec,'binSizeSec',dt, ...
            'trainSession','auto-trained from RHD','faTargetPerMin',faTarget, ...
            'note','auto-trained by closedloop_live_gui');
        save(h.modelPath, '-struct', 'mdl');

        h.model = mdl; h.hasModel = true;
        h.efTh.Value = theta; set(h.lineTheta,'Value',theta);
        h.efDwell.Value = dwell; h.efRefr.Value = refractorySec;
        h.lblModel.Text = sprintf('model: trained (%d pulls, theta=%.1f)', numel(onset), theta);
        h.lblModel.FontColor = [.1 .6 .2];
        h.lblProgress.Text = sprintf('Model saved! %d ch x %d lags, %d pulls detected', nChan, nLag, numel(onset));
        fprintf('Auto-trained LDA: %d features (%d ch x %d lags), theta=%.2f SD, %d pulls\n', ...
            numel(w), nChan, nLag, theta, numel(onset));
    catch err
        uialert(fig, sprintf('Training failed:\n%s', err.message), 'Train');
        h.btnTrain.Enable = 'on';
        h.lblProgress.Text = 'training failed';
    end
    fig.UserData = h;
end

%% ---- Connect LED ----
function onConnect(fig)
    h=getf(fig); port=h.ddCom.Value;
    try
        if ~isempty(h.sp), clear h.sp; end
        if ~h.cbLED.Value
            h.sp=[]; h.lblLED.Text='LED: disabled (lamp only)';
        elseif strcmp(port,'(none)')
            h.sp=[]; h.lblLED.Text='LED: enabled but no COM port';
        else
            h.sp=serialport(port,h.efBaud.Value); configureTerminator(h.sp,"LF");
            h.lblLED.Text=sprintf('LED: %s @ %d',port,h.efBaud.Value);
        end
    catch err
        uialert(fig,sprintf('Serial connect failed:\n%s',err.message),'LED');
        h.sp=[]; h.lblLED.Text='LED: FAILED';
    end
    fig.UserData=h;
end

%% ---- START ----
function onStart(fig)
    h=getf(fig);
    if ~h.hasModel
        uialert(fig, 'No model loaded. Click "Train Model" first, or place a live_model.mat next to this file.', 'Model');
        return;
    end
    r = struct();
    r.M = h.model;
    if ~isfield(r.M,'restMean'), r.M.restMean=0; r.M.restStd=1; end
    r.M.dwell = max(1, round(h.efDwell.Value));
    r.M.refractorySec = max(0.5, h.efRefr.Value);
    r.dt = r.M.binSizeSec;
    r.st = initLiveState(r.M, h.efTh.Value);

    % online-adaptation state
    r.adapt = h.cbAdapt.Value; r.updateSec = max(10, h.efUpd.Value);
    r.lastUpdT = 0; r.nAdapt = 0;
    nf = numel(r.M.w);
    r.Fpos = zeros(0,nf); r.Frest = zeros(0,nf); r.fbuf = nan(nf,60); r.fbk = 0;

    r.nAlarm = 0; r.tic = tic; r.processed = 0;
    r.win = round(12/r.dt);   % 12 s rolling window
    r.tbuf = nan(1,r.win); r.sbuf = nan(1,r.win); r.abuf = [];
    r.actbuf = nan(1,r.win); r.k = 0;
    r.lvbuf = nan(1,r.win); r.leverBin = []; r.leverThr = NaN; r.lampUntil = 0;
    r.leverPrev = NaN; r.lastPullT = -inf; r.lastLeadMs = NaN;
    r.xlh = gobjects(0); r.visKey = [-1 -1];
    r.cueT = []; r.rewT = []; r.cueWin = 2; r.nCued = 0; r.nSelf = 0; r.nNoPull = 0;
    r.pend = zeros(0,1); r.lastType = '';

    switch h.ddSpd.Value
        case '2x', r.speed=2; case '5x', r.speed=5; case 'max', r.speed=Inf;
        otherwise, r.speed=1;
    end

    r.source = h.ddSrc.Value;

    if startsWith(r.source, 'RHD')
        % ---- RHD files: direct raw processing ----
        if isempty(h.rhdFiles)
            uialert(fig,'No RHD files loaded. Click Browse Files or Browse Folder first.','RHD');
            return;
        end
        r.rhdFiles = h.rhdFiles;
        r.rhdIdx = 0; r.rawAll = []; r.leverAll = []; r.rawPos = 1;
        r.portName = h.rhdPortName;
        r.X = []; r.N = inf; r.rhx = [];
        r.front = initFrontEnd(r.M.nChan, h.rhdInfo.sample_rate);
        h.lblSrc.Text = sprintf('RHD (%d files)', numel(r.rhdFiles));

    elseif startsWith(r.source, 'Replay')
        % ---- Replay: pre-binned data ----
        sdir = fullfile(h.root, h.ddSess.Value);
        S = load(fullfile(sdir,'online_binned_data.mat'),'online_binned','binCenterTimes');
        r.X = double(S.online_binned); r.N = size(r.X,2); r.rhx = [];
        try
            E = load(fullfile(sdir,'behavioral_events.mat'),'leverTrace','leverTimeVec','cueTimes');
            r.leverBin = interp1(E.leverTimeVec(:)',E.leverTrace(:)',S.binCenterTimes,'linear','extrap');
            r.cueT = E.cueTimes(:)';
            v = load(fullfile(sdir,'lever_validation_results.mat'),'thr');
            r.leverThr = v.thr; set(h.lineLeverThr,'Value',r.leverThr);
        catch, end
        h.lblSrc.Text = sprintf('REPLAY %s', h.ddSess.Value);

    else
        % ---- RHX live (TCP) ----
        try
            r.rhx = rhxConnect(); r.rhxState = struct('buf',uint8([]));
        catch err
            uialert(fig,sprintf('RHX connect failed:\n%s',err.message),'RHX');
            return;
        end
        r.front = initFrontEnd(r.M.nChan, 20000);
        r.X = []; r.N = inf;
        h.lblSrc.Text = 'RHX LIVE';
        try
            v = load(fullfile(h.root,h.model.trainSession,'lever_validation_results.mat'),'thr');
            r.leverThr = v.thr; set(h.lineLeverThr,'Value',r.leverThr);
        catch, end
    end

    r.srcBase = h.lblSrc.Text;
    h.run = r; h.btnStart.Enable = 'off'; h.btnStop.Enable = 'on';
    h.timer = timer('ExecutionMode','fixedRate','Period',0.02,'BusyMode','drop', ...
        'TimerFcn',@(o,e) onTickSafe(fig));
    fig.UserData = h;
    start(h.timer);
end

%% ---- STOP / CLOSE ----
function onStop(fig)
    h=getf(fig);
    if ~isempty(h.timer) && isvalid(h.timer), stop(h.timer); delete(h.timer); end
    h.timer=[]; h.btnStart.Enable='on'; h.btnStop.Enable='off'; h.lblSrc.Text='stopped';
    fig.UserData=h;
end

function onClose(fig)
    try onStop(fig); h=getf(fig); if ~isempty(h.sp), clear h.sp; end, catch, end
    delete(fig);
end

%% ---- Timer: safe wrapper ----
function onTickSafe(fig)
    try
        if ~isvalid(fig), return; end
        onTick(fig);
    catch err
        try
            h=getf(fig);
            h.lblProgress.Text = sprintf('ERROR: %s', err.message);
            fig.UserData = h;
            disp(err.message);
            for ie=1:numel(err.stack), disp(err.stack(ie)); end
        catch, end
    end
end

%% ---- Timer: main tick ----
function onTick(fig)
    h=getf(fig); r=h.run; if isempty(r), return; end

    % ---- pull the bins that should have arrived ----
    if startsWith(r.source, 'RHD')
        % --- RHD files: stream raw data through front-end ---
        newBins = zeros(r.M.nChan, 0); newLever = []; done = false;
        binSamp = r.front.binSamp;

        % load next file if needed
        if isempty(r.rawAll) || r.rawPos + binSamp - 1 > size(r.rawAll, 1)
            r.rhdIdx = r.rhdIdx + 1;
            if r.rhdIdx > numel(r.rhdFiles)
                done = true;
            else
                h.lblSrc.Text = sprintf('%s | file %d/%d', r.srcBase, r.rhdIdx, numel(r.rhdFiles));
                drawnow limitrate;
                [ampData, adcData, digInData, ampCh, ~, digInCh, filefs] = readRhdDirect(r.rhdFiles{r.rhdIdx});

                % select Port B channels (sorted by chip_channel)
                rows = find(strcmp({ampCh.port_name}, r.portName));
                [~,so] = sort([ampCh(rows).chip_channel]); rows = rows(so);
                rows = rows(1:min(r.M.nChan, numel(rows)));
                r.rawAll = double(ampData(rows,:))';   % samples x nChan

                lr = nan(1, size(ampData,2));
                if ~isempty(adcData), lr = adcData(1,:); end
                r.leverAll = lr(:); r.rawPos = 1;

                % auto-detect lever threshold from first file
                if isnan(r.leverThr) && any(~isnan(lr))
                    rr = median(lr,'omitnan');
                    r.leverThr = rr + 0.33*(prctile(lr,99) - rr);
                    set(h.lineLeverThr, 'Value', r.leverThr);
                end
                
                % find cue/reward rises in this file
                if ~isempty(digInData)
                    names = {digInCh.native_channel_name};
                    rowCue = 0; rowRew = 0;
                    for i=1:numel(names)
                        if endsWith(names{i}, '4') || endsWith(names{i}, '04')
                            rowCue = digInCh(i).native_order;
                        elseif endsWith(names{i}, '3') || endsWith(names{i}, '03')
                            rowRew = digInCh(i).native_order;
                        end
                    end
                    baseT = r.k * r.dt;
                    if rowCue > 0
                        cueAll = bitget(digInData, rowCue+1);
                        rises = find(diff(cueAll) > 0) + 1;
                        if ~isempty(rises)
                            risesT = baseT + (rises - 1) / r.front.fs;
                            r.cueT = [r.cueT, risesT(:)'];
                        end
                    end
                    if rowRew > 0
                        rewAll = bitget(digInData, rowRew+1);
                        rises = find(diff(rewAll) > 0) + 1;
                        if ~isempty(rises)
                            risesT = baseT + (rises - 1) / r.front.fs;
                            r.rewT = [r.rewT, risesT(:)'];
                        end
                    end
                end
            end
        end

        if ~done && r.rawPos + binSamp - 1 <= size(r.rawAll, 1)
            if isfinite(r.speed)
                target = floor(toc(r.tic)*r.speed / r.dt);
            else
                target = r.k + 50;
            end
            avail = floor((size(r.rawAll,1) - r.rawPos + 1) / binSamp);
            nB = max(0, min(target - r.k, avail));
            newBins = zeros(r.M.nChan, nB); newLever = nan(1, nB);
            for b = 1:nB
                seg = r.rawAll(r.rawPos:r.rawPos+binSamp-1, :);
                [r.front, cnt] = frontEndBlock(r.front, seg);
                newBins(:,b) = cnt;
                newLever(b) = median(r.leverAll(r.rawPos:r.rawPos+binSamp-1));
                r.rawPos = r.rawPos + binSamp;
            end
        end

    elseif startsWith(r.source, 'Replay')
        if isfinite(r.speed)
            target = min(r.N, floor(toc(r.tic)*r.speed / r.dt));
        else
            target = min(r.N, r.processed + 50);
        end
        cols = r.processed+1:target; newBins = r.X(:,cols); r.processed = target;
        newLever = nan(1, numel(cols));
        if ~isempty(r.leverBin)
            cc = cols(cols <= numel(r.leverBin));
            newLever(1:numel(cc)) = r.leverBin(cc);
        end
        done = r.processed >= r.N;

    else
        % RHX live
        [blocks, leverBlk, r.rhxState] = rhxReadBlocks(r.rhx, r.rhxState);
        nB = size(blocks,3); newBins = zeros(r.M.nChan, nB);
        for bb=1:nB
            [r.front, cnt] = frontEndBlock(r.front, blocks(:,:,bb));
            newBins(:,bb) = cnt;
        end
        newLever = leverBlk; done = false;
    end

    % ---- score each new bin ----
    for j = 1:size(newBins,2)
        [r.st, sc, alarm, feat] = liveScoreStep(r.M, r.st, newBins(:,j));
        r.k = r.k + 1; idx = mod(r.k-1, r.win) + 1;
        r.tbuf(idx) = r.k * r.dt; r.sbuf(idx) = sc;
        r.actbuf(idx) = sum(newBins(:,j));
        if j <= numel(newLever), lvNow = newLever(j); else, lvNow = NaN; end
        r.lvbuf(idx) = lvNow;
        tNow = r.k * r.dt;

        % rolling feature buffer for labeling
        r.fbk = r.fbk + 1; r.fbuf(:, mod(r.fbk-1,60)+1) = feat;

        % detect real lever pull (rising threshold cross)
        if ~isnan(r.leverThr) && ~isnan(lvNow) && ~isnan(r.leverPrev) ...
                && lvNow > r.leverThr && r.leverPrev <= r.leverThr && (tNow-r.lastPullT) > 1
            r.lastPullT = tNow;
            if ~isempty(r.abuf)
                pa = r.abuf(r.abuf(:,1) >= tNow-1 & r.abuf(:,1) <= tNow, 1);
                if ~isempty(pa)
                    r.lastLeadMs = 1000*(tNow - pa(end));
                    isCued = ~isempty(r.cueT) && any(r.cueT >= tNow-r.cueWin & r.cueT <= tNow);
                    if isCued, r.nCued=r.nCued+1; r.lastType='CUED';
                    else, r.nSelf=r.nSelf+1; r.lastType='SELF'; end
                    r.pend(r.pend >= tNow-1 & r.pend <= tNow) = [];
                end
            end
            if r.adapt
                np = min(10, r.fbk); ii = mod((r.fbk-np):(r.fbk-1),60)+1;
                r.Fpos = [r.Fpos; r.fbuf(:,ii)']; %#ok<AGROW>
            end
        end
        % accumulate REST examples
        if r.adapt && ~isnan(lvNow) && lvNow < r.leverThr && (tNow-r.lastPullT) > 1 && rand < 0.12
            r.Frest = [r.Frest; feat']; %#ok<AGROW>
            if size(r.Frest,1) > 4000, r.Frest = r.Frest(end-3000:end,:); end
        end
        if ~isnan(lvNow), r.leverPrev = lvNow; end
        if alarm
            r.nAlarm = r.nAlarm + 1;
            r.abuf(end+1,:) = [tNow sc lvNow]; %#ok<AGROW>
            if size(r.abuf,1) > 200, r.abuf = r.abuf(end-100:end,:); end
            r.pend(end+1,1) = tNow;
            r.lampUntil = tNow + 0.15;
            if h.cbLED.Value && ~isempty(h.sp)
                try write(h.sp, uint8(1), "uint8"); catch, end
            end
        end
    end

    r.st.theta = h.efTh.Value;
    r.M.dwell = max(1, round(h.efDwell.Value));

    % online adaptation
    if r.adapt && (r.k*r.dt - r.lastUpdT) >= r.updateSec ...
            && size(r.Fpos,1) >= 20 && size(r.Frest,1) >= 60
        w = ldaFit(r.Fpos, r.Frest);
        r.M.w = w(1:end-1); r.M.b = w(end);
        raw = r.Frest * r.M.w + r.M.b;
        r.M.restMean = mean(raw); r.M.restStd = std(raw)+eps;
        r.lastUpdT = r.k*r.dt; r.nAdapt = r.nAdapt + 1;
        h.lblSrc.Text = sprintf('%s | adapted x%d (%dP/%dR)', ...
            r.srcBase, r.nAdapt, size(r.Fpos,1), size(r.Frest,1));
    end

    % pending alarms that never matched a pull
    if ~isempty(r.pend)
        exp = r.pend < (r.k*r.dt - 0.6);
        r.nNoPull = r.nNoPull + sum(exp); r.pend(exp) = [];
    end

    nowT = r.k * r.dt;

    % ---- redraw (skip in Fast mode) ----
    if ~h.cbFast.Value
        [tt,oo] = sort(r.tbuf); ss=r.sbuf(oo); aa=r.actbuf(oo); ll=r.lvbuf(oo); m=~isnan(tt);
        set(h.lineScore,'XData',tt(m),'YData',ss(m));
        set(h.lineAct,'XData',tt(m),'YData',aa(m));
        set(h.lineLever,'XData',tt(m),'YData',ll(m));
        if ~isempty(r.abuf)
            vis = r.abuf(r.abuf(:,1) > nowT-12, :);
            set(h.lineAlarm,'XData',vis(:,1),'YData',vis(:,2));
            set(h.lineLeverAlarm,'XData',vis(:,1),'YData',vis(:,3));
        else
            vis = zeros(0,3);
            set(h.lineAlarm,'XData',nan,'YData',nan);
            set(h.lineLeverAlarm,'XData',nan,'YData',nan);
        end
        vk = [size(vis,1), (size(vis,1)>0)*max([vis(:,1);0])];
        if ~isequal(vk, r.visKey)
            delete(r.xlh(isgraphics(r.xlh))); r.xlh = gobjects(0);
            for tv = vis(:,1)'
                for ax = [h.ax1 h.ax2 h.ax3]
                    r.xlh(end+1) = xline(ax,tv,'Color',[.85 .1 .1],'Alpha',0.35,'LineWidth',1); %#ok<AGROW>
                end
            end
            r.visKey = vk;
        end
        if ~isempty(r.cueT) && ~isnan(r.leverThr)
            cv = r.cueT(r.cueT > nowT-12 & r.cueT <= nowT);
            set(h.lineCue,'XData',cv,'YData',repmat(r.leverThr,size(cv)));
        end
        if ~isempty(r.rewT) && ~isnan(r.leverThr)
            rv = r.rewT(r.rewT > nowT-12 & r.rewT <= nowT);
            set(h.lineRew,'XData',rv,'YData',repmat(r.leverThr,size(rv)));
        end
        if nowT > 12
            xlim(h.ax1,[nowT-12 nowT]); xlim(h.ax2,[nowT-12 nowT]); xlim(h.ax3,[nowT-12 nowT]);
        end
    end

    on = nowT < r.lampUntil;
    h.lamp.Color     = ternary(on,[.2 .85 .2],[.3 .3 .3]);
    h.lblPred.Text   = ternary(on,'PREDICTED PULL','—');
    h.lblPred.FontColor = ternary(on,[.1 .6 .1],[.4 .4 .4]);
    h.lblElapsed.Text = sprintf('elapsed: %.1f s', nowT);
    h.lblAlarms.Text  = sprintf('alarms: %d', r.nAlarm);
    h.lblRate.Text    = sprintf('alarms/min: %.1f', 60*r.nAlarm/max(nowT,1e-6));
    if isnan(r.lastLeadMs), ltxt='last lead: —';
    else, ltxt = sprintf('last lead: %.0f ms [%s]', r.lastLeadMs, r.lastType); end
    h.lblLead.Text = sprintf('%s | TP cued %d self %d | alarm-noPull %d', ltxt, r.nCued, r.nSelf, r.nNoPull);
    if nowT < 5.0
        h.lblScore.Text = 'WARMUP (adaptive baseline)';
    elseif r.k > 0
        h.lblScore.Text = sprintf('score: %.2f (theta %.2f)', r.sbuf(mod(r.k-1,r.win)+1), h.efTh.Value);
    end

    h.run = r; fig.UserData = h;
    if done
        onStop(fig); h = getf(fig);
        h.lblSrc.Text = 'finished'; fig.UserData = h;
    end
end

%% ===================== built-in RHD reader =====================
function info = readRhdHeader(filepath)
%READRHDHEADER  Read only the header of an Intan RHD file (fast, ~1 ms).
%   Returns a struct with channel metadata, sample rate, and record time.
    fid = fopen(filepath, 'r');
    if fid < 0, error('Cannot open %s', filepath); end
    cleanObj = onCleanup(@() fclose(fid));

    magic = fread(fid, 1, 'uint32');
    if magic ~= hex2dec('c6912702'), error('Not a valid RHD file (bad magic number).'); end

    ver_main = fread(fid, 1, 'int16');
    ver_sec  = fread(fid, 1, 'int16');
    if ver_main == 1, spb = 60; else, spb = 128; end   % samples per data block

    sample_rate = fread(fid, 1, 'single');
    fread(fid, 1, 'int16');   % dsp_enabled
    fread(fid, 3, 'single');  % actual_dsp_cutoff, actual_lower_bw, actual_upper_bw
    fread(fid, 3, 'single');  % desired_dsp_cutoff, desired_lower_bw, desired_upper_bw
    fread(fid, 1, 'int16');   % notch_filter_mode
    fread(fid, 2, 'single');  % desired/actual impedance test freq
    fread_QString(fid); fread_QString(fid); fread_QString(fid);  % notes

    num_temp = 0;
    if (ver_main==1 && ver_sec>=1) || ver_main>1, num_temp=fread(fid,1,'int16'); end
    board_mode = 0;
    if (ver_main==1 && ver_sec>=3) || ver_main>1, board_mode=fread(fid,1,'int16'); end
    if ver_main > 1, fread_QString(fid); end  % reference_channel

    % signal groups -> channel arrays
    amp_ch = []; adc_ch = []; dig_in_ch = []; nAux=0; nSupply=0; nDigIn=0; nDigOut=0;
    nGroups = fread(fid, 1, 'int16');
    for sg = 1:nGroups
        gName   = fread_QString(fid);
        gPrefix = fread_QString(fid);
        gEnabled = fread(fid, 1, 'int16');
        gNumCh   = fread(fid, 1, 'int16');
        gNumAmp  = fread(fid, 1, 'int16'); %#ok<NASGU>
        if gNumCh > 0 && gEnabled > 0
            for sc = 1:gNumCh
                ch.native_channel_name = fread_QString(fid);
                ch.custom_channel_name = fread_QString(fid);
                ch.native_order  = fread(fid,1,'int16');
                ch.custom_order  = fread(fid,1,'int16');
                signal_type      = fread(fid,1,'int16');
                channel_enabled  = fread(fid,1,'int16');
                ch.chip_channel  = fread(fid,1,'int16');
                ch.board_stream  = fread(fid,1,'int16');
                fread(fid, 4, 'int16');    % spike trigger fields
                fread(fid, 2, 'single');   % impedance mag + phase
                ch.port_name   = gName;
                ch.port_prefix = gPrefix;
                if channel_enabled
                    switch signal_type
                        case 0, amp_ch = [amp_ch, ch]; %#ok<AGROW>
                        case 1, nAux = nAux+1;
                        case 2, nSupply = nSupply+1;
                        case 3, adc_ch = [adc_ch, ch]; %#ok<AGROW>
                        case 4, dig_in_ch = [dig_in_ch, ch]; nDigIn = nDigIn+1; %#ok<AGROW>
                        case 5, nDigOut = nDigOut+1;
                    end
                end
            end
        end
    end

    header_bytes = ftell(fid);
    s = dir(filepath); filesize = s.bytes;
    nAmp = numel(amp_ch); nAdc = numel(adc_ch);

    % bytes per data block
    bpb = spb*4;                           % timestamps
    bpb = bpb + spb*2*nAmp;               % amplifier
    bpb = bpb + (spb/4)*2*nAux;           % aux
    bpb = bpb + 1*2*nSupply;              % supply
    if num_temp > 0, bpb = bpb + 1*2*num_temp; end
    bpb = bpb + spb*2*nAdc;               % board ADC
    if nDigIn > 0,  bpb = bpb + spb*2; end
    if nDigOut > 0, bpb = bpb + spb*2; end

    data_bytes = filesize - header_bytes;
    nBlocks = floor(data_bytes / bpb);
    nSamples = nBlocks * spb;
    record_time = nSamples / sample_rate;

    info = struct('sample_rate',sample_rate, 'samples_per_block',spb, ...
        'amplifier_channels',amp_ch, 'board_adc_channels',adc_ch, 'board_dig_in_channels',dig_in_ch, ...
        'num_amp',nAmp, 'num_adc',nAdc, 'num_aux',nAux, ...
        'num_supply',nSupply, 'num_temp',num_temp, ...
        'num_dig_in',nDigIn, 'num_dig_out',nDigOut, ...
        'board_mode',board_mode, 'ver_main',ver_main, 'ver_sec',ver_sec, ...
        'header_bytes',header_bytes, 'bytes_per_block',bpb, ...
        'num_blocks',nBlocks, 'num_samples',nSamples, 'record_time',record_time);
end

function [ampData, adcData, digInData, ampCh, adcCh, digInCh, fs] = readRhdDirect(filepath)
%READRHDDIRECT  Read an Intan RHD file directly. Returns amplifier data
%   (in uV), ADC data (in V), digital input data (uint16 raw), channel metadata, and sample rate.
%   No evalin, no base workspace, no dialogs.
    info = readRhdHeader(filepath);
    fs = info.sample_rate;
    ampCh = info.amplifier_channels;
    adcCh = info.board_adc_channels;
    digInCh = info.board_dig_in_channels;
    nAmp = info.num_amp; nAdc = info.num_adc;
    nAux = info.num_aux; nSupply = info.num_supply;
    nTemp = info.num_temp; nDigIn = info.num_dig_in; nDigOut = info.num_dig_out;
    spb = info.samples_per_block; nBlocks = info.num_blocks;
    nSamp = info.num_samples;
    bm = info.board_mode;
    signedTS = (info.ver_main==1 && info.ver_sec>=2) || info.ver_main>1;

    fid = fopen(filepath, 'r');
    cleanObj = onCleanup(@() fclose(fid));
    fseek(fid, info.header_bytes, 'bof');

    % pre-allocate
    ampData = zeros(nAmp, nSamp, 'uint16');
    adcData = zeros(nAdc, nSamp, 'uint16');
    digInData = zeros(1, nSamp, 'uint16');
    ai = 1; di = 1;

    for blk = 1:nBlocks
        % timestamps (skip)
        if signedTS, fread(fid, spb, 'int32'); else, fread(fid, spb, 'uint32'); end
        % amplifier
        if nAmp > 0
            ampData(:, ai:ai+spb-1) = fread(fid, [spb, nAmp], 'uint16')';
        end
        % aux input (skip)
        if nAux > 0, fread(fid, (spb/4)*nAux, 'uint16'); end
        % supply voltage (skip)
        if nSupply > 0, fread(fid, nSupply, 'uint16'); end
        % temperature (skip)
        if nTemp > 0, fread(fid, nTemp, 'int16'); end
        % board ADC
        if nAdc > 0
            adcData(:, di:di+spb-1) = fread(fid, [spb, nAdc], 'uint16')';
        end
        % digital in/out (skip)
        if nDigIn > 0,  digInData(1, di:di+spb-1) = fread(fid, spb, 'uint16')'; end
        if nDigOut > 0, fread(fid, spb, 'uint16'); end
        ai = ai + spb; di = di + spb;
    end

    % scale to physical units
    ampData = 0.195 * (double(ampData) - 32768);   % uV
    if nAdc > 0
        if bm == 1
            adcData = 152.59e-6 * (double(adcData) - 32768);   % V
        elseif bm == 13
            adcData = 312.5e-6 * (double(adcData) - 32768);    % V
        else
            adcData = 50.354e-6 * double(adcData);              % V (unipolar)
        end
    else
        adcData = [];
    end
end

function a = fread_QString(fid)
%FREAD_QSTRING  Read a Qt-style QString from a binary file.
    a = '';
    len = fread(fid, 1, 'uint32');
    if isempty(len) || len >= 4294967295, return; end   % null string
    nChars = len / 2;
    if nChars > 0
        raw = fread(fid, nChars, 'uint16');
        a = char(raw(:)');
    end
end

%% ===================== causal front-end =====================
function f = initFrontEnd(nChan, fs)
    f.fs = fs; f.blk = round(0.02*fs); f.binSamp = round(0.02*fs);
    f.filt = initCausalFilterState(nChan, fs, 300, 3);
    f.base = initAdaptiveBaselineState(fs, 5, 1, nChan, 4);
    f.lockout = round(1e-3*fs);
    f.acc = zeros(nChan,1); f.into = 0;
end

function [f, count] = frontEndBlock(f, xRaw)
    xRef = commonMedianRef(xRaw);
    [xF, f.filt] = applyCausalFilter(xRef, f.filt);
    f.base = updateAdaptiveBaseline(xF, f.base);
    [sp, f.base] = detectSpikesCausal(xF, f.base, f.lockout);
    f.acc = f.acc + sum(sp,1)'; f.into = f.into + size(xRaw,1);
    count = f.acc;
    f.acc = zeros(size(f.acc)); f.into = 0;
end

%% ===================== RHX TCP =====================
function c = rhxConnect(host, cmdPort, dataPort)
    if nargin<1||isempty(host),     host='127.0.0.1'; end
    if nargin<2||isempty(cmdPort),  cmdPort=5000;      end
    if nargin<3||isempty(dataPort), dataPort=5001;     end
    cmd = tcpclient(host, cmdPort, 'Timeout',3);
    c.data = tcpclient(host, dataPort, 'Timeout',3);
    c.nAmp = 64; c.nAdc = 1;
    cfg = {'set runmode stop','execute clearalldataoutputs'};
    for ch=0:c.nAmp-1, cfg{end+1}=sprintf('set b-%03d.tcpdataoutputenabled true',ch); end %#ok<AGROW>
    cfg{end+1} = 'set analog-in-2.tcpdataoutputenabled true';
    cfg{end+1} = 'set runmode run';
    for i=1:numel(cfg), writeline(cmd,cfg{i}); pause(0.02); end
    c.cmd = cmd;
end

function [blocks, leverBlk, state] = rhxReadBlocks(c, state)
    if ~isfield(state,'buf'),  state.buf = uint8([]); end
    if ~isfield(state,'samp'), state.samp = zeros(0,c.nAmp); state.lev = zeros(0,1); end
    if c.data.NumBytesAvailable > 0
        state.buf = [state.buf; read(c.data, c.data.NumBytesAvailable, 'uint8')'];
    end
    [amp, adc, state.buf] = rhxDecode(state.buf, c.nAmp, c.nAdc);
    if ~isempty(amp)
        state.samp = [state.samp; amp'];
        state.lev  = [state.lev;  adc'];
    end
    binSamp=400; K=floor(size(state.samp,1)/binSamp);
    blocks=zeros(binSamp,c.nAmp,K); leverBlk=zeros(1,K);
    for k=1:K
        idx=(k-1)*binSamp+(1:binSamp);
        blocks(:,:,k)=state.samp(idx,:);
        leverBlk(k)=median(state.lev(idx));
    end
    used=K*binSamp; state.samp(1:used,:)=[]; state.lev(1:used)=[];
end

function [amp, adc, rem] = rhxDecode(buf, nAmp, nAdc, fpb)
    if nargin<4, fpb=128; end
    mg = uint8([8 122 240 46]);   % 0x2ef07a08 little-endian
    per = 4 + nAmp*2 + nAdc*2; blk = 4 + fpb*per;
    ampC = {}; adcC = {}; N = numel(buf); i = 1;
    while i<=N-3 && ~(buf(i)==mg(1)&&buf(i+1)==mg(2)&&buf(i+2)==mg(3)&&buf(i+3)==mg(4)), i=i+1; end
    while i+blk-1 <= N
        if ~(buf(i)==mg(1)&&buf(i+1)==mg(2)&&buf(i+2)==mg(3)&&buf(i+3)==mg(4)), i=i+1; continue; end
        seg = reshape(buf(i+4:i+blk-1), per, fpb);
        aB = seg(5:4+nAmp*2,:); a = typecast(uint8(aB(:)),'uint16');
        ampC{end+1} = (double(reshape(a,nAmp,fpb))-32768)*0.195; %#ok<AGROW>
        dB = seg(5+nAmp*2:4+nAmp*2+nAdc*2,:); d = typecast(uint8(dB(:)),'uint16');
        adcC{end+1} = double(reshape(d,nAdc,fpb))*0.000050354; %#ok<AGROW>
        i = i + blk;
    end
    amp = [ampC{:}]; adc = [adcC{:}]; rem = buf(i:end);
end

%% ===================== helpers =====================
function F = lagFeat(D, W)
    [d,N] = size(D); F = zeros(N, d*W);
    for w = 0:W-1, F(:, w*d+(1:d)) = [zeros(w,d); D(:,1:N-w)']; end
end

function w = ldaFit(Xp, Xn)
    mu1 = mean(Xp,1); mu0 = mean(Xn,1);
    Sw = ((size(Xp,1)-1)*cov(Xp) + (size(Xn,1)-1)*cov(Xn)) / (size(Xp,1)+size(Xn,1)-2);
    Sw = Sw + 1e-2*mean(diag(Sw))*eye(size(Sw,1));
    a = Sw \ (mu1-mu0)'; b = -a'*((mu1+mu0)'/2); w = [a;b];
end

function al = binAlarmsAll(sc, binT, thv, dwell, refrB, live)
    pred = (sc>thv) & live; al=[]; run=0; last=-inf;
    for j=1:numel(sc)
        if pred(j), run=run+1; else, run=0; end
        if run>=dwell && (j-last)>=refrB, al(end+1)=binT(j); last=j; end %#ok<AGROW>
    end
end

function h = getf(fig), h = fig.UserData; end

function s = listSessions(root)
    d = dir(fullfile(root,'6119*'));
    s = {d([d.isdir]).name};
    if isempty(s), s = {'(none)'}; end
end

function c = rowcell(x)
    if isempty(x), c = {}; else, c = cellstr(x(:)'); end
end

function o = ternary(c, a, b)
    if c, o = a; else, o = b; end
end
