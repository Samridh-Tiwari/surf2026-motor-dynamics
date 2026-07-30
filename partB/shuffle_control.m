function shuffle_control(runmat)
%SHUFFLE_CONTROL  Base-rate-corrected test: is the detector genuinely predictive,
%   or does it just catch frequent pulls by chance? Uses a CIRCULAR-SHIFT null
%   (keeps the alarm rate AND any periodicity, only destroys alignment to pulls).
%   If the model fires randomly/periodically, shifted trains score the same ->
%   z~0. If it truly aligns to movement, shifts break it -> large positive z.
    if nargin<1
        runmat='C:\Users\tiwar133\gpfa_v0203\Data_Realtime\ClosedLoopGUI_standalone\runs\run_20260724_160137\run_stats.mat';
    end
    S=load(runmat); al=S.alarms(:); pl=S.pulls(:); T=S.elapsed; W=1.5;   % "pull follows within W s"
    if numel(al)<5 || numel(pl)<5, fprintf('too few events\n'); return; end
    outDir=fileparts(runmat);

    % --- metric 1: PRECISION (fraction of alarms with a pull soon after) ---
    prec=@(a) mean(arrayfun(@(x) any(pl>x & pl<=x+W), a));
    precReal=prec(al);
    % --- metric 2: RECALL (fraction of pulls with an alarm just before) ---
    rec=@(a) mean(arrayfun(@(o) any(a>=o-W & a<=o), pl));
    recReal=rec(al);

    nperm=5000; precN=zeros(nperm,1); recN=zeros(nperm,1);
    for i=1:nperm
        sh=mod(al + rand*T, T);      % circular shift by a random offset
        precN(i)=prec(sh); recN(i)=rec(sh);
    end
    zP=(precReal-mean(precN))/(std(precN)+eps); pctP=100*mean(precReal>precN);
    zR=(recReal -mean(recN)) /(std(recN)+eps);  pctR=100*mean(recReal>recN);

    % --- metric 3: lead-time concentration vs uniform ---
    lead=S.leads_ms(:); lead=lead(~isnan(lead));
    fracTight=mean(lead>=0 & lead<=150);           % model: fraction of leads in a tight pre-pull window
    chanceTight=(150/1000)/W;                        % uniform expectation within W

    fprintf('\n================ SHUFFLE CONTROL (%d alarms, %d pulls, %.0f s) ================\n',numel(al),numel(pl),T);
    fprintf('base rate: a pull occurs somewhere ~every %.1f s -> chance is HIGH, so raw hit-rate is meaningless\n', T/numel(pl));
    fprintf('\nPRECISION (alarm -> pull within %.1fs):\n',W);
    fprintf('   model %.1f%%  |  chance %.1f%% +/- %.1f  |  z = %+.1f  (beats %.1f%% of shuffles)\n', ...
        100*precReal, 100*mean(precN), 100*std(precN), zP, pctP);
    fprintf('RECALL (pull preceded by alarm within %.1fs):\n',W);
    fprintf('   model %.1f%%  |  chance %.1f%% +/- %.1f  |  z = %+.1f  (beats %.1f%% of shuffles)\n', ...
        100*recReal, 100*mean(recN), 100*std(recN), zR, pctR);
    fprintf('LEAD concentration (fraction of leads in +0..150ms):\n');
    fprintf('   model %.1f%%  |  uniform-chance %.1f%%  |  ratio %.1fx\n', 100*fracTight, 100*chanceTight, fracTight/max(chanceTight,eps));
    verdict=ternary(zP>3||zR>3,'ABOVE CHANCE - genuinely predictive','NOT above chance - could be base-rate artifact');
    fprintf('\nVERDICT: %s\n',verdict);
    fprintf('=================================================================================\n');

    f=figure('Color','w','Position',[80 80 1000 400],'Visible','off');
    tiledlayout(f,1,2,'Padding','compact','TileSpacing','compact');
    nexttile; histogram(100*precN,40,'FaceColor',[.6 .6 .65],'EdgeColor','none'); hold on;
    xline(100*precReal,'-','Color',[.13 .55 .29],'LineWidth',3,'Label','model'); grid on;
    xlabel('precision (%)'); ylabel('# shuffles'); title(sprintf('Precision vs circular-shift null (z=%+.1f)',zP));
    nexttile; histogram(100*recN,40,'FaceColor',[.6 .6 .65],'EdgeColor','none'); hold on;
    xline(100*recReal,'-','Color',[.13 .55 .29],'LineWidth',3,'Label','model'); grid on;
    xlabel('recall (%)'); ylabel('# shuffles'); title(sprintf('Recall vs null (z=%+.1f)',zR));
    exportgraphics(f,fullfile(outDir,'shuffle_control.png'),'Resolution',150); close(f);
    fprintf('Saved -> %s\n',fullfile(outDir,'shuffle_control.png'));
end
function o=ternary(c,a,b), if c, o=a; else, o=b; end, end
