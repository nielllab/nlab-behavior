clear all
%%% to do

%%% save out results, and basic plots

%%% execute subject file to get parameters
[ f p] = uigetfile('*.m','subject file');
run(fullfile(p,f));

%%% load in taskFile, contains stimulus and details
load(subj.taskFile);

%%% save out session information
sessionfile = [subj.dataLocation '\' datestr(now,30) '_' num2str(sessions)];
fileList{sessions} = sessionfile;
save(subj.subjFile,'fileList','-append')
save(sessionfile,'sessions','subj','stimDetails');

%%% set up i/o for valves
setupPP;
pinDefs; %%% read in pin definitions
trigT=[];
global pinState
pinState=0;

%%% prepare screen
Screen('Preference', 'SkipSyncTests', 1);
win = Screen('OpenWindow',0,128);
framerate = Screen('FrameRate',win);

for i = 1:size(stimulus,3);
     tex(i)=Screen('MakeTexture', win, stimulus(:,:,i)');
end

%%% loop over trials
done = 0;
trial =0;
ListenChar(2);

while ~done
    trial = trial+1;
    
    %%% choose condition for this trial
    trialCond(trial) = ceil(rand*length(stimDetails))
    
    %%% correction trial?
    if trial>1 && allResp(trial-1).correct==0 && rand(1)<subj.correctionProb
        trialCond(trial) = trialCond(trial-1);
    end
    
    %%% make label for screen with info
    if trial  ==1
        label=sprintf('  N = %d',1);
    else
        correct = field2array(allResp,'correct');
        bias = field2array(allResp,'response')>0;
        if trial>50
            running = mean(correct(end-49:end)); 
            else running = NaN;
        end
        label = sprintf('  N= %d c = %0.2f cr = %0.2f b = %0.2f',trial,mean(correct), running,mean(bias));
    end
    
    %%% do stopping period
    [stopDetails trigT] = doStopping_yoke(interImg,framerate,subj,win,pp,label,pin,trigT, subj.meanStop);
    if stopDetails.quit
        done =1;
        break
    end
    allStop(trial) = stopDetails;
    %%% do stimulus/response
    [respDetails trigT] = doStimPeriod_yoke(tex(trialCond(trial)),stimDetails(trialCond(trial)),framerate,subj,win,pp,label,pin,trigT, subj.meanResp);
    if respDetails.quit
        done=1;
        break
    end
    allResp(trial) = respDetails;
    
    save(sessionfile,'allStop', 'allResp','trialCond','-append');
end
trialCond = trialCond(1:length(allResp)); allStop = allStop(1:length(allResp)); %%% trim cond/stop in case didn't finish trial
ListenChar(1);
Priority(0);
Screen('CloseAll');

%%% plot basic results
correct = field2array(allResp,'correct');
bias = field2array(allResp,'response')>0;

figure
subplot(2,2,1);
bar([1 2],[mean(correct) mean(bias)]); set(gca, 'xticklabel',{'correct','bias'}); ylim([0 1]);

subplot(2,2,2);
plot(conv(correct, ones(1,10),'valid')/10,'g'); hold on
plot(conv(double(bias), ones(1,10),'valid')/10,'r'); legend('correct','bias'); ylim([0 1])

subplot(2,2,3);
plot(log10(field2array(allStop,'stopSecs')),'.'); title('stop time log10')

subplot(2,2,4);
r= field2array(allResp,'respTime');
plot(log10(r),'.'); title('response time log10'); ylim([-1 1.1*max(log10(r))])
saveas(gcf,[sessionfile '_fig'],'jpg')

clear label flankResp flankBias biasLower biasUpper respLower respUpper
if isfield(stimDetails,'flankContrast')
    figure
    flankC = field2array(stimDetails(trialCond),'flankContrast');
    c = unique(flankC);
    for i = 1: length(c)
       label{i} = num2str(c(i));
       use = flankC==c(i);       
      [mn ci] = binofit( sum(correct(use)),sum(use));
      flankResp(i) = mn; respLower(i) = mn-ci(1); respUpper(i) = ci(2)-mn;
         [mn ci] = binofit(sum(bias(use)),sum(use));
      flankBias(i) = mn; biasLower(i) = mn-ci(1); biasUpper(i) = ci(2)-mn;
    end
    figure
     errorbar((1:length(c))+0.1,flankBias,biasLower,biasUpper,'r-o');hold on; 
    errorbar((1:length(c))-0.1,flankResp,respLower,respUpper,'b-o'); ylim([0 1])
    set(gca,'Xtick',1:length(c));set(gca,'XTickLabel',label); xlabel('contrast')
end

clear label flankResp flankBias biasLower biasUpper respLower respUpper
if isfield(stimDetails,'flankContrast')
    figure
    flankC = field2array(stimDetails(trialCond),'flankContrast');
    targC = field2array(stimDetails(trialCond),'targContrast');
    fc = unique(flankC);
    tc = unique(targC);
    for f = 1: length(fc);
        for t = 1:length(tc)
            label{f} = num2str(fc(f));
            tlabel{t} = num2str(tc(t));
            use = flankC==fc(f) & targC==tc(t);
            [mn ci] = binofit( sum(correct(use)),sum(use));
            flankResp(f,t) = mn; respLower(f,t) = mn-ci(1); respUpper(f,t) = ci(2)-mn;
            [mn ci] = binofit(sum(bias(use)),sum(use));
            flankBias(f,t) = mn; biasLower(f,t) = mn-ci(1); biasUpper(f,t) = ci(2)-mn;
        end
    end
    
    col = 'rgbcmy';
    figure
    for i = 1:length(tc);
        errorbar((1:length(fc))+0.1,flankBias(:,i),biasLower(:,i),biasUpper(:,i),[col(i) '-o']);hold on;
    end
    title('bias'); set(gca,'Xtick',1:length(c));set(gca,'XTickLabel',label); xlabel('contrast'); ylim([0 1]);
    legend(tlabel);
    
    col = 'rgbcmy';
    figure
    for i = 1:length(tc);
        errorbar((1:length(fc))+0.1,flankResp(:,i),respLower(:,i),respUpper(:,i),[col(i) '-o']);hold on;
    end
    title('correct'); set(gca,'Xtick',1:length(c));set(gca,'XTickLabel',label); xlabel('contrast'); ylim([0 1]);
    legend(tlabel);
end

figure
plot(diff(trigT)); xlabel('frame'); ylabel('trigger interval secs'); ylim([0.05 0.15]);

ft = [];
for i = 1:length(allResp);
    ft = [ft allStop(i).frameT allResp(i).frameT];
end
figure
plot(diff(ft)); ylim([0 0.05]); ylabel('video frame interval'); 
