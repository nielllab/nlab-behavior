clear all
%%% to do

%%% save out results, and basic plots

%%% execute subject file to get parameters
[ f p] = uigetfile('*.m','subject file');
run(fullfile(p,f));

%%% load in taskFile, contains stimulus and etails
load(subj.taskFile);

%%% save out session information
sessionfile = [subj.dataLocation datestr(now,30) '_' num2str(sessions)];
fileList{sessions} = sessionfile;
save(subj.subjFile,'fileList','-append')
save(sessionfile,'sessions','subj','stimDetails');

%%% set up i/o for valves
setupPP;

%%% prepare screen
Screen('Preference', 'SkipSyncTests', 1);
win = Screen('OpenWindow',0,128);
framerate = Screen('FrameRate',win);

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
        label = sprintf('  N= %d c = %0.2f b = %0.2f',trial,mean(correct), mean(bias));
    end
    
    %%% do stopping period
    stopDetails = doStopping(interImg,framerate,subj,win,pp,label);
    if stopDetails.quit
        done =1;
        break
    end
    allStop(trial) = stopDetails;
    %%% do stimulus/response
    respDetails = doStimPeriod(stimulus(:,:,trialCond(trial)),stimDetails(trialCond(trial)),framerate,subj,win,pp,label);
    if respDetails.quit
        done=1;
        break
    end
    allResp(trial) = respDetails;
    
    save(sessionfile,'allStop', 'allResp','-append');
end
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
plot(field2array(allStop,'stopSecs')); title('stop time')

subplot(2,2,4);
r= field2array(allResp,'respTime');
plot(r); title('response time'); ylim([0 1.1*max(r)])
saveas(gcf,[sessionfile '_fig'],'jpg')
