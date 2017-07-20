clear all
%%% to do
%%% correction trials
%%% release water and quit
%%% save out results, and basic plots



%%% execute subject file to get parameters
[ f p] = uigetfile('*.m','subject file');
run(fullfile(p,f));
%%% load in taskFile, contains stimulus and etails
load(subj.taskFile);

addr='0378'; ioObj = io64; status = io64(ioObj);
if status~=0
    status, error('driver installation not successful')
end
valveAddr = hex2dec(addr);
io64(ioObj,valveAddr,0);

%%% prepare scree
Screen('Preference', 'SkipSyncTests', 1);
win = Screen('OpenWindow',0,128);
framerate = Screen('FrameRate',win);

%%% loop over trials
done = 0;
trial =0;
ListenChar(2);
while ~done
    trial = trial+1;
    if trial>4
        done=1;
    end
    
    %%% choose condition for this trial
    trialCond(trial) = ceil(rand*length(stimDetails))
    
    %     if trial>1 && allResp(trial-1).correct==0 && rand(1)<subj.correctionProb
    %         trialCond(trial) = trialCond(trial-1);
    %     end
    
    %%% do stopping period
    stopDetails = doStopping(interImg,framerate,subj,win);
    if stopDetails.quit
        done =1;
        break
    end
    allStop(trial) = stopDetails;
    %%% do stimulus/response
    respDetails = doStimPeriod(stimulus(:,:,trialCond(trial)),stimDetails(trialCond(trial)),framerate,subj,win);
    if respDetails.quit
        done=1;
    end
    allResp(trial) = respDetails;
end
ListenChar(1);
Priority(0);
Screen('CloseAll');
