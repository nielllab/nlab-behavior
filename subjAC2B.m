%%% subject information
subj.name = 'g62b7lt';
subj.dataLocation = 'c:\balldata\g62b7lt\';
subj.taskFile = 'HvVtask';

%%% stopping
subj.stopDuration=1;
subj.stopThresh=100;
subj.stopReward=0.15; %%% duration

%%% response
subj.respThresh = 300;
subj.maxStimduration = 10;
subj.rewardDuration=0.15;

%%% post-response
subj.correctDuration = 0.5;
subj.errorDuration = 0.5;
subj.correctionProb = 0.25;  %%% probability of correction trial after error



%%% set up files

%%% directory
if ~isdir(subj.dataLocation)
    mkdir(subj.dataLocation);
end

%%% update subject data file (if it exists) or create
subj.subjFile = [subj.dataLocation subj.name '_subj'];
if exist(subj.subjFile,'file')
    load(subj.subjFile);
    sessions = sessions+1;
    subjData{sessions} = subj;
    save(subj.subjFile,'sessions','subjData','-append');
else
    sessions = 1;
    subjData{sessions} = subj;
    save(subj.subjFile,'sessions','subjData');
end


