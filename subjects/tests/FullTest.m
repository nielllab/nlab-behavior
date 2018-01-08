
%%% subject information
subj.name = 'the dude';
subj.dataLocation = 'c:\balldata\the_dude\';
subj.taskFile = 'FullFlankerTask';

%%% monitor position
subj.monitorPosition = 'landscape'

%%% stopping
subj.stopDuration=1;
subj.stopThresh=60;
subj.stopReward=0; %%% duration

%%% response
subj.respThresh = 600;
subj.maxStimduration = 10000;  %%%% timeout
subj.rewardDuration=0.182;

%%% post-response
subj.correctDuration = 1;
subj.errorDuration = 1;
subj.correctionProb = 0.5;  %%% probability of correction trial after error



%%% set up files

%%% directory
if ~isdir(subj.dataLocation)
    mkdir(subj.dataLocation);
end

%%% update subject data file (if it exists) or create
subj.subjFile = [subj.dataLocation subj.name '_subj.mat'];
if exist(subj.subjFile,'file')
   display('exists')
   load(subj.subjFile);
    sessions = sessions+1;
    subjData{sessions} = subj;
    save(subj.subjFile,'sessions','subjData','-append');
else
    sessions = 1;
    subjData{sessions} = subj;
    save(subj.subjFile,'sessions','subjData');
end


