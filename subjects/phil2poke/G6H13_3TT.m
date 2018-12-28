%%% subject information
subj.name = 'G6H13.3TT';
subj.dataLocation = 'C:\Users\nlab\Documents\MATLAB\behavior';
subj.taskFile = 'BigSmall2AFC'; %make this
subj.bigsmall = 'big'; %select whether this subject is rewarded for big or small
subj.maxdur = 20; %max time they have to respond after stim is up
subj.opendur0 = 0.12; %open time of Ch0 valve
subj.opendur1 = 0.1; %open time of Ch1 valve

%%% monitor position
subj.monitorPosition = 'landscape'

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


