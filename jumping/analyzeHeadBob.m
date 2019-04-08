clear all
close all
batchPhilJumping %load batch file
cd(datapathname) %change directory to video/tracking data

numAni = length(trialdata); %number of animals in dataset
frdur = 240; %duration of video to display (frames*seconds)
%%
for ani = 1:numAni %cycle through animals
    sprintf('doing animal %d of %d',ani,numAni)
    expts=length(trialdata(ani).expt); %num expts for this animal
    for expt = 1:expts %cycle through experiments
        sprintf('doing experiment %d of %d',expt,expts)
        vids = length(trialdata(ani).expt(expt).vidnames); %num videos for this expt
        for vid = 1:vids %cycle through vids
            sprintf('doing video %d of %d',vid,vids)
            clear trace
            fname = sprintf('%s_%s_%d_headbob.mat',trialdata(ani).name,trialdata(ani).expt(expt).date,vid);
            if ~exist(fname,'file')
                trace = {};
                vidfile = char(trialdata(ani).expt(expt).vidnames(vid)); %vid file name
                jumps = trialdata(ani).expt(expt).jumptime{vid}; %jump times for this vid
                for jump = 1:length(jumps) %cycle through jumps
                    sprintf('doing jump %d of %d',jump,length(jumps))
                    trace{jump} = analyzeJumpVid(vidfile,jumps(jump)-5,frdur,jump); %get head bob trace for each jump, -5sec to get acutal jump
                end
                save(fullfile(outpathname,fname),'trace')
            end
        end
    end
end
