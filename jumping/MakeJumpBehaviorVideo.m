%% MakeJumpBehaviorVideo
%%%creates a video from just a single side camera for normal behavior

clear all
close all
dbstop if error

batchPhilJumping %load batch file
cd(datapathname) %change directory to video/tracking data

numAni = length(trialdata); %number of animals in dataset
frdur = 10; %duration of video to display (seconds)
goback = 4; %time to go back from jumptime in video
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
%             try
%                 load(fullfile(outpathname,fname))
%                 a = trace;
%             catch
                trace = {};
                vidfile = char(trialdata(ani).expt(expt).vidnames(vid)); %vid file name
                jumps = trialdata(ani).expt(expt).jumptime{vid}; %jump times for this vid
                for jump = 1:length(jumps) %cycle through jumps
                    sprintf('doing jump %d of %d',jump,length(jumps))
                    trace{jump} = analyzeJumpVid(vidfile,jumps(jump)-goback,frdur,jump); %get head bob trace for each jump
%                     trace{jump} = analyzeJumpVid(vidfile,jumps(jump)-goback,frdur); %get head bob trace for each jump
                end
%                 save(fullfile(outpathname,fname),'trace')
%             end
        end
    end
end