clear all
close all
batchPhilJumping %load batch file
cd(datapathname) %change directory to video/tracking data

numAni = length(trialdata); %number of animals in dataset
frdur = 8; %duration of video to display (seconds)
goback = 5; %time in s to go back from 'jump time' recorded by experimenter
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
                    trace{jump} = analyzeJumpVid(vidfile,jumps(jump)-goback,frdur,jump); %get head bob trace for each jump, -5sec to get acutal jump
                    satis = input('satisfied? 1 = yes, other number = change by that time: ');
                    while satis~=1
                        trace{jump} = analyzeJumpVid(vidfile,jumps(jump)-goback+satis,frdur,jump); %get head bob trace for each jump, -5sec to get acutal jump
                        satis = input('satisfied? 1 = yes, other number = change by that time: ');
                    end
                end
                save(fullfile(outpathname,fname),'trace')
            end
        end
    end
end

%% plot individual experiment data
pixperin = 60; %number of pixels per inch
FR = 50; %frame rate
for ani = 1:numAni %cycle through animals
    sprintf('plotting animal %d of %d',ani,numAni)
    expts=length(trialdata(ani).expt); %num expts for this animal
    for expt = 1:expts %cycle through experiments
        sprintf('plotting experiment %d of %d',expt,expts)
        vids = length(trialdata(ani).expt(expt).vidnames); %num videos for this expt
        tracetemp = {};
        for vid = 1:vids %cycle through vids
            clear trace
            fname = sprintf('%s_%s_%d_headbob.mat',trialdata(ani).name,trialdata(ani).expt(expt).date,vid);
            load(fullfile(outpathname,fname),'trace')
            tracetemp = [tracetemp trace];
        end
        trace = tracetemp;
        dists = trialdata(ani).expt(expt).distance;
        [B,I] = sort(dists);

        figure;set(gcf,'color','w')
        subplot(1,3,1)
        hold on
        for i = 1:length(B)
            tr = -trace{I(i)}(1,:)/60;tr = tr-(mean(tr));
            plot(1:length(tr),tr+1*i)
        end
        xlabel('time (s)')
        ylabel('platform distance (in)')
        axis([0 150 0 length(trace)+5])
        set(gca,'xtick',0:50:150,'xticklabel',0:3,'ytick',1:length(trace),'yticklabel',B)
        title('nose y-position')

        subplot(1,3,2)
        hold on
        for i = 1:length(B)
            tr = -trace{I(i)}(2,:)/60;tr = tr-(mean(tr));
            plot(1:length(tr),tr+1*i)
        end
        xlabel('time (s)')
        ylabel('platform distance (in)')
        axis([0 150 0 length(trace)+5])
        set(gca,'xtick',0:50:150,'xticklabel',0:3,'ytick',1:length(trace),'yticklabel',B)
        title('nose x-position')
        
        
        bobdurs = nan(1,length(trace));
        for i = 1:length(trace)
            bobdurs(i) = length(trace{i});
        end
        bobdurs = bobdurs(I)/FR;
        y = polyfit(B,bobdurs,1);
        y_est = polyval(y,B);
        subplot(1,3,3)
        hold on
        scatter(B,bobdurs,'ko')
        plot(B,y_est,'r-')
        axis square
        xlabel('distance (in)')
        ylabel('duration of bobbing (s)')
        mtit(sprintf('%s %s',trialdata(ani).name,trialdata(ani).expt(expt).date))
        

        figure;set(gcf,'color','w')
        nrows = ceil(length(trace)/5);
        for i = 1:length(B)
            tr1 = -trace{I(i)}(1,:)/60;%tr = tr-(mean(tr));
            tr2 = -trace{I(i)}(2,:)/60;
            subplot(nrows,5,i)
            plot(tr2,tr1)
        end
%             xlabel('nose x-position')
%             ylabel('nose y-position')
%             axis([0 150 0 length(trace)+5])
%             set(gca,'xtick',0:50:150,'xticklabel',0:3,'ytick',1:length(trace),'yticklabel',B)
        title('nose movement')
        
        mtit(sprintf('%s %s',trialdata(ani).name,trialdata(ani).expt(expt).date))
    end
end
