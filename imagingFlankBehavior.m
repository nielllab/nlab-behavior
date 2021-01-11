close all
clear all

% load('G62AAA4TT_subj.mat')
% load('20171027T133757_48.mat')  %%% frames 1:116 are noise

%%% first - load relevant files!
%%% these could also come in from a batch file

%load subject file and behavioral output (in ballData?)
% could also use uigetfile instead
load('G62AAA4TT_subj.mat')
load('20171027T133757_48.mat')  %%% frames 1:116 are noise

%load('20171102T171734_85.mat');

% load topoX and topoY results from dfofMovie
[f p] = uigetfile('*.mat','topox maps file')
load(fullfile(p,f),'mapNorm');
topox = polarMap(mapNorm{3});
figure
imshow(topox)

[f p] = uigetfile('*.mat','topoy maps file')
load(fullfile(p,f),'mapNorm','map');
topoy = polarMap(mapNorm{3});
figure
imshow(topoy)

%%% load behavior dfof output
[f p] = uigetfile('*.mat','behavior maps file');

downsize = 0.25;
load(fullfile(p,f));
df = imresize(dfof_bg,downsize);
topox = imresize(topox,downsize);
topoy = imresize(topoy,downsize);


mn = mean(mean(abs(df),2),1);
figure
plot(squeeze(mn)); title('mean abs dfof'); xlabel('frame')


%%% stimulus timing
%%% stimDetails = information for each condition
%%% allStop.frameT = time (in absolute value) of each frame, beginning  at start of stopping period
%%% allResp.frameT = time (in absolute value) of each frame, beginning  at start of stopping period

t0 = allStop(1).frameT(1);  %%% starting gun

% find when each stimulus presentation began
for i = 1:length(allResp);
    onsets(i) = allResp(i).frameT(1)-t0;
end

% remove part of movie if necessary
%goodStart=150;
%df(:,:,1:116)=0; %%% frames 1:116 of 102717 AAA4TT are noise
%df(:,:,1:goodStart)=0; %% frames 1:150 of 110217WW3RT are noise

%%% calculate stdev at each pixel
stdMap = std(df(:,:,1:10:end),[],3);
figure
imagesc(stdMap,[0 0.1])

%%% crop data based on stdev map (select upper left, bottom right
[x y] = ginput(2);
df = df(y(1):y(2),x(1):x(2),:);
topox_crop = topox(y(1):y(2),x(1):x(2),:);
topoy_crop = topoy(y(1):y(2),x(1):x(2),:);

%%% cropped stdev map
stdMap = std(df(:,:,1:10:end),[],3);
figure
imagesc(stdMap,[0 0.1])

%%% camera frame timing
frameT = frameT-frameT(1);  %%% time of camera frames; (from dfofMovie maps file?)
figure
plot(diff(frameT)); xlabel('frame'); ylabel('delta frame time')


clear onsetDf
for i = 1:length(onsets);
    onsetFrame(i) = find(diff(frameT>onsets(i)));  %%% frame immediate after stim onset
    onsetDf(:,:,:,i) = df(:,:,onsetFrame(i)-10:onsetFrame(i)+40);  %%% get frames from 10 before to 40 afterwards
end

%%% more dealing with bad trials - ignore for now
%badtrials = onsetFrame<=goodStart;
%sum(badtrials)

%%% plot mean image at multiple timepoints pre/post stim onset
figure
for f = 1:12;
    subplot(3,4,f)
    imagesc(mean(onsetDf(:,:,f*3,:),4)-mean(onsetDf(:,:,11,:),4) ,[-0.01 0.05])  %%% plot mean image for every 3rd frame, minus image at t=11 (stim onset)
    axis equal
end

%%% get contrast of the stimuli (targ + flanker)
for i=1:length(stimDetails);
    tc(i) = stimDetails(i).targContrast;
    fc(i) = stimDetails(i).flankContrast;
end

%%% assign appropriate contrast on each trial, based on which condition # it was
tcTrial = tc(trialCond);
fcTrial = fc(trialCond);

% more dealing with bad trials
%tcTrial(badtrials)=NaN;
%fcTrial(badtrials)=NaN;

range = [0 0.1];

% plot topox, topoy
figure
subplot(1,2,1);
imshow(topox_crop);
subplot(1,2,2);
imshow(topoy_crop);

% select 5 points to analyze (currently on topox/y, could also use activation map)
clear xpts ypts
for i = 1:5
    i
    [xpts(i) ypts(i)] = ginput(1);
    for s = 1:2
        subplot(1,2,s)
        hold on
        plot(xpts(i),ypts(i),'*');
    end
end



%%% show mean response map as a function of target contrasts
contrasts = unique(tc);
for c = 1:length(contrasts);
    figure
     trials =  abs(tcTrial)==contrasts(c);
    set(gcf,'Name',sprintf('tc = %0.3f  n = %d',contrasts(c),sum(trials)));
    for f = 1:12;
        subplot(3,4,f)
        imagesc(mean(onsetDf(:,:,f*3,trials),4)-mean(onsetDf(:,:,11,trials),4) ,range);
        hold on; plot(xpts,ypts,'r.')
        axis equal; axis off
    end
end

%%% show mean response map as a function of  flanker contrasts
contrasts = unique(abs(fc));
for c = 1:length(contrasts);
    figure
    trials =  abs(fcTrial)==contrasts(c);
    set(gcf,'Name',sprintf('fc = %0.3f  n = %d',contrasts(c),sum(trials)));
    for f = 1:12;
        subplot(3,4,f)
        imagesc(mean(onsetDf(:,:,f*3,trials),4)-mean(onsetDf(:,:,11,trials),4) ,range)
        axis equal; axis off
    end
end

%%% mean map as funciton of full parameters = flanker and target
contrasts = unique(abs(fc));
tcontrasts = unique(tc);
for c = 1:length(contrasts);
    for c2= 1:length(tcontrasts)
% for c = [1 3];
%     for c2 = [1 3];
        trials = abs(fcTrial)==contrasts(c) & tcTrial ==tcontrasts(c2);
        figure
        set(gcf,'Name',sprintf('fc = %0.3f tc = %0.3f n = %d',contrasts(c),tcontrasts(c2),sum(trials)));
        for f = 1:12;
            subplot(3,4,f)
            imagesc(median(onsetDf(:,:,f*3,trials),4)-mean(onsetDf(:,:,11,trials),4) ,range)
            axis equal; axis off
            hold on; plot(xpts,ypts,'r.'); colormap jet
        end
    end
end

figure
subplot(1,2,1);
imshow(topox_crop); hold on; plot(xpts,ypts,'wo')
subplot(1,2,2);
imshow(topoy_crop);hold on; plot(xpts,ypts,'wo')

% next step = don't just show points on the map, select onsetDf over time at these locations

