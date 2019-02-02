%% analyzePhilJumping

close all
clear all
dbstop if error

batchPhilJumping
cd(datapathname)

vidan = input('analyze video? 0=no, 1=yes: ');

ndist = 11; % number of potential distances to jump
mindist = 3; % minimum jumping distance (usually 3 in) to ignore stepping
nplat = 3; % number of platforms used
platsz = {'3.5"','4.0"','4.5"'}; % platform sizes
mycol = {'k','b','r'}; % list of colors for platforms in plots

numAni = length(trialdata);
disp(sprintf('%d animals to analyze',numAni))

%% get success rate for each platform at each distance
psfilename = 'c:\tempPhil.ps';
if exist(psfilename,'file')==2;delete(psfilename);end

grpjump = nan(nplat,ndist,numAni);
for ani = 1:numAni
    numExpt = length(trialdata(ani).expt);
    disp(sprintf('animal %d has %d experiments to analyze',ani,numExpt))
    jumpdata = nan(nplat,ndist,numExpt);
    
    for expt = 1:numExpt
        trials = trialdata(ani).expt(expt).trials;
        platform = trialdata(ani).expt(expt).platform;
        distance = trialdata(ani).expt(expt).distance;
        success = trialdata(ani).expt(expt).success;
        
        for plat = 1:nplat
            for dist = mindist:ndist
                jumpdata(plat,dist,expt) = sum(success(intersect(find(distance==dist),find(platform==plat))))/...
                    length(success(intersect(find(distance==dist),find(platform==plat))));
            end
        end
    end
    mnjump = nanmean(jumpdata,3);
    sejump = nanstd(jumpdata,[],3)/sqrt(numExpt);
    trialdata(ani).mnjump = mnjump;
    trialdata(ani).sejump = sejump;
    
    figure;set(gcf,'color','w')
    hold on
    for plat = 1:nplat
        rndoff = 0.1*(rand([1,ndist])-0.5); %random x offset
        errorbar([1:ndist]+rndoff,mnjump(plat,:),sejump(plat,:),'o:','color',mycol{plat},'MarkerSize',15)
    end
    xlabel('Jump Distance (in)')
    ylabel('Success Rate')
    title(sprintf('%s',trialdata(ani).name))
    axis([2 ndist+1 0 1.05])
    set(gca,'xtick',3:2:11,'ytick',0:0.25:1,'tickdir','out')
    legend(platsz,'location','southwest')
    if exist('psfilename','var')
        set(gcf, 'PaperUnits', 'normalized', 'PaperPosition', [0 0 1 1], 'PaperOrientation', 'landscape');
        print('-dpsc',psfilename,'-append');
    end
    
    grpjump(:,:,ani) = mnjump;
end

figure;set(gcf,'color','w')
hold on
for plat = 1:nplat
    rndoff = 0.1*(rand([1,ndist])-0.5); %random x offset
    errorbar([1:ndist]+rndoff,nanmean(grpjump(plat,:,:),3),nanstd(grpjump(plat,:,:),[],3)/sqrt(numAni),'o:','color',mycol{plat},'MarkerSize',15)
end
legend
xlabel('Jump Distance (in)')
ylabel('Success Rate')
title('Group Average')
axis([2 ndist+1 0 1.05])
set(gca,'xtick',3:2:11,'ytick',0:0.25:1,'tickdir','out')
legend(platsz,'location','southwest')
if exist('psfilename','var')
    set(gcf, 'PaperUnits', 'normalized', 'PaperPosition', [0 0 1 1], 'PaperOrientation', 'landscape');
    print('-dpsc',psfilename,'-append');
end

figure;set(gcf,'color','w')
hold on
errorbar(1:ndist,nanmean(nanmean(grpjump,1),3),nanstd(nanmean(grpjump,1),[],3)/sqrt(numAni),'ko:','MarkerSize',15)
xlabel('Jump Distance (in)')
ylabel('Success Rate')
title('Group Average')
axis([2 ndist+1 0 1.05])
set(gca,'xtick',3:2:11,'ytick',0:0.25:1,'tickdir','out')
if exist('psfilename','var')
    set(gcf, 'PaperUnits', 'normalized', 'PaperPosition', [0 0 1 1], 'PaperOrientation', 'landscape');
    print('-dpsc',psfilename,'-append');
end

try
    dos(['ps2pdf ' psfilename ' "' 'JumpingGroupData.pdf' '"'])
catch
    display('couldnt generate pdf');
end

%% analyze video
if vidan
    frdur = 250; %~10sec for ~25fps
    frrate = 23; %fps
    for ani = 1:numAni
        numExpt = length(trialdata(ani).expt);
        disp(sprintf('animal %d has %d potential video experiments to analyze',ani,numExpt))
        for expt = 1:length(numExpt)
            fname = [trialdata(ani).name '_' trialdata(ani).expt(expt).date '.mat'];
            trials = trialdata(ani).expt(expt).trials;
            trace = nan(frdur,5,2,trials); %time points by number of tracked points,x/y,trial#
            if isempty(trialdata(ani).expt(expt).vidnames)
                disp(sprintf('expt %d has no video, skipping...',expt))
                continue
            elseif exist(fullfile(outpathname,fname),'file')
                disp(sprintf('expt %d video already analyzed, skipping...',expt))
                continue
            else
                cnt=1;
                nvid = length(trialdata(ani).expt(expt).vidnames);
                for vid = 1:nvid
                    jumptime = cell2mat(trialdata(ani).expt(expt).jumptime(vid));
                    vidfile = cell2mat(fullfile(trialdata(ani).expt(expt).vidnames(vid)));
                    for i = 1:length(jumptime)
                        [Pointsx,Pointsy] = analyzeJumpVid(vidfile,jumptime(i),frdur-1);
                        trace(:,:,1,cnt) = Pointsx;
                        trace(:,:,2,cnt) = Pointsy;
                        cnt=cnt+1;
                        disp(sprintf('done %d of %d vids',i,length(jumptime)))
                    end
                end
            end
            save(fname,'trace')
        end
    end
end

        
%% test plotting of bobs
fouri = {};
for tp = 1:size(trace,4)
    trY = squeeze(trace(:,1,2,tp));
    trX = squeeze(trace(:,1,1,tp));
    figure;
    hold on
    plot(trY,'k-')
    plot(trX,'b-')    
    tvals = round(ginput(2));
    sig = trY(tvals(1):tvals(2));
    close
    
    Fs = frrate;            % Sampling frequency                    
    T = 1/Fs;             % Sampling period       
    L = length(sig);             % Length of signal
    t = (0:L-1)*T;        % Time vector

    n = 2^nextpow2(L);
    Y = fft(sig,n);
    P2 = abs(Y/L);
    P1 = P2(1:n/2+1);
    P1(2:end-1) = 2*P1(2:end-1);

    figure
    plot(0:(Fs/n):(Fs/2-Fs/n),P1(1:n/2))
    xlabel('freq')
    ylabel('power')
    % close

    fouri{tp} = P1;
end

