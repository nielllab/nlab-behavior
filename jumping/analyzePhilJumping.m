%% analyzePhilJumping

close all
clear all
dbstop if error

batchPhilJumping
% cd(pathname)

ndist = 11; % number of potential distances to jump
mindist = 3; % minimum jumping distance (usually 3 in) to ignore stepping
nplat = 3; % number of platforms used
platsz = {'3.5"','4.0"','4.5"'}; % platform sizes
mycol = {'k','b','r'}; % list of colors for platforms in plots

psfilename = 'c:\tempPhil.ps';
if exist(psfilename,'file')==2;delete(psfilename);end

%% get success rate for each platform at each distance

numAni = length(trialdata);
disp(sprintf('%d animals to analyze',numAni))
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