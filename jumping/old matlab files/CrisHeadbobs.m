clear all
close all
batchPhilJumping %load batch file
cd(datapathname) %change directory to video/tracking data
numAni = length(trialdata); %number of animals in dataset

% [f p] = uigetfile('*.mat','headbob .mat data');
% load(fullfile(p,f));

pixperin = 60; %number of pixels per inch
FR = 50; %frame rate
for ani = 1%:numAni %cycle through animals
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

        xfig = figure; hold on; set(gcf,'Name','x position');
        yfig = figure; hold on; set(gcf,'Name','y position');

        %xyfig = figure; hold on
        for i = 1:length(trace)

            % pre-process x values
            x = trace{i}(2,end:-1:1); %%% reverse so they are all aligned on takeoff time
            x = x-mean(x); % zero-center
            x = medfilt1(x,3); %median filtering helps remove outlier points
            figure(xfig); plot(x/20 + i);

            % do same for y values
            y = trace{i}(1,end:-1:1);
            y = y-mean(y);  
            y= medfilt1(y,3);
             figure(yfig); plot(y/20 + i);

             %%% store out data into one large matrix
             xall(i,1:length(x)) = x;
             vxall(i,1:length(x)-1) = diff(x);
             yall(i,1:length(y)) = y;
             vyall(i,1:length(y)-1) = diff(y);
             n(i) = length(x);
        %     figure
        %     plot(x,y); axis([-40 40 -40 40]); axis square

        end
        xall = xall(I,:);
        vxall = vxall(I,:);
        yall = yall(I,:);
        vyall = vyall(I,:);
        n = n(I);
        %%% flip time axis back now that the matrices are filled up
        xall = fliplr(xall); yall=fliplr(yall); vxall=fliplr(vxall); vyall=fliplr(vyall);

        % [val i] = sort(n);
        % 
        % 
        % cmap = cbrewer('div','RdBu',64); cmap = flipud(cmap);
        % figure
        % imagesc(xall,[-50 50]); colormap(cmap)
        % figure
        % imagesc(yall,[-50 50]);  colormap(cmap)
        % 
        % figure
        % imagesc(vxall,[-20 20]);  colormap(cmap)
        % figure
        % imagesc(vyall,[-20 20]);  colormap(cmap)


        %%% find all peaks
        figure
        hold on;
        np = 0; clear allpeaks allpeaksy height
        height_thresh=10; % minimum amplitude to be considered a "bob"
        range = 12;
        avgamp = nan(1,size(vxall,1)); totbob = avgamp;
        for i = 1:size(vxall,1);
            x = xall(i,:); y = yall(i,:); vx = vxall(i,:); vy = vyall(i,:);
            vsmooth= conv(vy,[1 1 1],'same'); %%% need to smooth since median filter leaves some values identical, meaning derivative=0
            peaks = find(vsmooth(1:end-1)<0 & vsmooth(2:end)>0)+1;  %%% find peaks based on difference in derivative on either side
            peak= peaks(peaks>range & peaks<max(n)-range);  % remove peaks that are too close to edge of data
            npeaks(i) = length(peak);
            plot(y/20+i);
            allheight = 0;cnt=0;
            for j = 1:length(peak)
                np = np+1;
                r = peak(j)-range : peak(j)+range;  % data range around each peak
                allpeaks(np,:) = y(r); % get data snippet around peak
                allpeaks(np,:) = allpeaks(np,:)-mean(allpeaks(np,:));
                height(np) = allpeaks(np,round(end/2));  %% height  = value at midpoint of snippet
                allpeaksx(np,:) = x(r);  % get corresponding x values
                allpeaksx(np,:) = allpeaksx(np,:)-mean(allpeaksx(np,:));
                % mark peaks on traces (blue= above threshold, red=below)
                if height(np)>height_thresh
                    plot(peak(j),y(peak(j))/20 +i,'b.');
                    allheight = allheight + y(peak(j));
                    cnt = cnt + 1;
                else
                    plot(peak(j),y(peak(j))/20 +i,'r.');
                end
            end
            avgamp(i) = allheight/cnt/pixperin;totbob(i) = cnt;
        end


        figure
        bobdurs = n/FR;
        y = polyfit(B,bobdurs,1);
        y_est = polyval(y,B);
        R = corrcoef(B,bobdurs);
        subplot(1,3,1)
        hold on
        scatter(B,bobdurs,'ko')
        plot(B,y_est,'r-')
        axis square
        ylim([0 6])
        xlabel('distance (in)')
        ylabel('duration of bobbing (s)')
        title(sprintf('R2=%0.3f',R(2)))
        hold off

        subplot(1,3,2)
        y = polyfit(B,totbob,1);
        y_est = polyval(y,B);
        R = corrcoef(B,totbob);
        hold on
        scatter(B,totbob,'ko')
        plot(B,y_est,'r-')
        axis square
        ylim([0 10])
        xlabel('distance (in)')
        ylabel('number of bobs')
        title(sprintf('R2=%0.3f',R(2)))
        hold off
        
        subplot(1,3,3)
        ind = ~isnan(avgamp);B = B(ind);avgamp = avgamp(ind);
        y = polyfit(B,avgamp,1);
        y_est = polyval(y,B);
        R = corrcoef(B,avgamp);
        hold on
        scatter(B,avgamp,'ko')
        plot(B,y_est,'r-')
        axis square
        ylim([0 1.15])
        xlabel('distance (in)')
        ylabel('bob amplitude')
        title(sprintf('R2=%0.3f',R(2)))
        hold off
        mtit(sprintf('%s %s',trialdata(ani).name,trialdata(ani).expt(expt).date))
        
        %%% summary figure

        figure

        % histogram of heights
        subplot(2,2,1)
        histogram(height/pixperin,10)
        xlim([-1 1])
        xlabel('bob amp (in)')
        ylabel('frequency')

        % mean trace for x and y position
        subplot(2,2,2)
        mn = median(allpeaks(height>height_thresh,:),1); mny = mn-min(mn);
        plot((mn-min(mn))/pixperin,'r');
        xlim([1 size(allpeaks,2)])
        ylim([0 50/pixperin])
        hold on
        mn = median(allpeaksx(height>height_thresh,:),1); mnx = mn-min(mn);
        plot((mn-min(mn))/pixperin,'b');
        legend('y','x');
        title(sprintf('bobs / trial = %0.2f',sum(height>height_thresh)/size(xall,1)));
        xlabel('time (ms)')
        ylabel('bob amp (in)')
        set(gca,'xtick',0:6:30,'xticklabel',0:200:1000)

        % all y positions (colorcoded, blue = above height threshold)
        subplot(2,2,3);
        plot(allpeaks(height<height_thresh,:)'/pixperin,'r'); hold on
        plot(allpeaks(height>height_thresh,:)'/pixperin,'b'); 
        title('ypos'); axis([1 size(allpeaks,2) -75 75])
        xlabel('time (ms)')
        ylabel('bob amp (in)')
        set(gca,'xtick',0:6:30,'xticklabel',0:200:1000)
        ylim([-75/pixperin 75/pixperin])

        % all x positions
        subplot(2,2,4);
        plot(allpeaksx(height<height_thresh,:)'/pixperin,'r'); hold on
        plot(allpeaksx(height>height_thresh,:)'/pixperin,'b'); 
        title('x pos'); axis([1 size(allpeaks,2) -75 75])
        xlabel('time (ms)')
        ylabel('bob amp (in)')
        set(gca,'xtick',0:6:30,'xticklabel',0:200:1000)
        ylim([-75/pixperin 75/pixperin])

        %%%
        figure;colormap jet
        hold on
        colors = jet(length(mnx));
        for i = 1:length(colors)-1
            plot(mnx(i:i+1),mny(i:i+1),'-','color',colors(i,:),'LineWidth',2); 
        end
        axis square
        axis([0 50 0 50])
        title('mean bob trajectory')
        colorbar
    end
end
