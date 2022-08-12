%% MakeJump3CamVideo
close all
clear all
FR = 30;
% jumptime = [164 557 1254 1670 1891];
% jumpfrms = jumptime*FR;
jumptime = [4946 16741 37636 50121 56756;
            4947 16735 37659 50084 56718;
            4947 16678 37437 49931 56559];
% Loff = [0 0 -1 -1 -1.1];
% Roff = [0 -2 -6 -6.1 -6.8];
xlims = [0 177; 110 182; 0 162; 4 167; 95 146];
% jumpfrms = [    4920       16710       37620       50100       56730]

frdur = 10;
goback = 5;

[sidevid, pside] = uigetfile('*.avi','side-view video');
[leyevid, pleye] = uigetfile('*.avi','left eye video');
[reyevid, preye] = uigetfile('*.avi','right eye video');

cd(pside)

for cnt=1:length(jumptime)
    [xside, yside, VidSide] = analyzeJumpVid(sidevid,jumptime(1,cnt)-goback*FR,frdur,cnt);
    [xreye, yreye, VidReye] = analyzeJumpVid(reyevid,jumptime(2,cnt)-goback*FR,frdur,cnt);
    [xleye, yleye, VidLeye] = analyzeJumpVid(leyevid,jumptime(3,cnt)-goback*FR,frdur,cnt);
    
%     stfrm = (jumptime(cnt)-goback)*FR;
%     frmrange = stfrm:stfrm+frdur*FR;
    
    figure;set(gcf,'color','w')
    %side cam
    stfrm = (jumptime(1,cnt)-goback*FR);%*FR;
    frmrange = stfrm:stfrm+frdur*FR;
    csvf = dir([sidevid(1:end-4) '*.csv']);
    Pointsxy = csvread(csvf.name,3,0);
    Pointsx = Pointsxy(frmrange,[2:3:end]); Pointsy = Pointsxy(frmrange,[3:3:end]);
    
    cd('extra')
    %left eye
    stfrm = (jumptime(2,cnt)-goback*FR);%*FR;
    frmrange = stfrm:stfrm+frdur*FR;
    csvf = dir([leyevid(1:end-4) '*.csv']);
    Pointsxy = csvread(csvf.name,1,0);
    %     Pointsx = Pointsxy(frmrange,[7:3:end]); Pointsy = Pointsxy(frmrange,[8:3:end]);
    xcent = Pointsxy(frmrange,2);
    ycent = Pointsxy(frmrange,3);
    longax = Pointsxy(frmrange,4);
    longax(longax>50) = interp1(1:length(longax),longax,longax(longax>50));
    shortax = Pointsxy(frmrange,5);
    tilt = Pointsxy(frmrange,6);
    ynose = -Pointsy(:,1);
    xnose = -Pointsx(:,1);

    subplot(1,3,1)
    hold on
    plot((ynose-mean(ynose))/mean(ynose)+0.25,'k')
    plot((xnose-mean(xnose))/mean(xnose),'color',[0.7 0.7 0.7])
    plot((xcent-mean(xcent))/mean(xcent)+0.5,'b')
    
    subplot(1,3,2)
    hold on
    plot((ynose-mean(ynose))/mean(ynose)+0.25,'k')
    plot((xnose-mean(xnose))/mean(xnose),'color',[0.7 0.7 0.7])
    plot((ycent-mean(ycent))/mean(ycent)+0.5,'b')
    
    subplot(1,3,3)
    hold on
    plot((ynose-mean(ynose))/mean(ynose)+0.25,'k')
    plot((xnose-mean(xnose))/mean(xnose),'color',[0.7 0.7 0.7])
    plot((longax-mean(longax))/mean(longax)+0.5,'b')
    
    %right eye
    stfrm = (jumptime(3,cnt)-goback*FR);%*FR;
    frmrange = stfrm:stfrm+frdur*FR;
    csvf = dir([reyevid(1:end-4) '*.csv']);
    Pointsxy = csvread(csvf.name,1,0);
    xcent = Pointsxy(frmrange,2);
    ycent = Pointsxy(frmrange,3);
    longax = Pointsxy(frmrange,4);
    longax(longax>50) = interp1(1:length(longax),longax,longax(longax>50));
    shortax = Pointsxy(frmrange,5);
    tilt = Pointsxy(frmrange,6);

    subplot(1,3,1)
    hold on
    plot((xcent-mean(xcent))/mean(xcent)+0.75,'r')
    title('xcent')
    xlim(xlims(cnt,:))
    ylim([0,1])
    set(gca,'xtick',96:15:146,'xticklabel',-2000:500:0,'ytick',0:0.25:1)
    
    subplot(1,3,2)
    hold on
    plot((ycent-mean(ycent))/mean(ycent)+0.75,'r')
    title('ycent')
    xlim(xlims(cnt,:))
    ylim([0,1])
    set(gca,'xtick',96:15:146,'xticklabel',-2000:500:0,'ytick',0:0.25:1)
    
    subplot(1,3,3)
    hold on
    plot((longax-mean(longax))/mean(longax)+0.75,'r')
    legend('Ynose','Xnose','Leye','Reye')
    title('pupil diam')
    xlim(xlims(cnt,:))
    ylim([0,1])
    set(gca,'xtick',96:15:146,'xticklabel',-2000:500:0,'ytick',0:0.25:1)
    
    cd ..

    %%make video
    figure;
    axis tight manual
    ax = gca;
    ax.NextPlot = 'replaceChildren';
    vname = sprintf('ThreeViewsJump_%d.mp4',cnt);
    vidfile = VideoWriter(vname,'MPEG-4');
    vidfile.FrameRate = 30/4;
    open(vidfile);
    for  v=1:size(VidSide,3)
        p(1) = subplot(2,2,[1 2]);
        imagesc(VidSide(:,:,v)); colormap gray; hold on; axis equal off;
%         scatter(xside(v,:),yside(v,:),100,'.r'); hold off; % Uncomment if to plot DLC points too

        p(2) = subplot(2,2,3);
        imagesc(VidLeye(:,:,v)); colormap gray; hold on; axis equal off;
%         scatter(xleye(v,:),yleye(v,:),100,'.r'); hold off; % Uncomment if to plot DLC points too

        p(3) = subplot(2,2,4);
        imagesc(VidReye(:,:,v)); colormap gray; hold on; axis equal off;
%         scatter(xreye(v,:),yreye(v,:),100,'.r'); hold off; % Uncomment if to plot DLC points too

        drawnow limitrate;
        F(v) = getframe(gcf); 
        writeVideo(vidfile,F(v));
    end
    close(vidfile)
    sprintf('done %d of %d',cnt,length(jumptime))
end

%% plot data only
% [sidevid, pside] = uigetfile('*.avi','side-view video');
% [leyevid, pleye] = uigetfile('*.avi','left eye video');
% [reyevid, preye] = uigetfile('*.avi','right eye video');
% 
% csvf = dir([sidevid(1:end-4) '*.csv']);
% Pointsxy = csvread(csvf.name,3,0);
% Pointsx = Pointsxy(:,[2:3:end]); Pointsy = Pointsxy(:,[3:3:end]);
% ynose = -Pointsy(:,1);
% xnose = -Pointsx(:,1);
%     
% cd('extra')
% lcsvf = dir([leyevid(1:end-4) '*.csv']);
% rcsvf = dir([reyevid(1:end-4) '*.csv']);
% lPointsxy = csvread(lcsvf.name,1,0);
% llongax = lPointsxy(:,4);
% rPointsxy = csvread(rcsvf.name,1,0);
% rlongax = rPointsxy(:,4);
% 
% figure;
% hold on
% plot((llongax-mean(llongax))/mean(llongax),'b')
% plot((rlongax-mean(rlongax))/mean(rlongax),'r')
% plot((xnose-mean(xnose))/mean(xnose),'color',[0.5 0.5 0.5])
% plot((ynose-mean(ynose))/mean(ynose),'k')
% 
% cd ..