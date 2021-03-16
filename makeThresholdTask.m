
%% make movie and task variables for simplest HvV
%%% one location, one spatial frequency, theta =0,pi/2

clear all

%%% set parameters
sfrange = 0.1067;
phaserange = linspace(0, 2*pi,5);
phaserange=phaserange(1:4);
ntheta =2;
nx = 1; ny =1;
% targContrast = [ 0.125 0.25 0.5 1];
targContrast = [ 0.0625 0.25 1];
flankContrast  = [0];
durationRange = [5 15 30 100000];

randomTheta=0;
binarize=1;

%%% set up stimulus size & geometry
xsz = 1920;
ysz = xsz*9/16;
dist = 25;
width = 50;
widthdeg = 2*atand(0.5*width/dist)
degperpix = widthdeg/xsz
sfrange = sfrange*degperpix;

%%% make blocks to put stimuli into
blockwidth = 800*5/7;
xposrange = [0.5 ]*xsz - blockwidth/2
ypos = linspace(1,ysz-1,ny+1);
yposrange = ypos(1:end-1);
yposrange = 0.5*ysz - blockwidth/2;

flankrange = [0.175 0.825]*xsz - blockwidth/2;

%%% create stimulus permutations
thetarange = [0  pi/2 ];
trial=0;
for n= 1:length(thetarange);
    for i = 1:length(xposrange);
        for j = 1:length(yposrange);
            for k = 1:length(sfrange);
                for m= 1:length(phaserange);
                    for a = 1:length(targContrast);
                        for b = 1:length(flankContrast)
                            for d = 1:length(durationRange)
                                trial = trial+1;
                                xpos(trial) = xposrange(i); ypos(trial)=yposrange(j);
                                sf(trial)=sfrange(k);
                                phase(trial) = phaserange(m); theta(trial) = thetarange(n);
                                flankC(trial) = flankContrast(b);
                                targC(trial) = targContrast(a);
                                duration(trial) = durationRange(d);
                                if randomTheta
                                    theta(trial) = rand*2*pi;
                                end
                            end
                        end
                    end
                    
                end
            end
        end
    end
end

%%% set correct responses 1=left -1=right in landscape
correctResp(theta==0) = 1;
correctResp(theta == pi/2) = -1;

%%% make circular mask
[x y] =meshgrid(1:blockwidth,1:blockwidth);
xgrid=(x-mean(x(:))); ygrid=(y-mean(y(:)));
gaussian = sqrt((xgrid.^2 +ygrid.^2))<blockwidth/2;
figure
imagesc(gaussian)

%%% make individual stimuli
stimulus = zeros(xsz,ysz,trial,'uint8')+128;
for tr = 1:trial
    tr
    ph = (x*cos(theta(tr)) + y*sin(theta(tr)))*2*pi*sf(tr) + phase(tr);
    frame = uint8(0.5*255*(targC(tr)*sign(cos(ph)).*gaussian+1));
    stimulus(xpos(tr):xpos(tr)+blockwidth-1, ypos(tr):ypos(tr)+blockwidth-1,tr) = frame;
    flanker = uint8(0.5*255*(flankC(tr)*gaussian+1));
    for i = 1:2
        stimulus(flankrange(i):flankrange(i)+blockwidth-1, ypos(tr):ypos(tr)+blockwidth-1,tr) = flanker;
    end
end
% if binarize
%     stimulus(stimulus>128)=255;
%     stimulus(stimulus<128)=0;
% end

%%% crop, in case it's too large
stimulus = stimulus(1:xsz,1:ysz,:);
figure
for i = 1:trial
    subplot(ceil(sqrt(trial)),ceil(sqrt(trial)),i)
    imshow(stimulus(:,:,i));
    drawnow
end



for i = 1:trial
    stimDetails(i).xpos = xpos(i);
    stimDetails(i).theta = theta(i);
    stimDetails(i).sf = sf(i);
    stimDetails(i).correctResp  = correctResp(i);
    stimDetails(i). phase = phase(i);
    stimDetails(i).static=1;
    stimDetails(i).targContrast = targC(i);
    stimDetails(i).flankContrast = flankC(i);
    stimDetails(i).duration = duration(i)
end


interImg = ones(xsz,ysz)*128;
save thresholdTask interImg stimulus stimDetails xpos ypos correctResp sf phase theta nx ny targC flankC


