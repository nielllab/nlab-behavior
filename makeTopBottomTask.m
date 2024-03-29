%%% make movie and task variables for simplest HvV
%%% one location, one spatial frequency, theta =0,pi/2

clear all

%%% set parameters
sfrange = 0.1067;
phaserange = linspace(0, 2*pi,9);
phaserange=phaserange(1:8);
ntheta =2;
nx = 2; ny =1;

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
xposrange = [0.33 0.66]*xsz - blockwidth/2
ypos = linspace(1,ysz-1,ny+1);
yposrange = ypos(1:end-1);
yposrange = 0.5*ysz - blockwidth/2;

%%% create stimulus permutations
thetarange = [0  pi/2 ];
trial=0;
for n= 1:length(thetarange);
    for i = 1:length(xposrange);
        for j = 1:length(yposrange);
            for k = 1:length(sfrange);
                for m= 1:length(phaserange);
                    
                    trial = trial+1;
                    xpos(trial) = xposrange(i); ypos(trial)=yposrange(j);
                    sf(trial)=sfrange(k);
                    phase(trial) = phaserange(m); theta(trial) = thetarange(n);
                    if randomTheta
                        theta(trial) = rand*2*pi;
                    end
                end
            end
        end
    end
end

%%% set correct responses
correctResp(xpos==xposrange(1)) = -1;
correctResp(xpos==xposrange(2)) =1;

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
    frame = uint8(0.5*255*(cos(ph).*gaussian+1));
    stimulus(xpos(tr):xpos(tr)+blockwidth-1, ypos(tr):ypos(tr)+blockwidth-1,tr) = frame;
end
if binarize
    stimulus(stimulus>128)=255;
    stimulus(stimulus<128)=0;
end

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
end


interImg = ones(xsz,ysz)*128;
save TopBottomTask interImg stimulus stimDetails xpos ypos correctResp sf phase theta nx ny


