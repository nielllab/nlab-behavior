%%% converts behavior stim into a movie to be played by psychstimcontroller
%%% cmn 05/19/19

clear all
load FullFlankerTask  %%% select the behavior protocol you want

reps = 3;  %%% total number of reps of all stim

framerate = 60;
duration = 1;
isi = 0.5

downsamp = 0.125;
img = imresize(stimulus,downsamp); %%% make it smaller
interRep = imresize(interImg,downsamp);
interRep = repmat(interRep,[1 1 isi*framerate]);

n=0;
moviedata = zeros(size(img,1),size(img,2),size(img,3)*framerate*(duration+isi)*reps,'uint8');
size(moviedata)
for r = 1:reps
    r
order = Shuffle(1:length(stimDetails));
    for i = 1:length(order);
        n= n+1;
        moviedata(:,:,(n-1)*framerate*(duration+isi) + (1:framerate*duration)) = repmat(img(:,:,order(i)),[1 1 framerate*duration]);
        moviedata(:,:,(n-1)*framerate*(duration+isi) + framerate*duration + (1:framerate*isi)) = interRep;
        stimInfo(n) = stimDetails(order(i));
    end
end

