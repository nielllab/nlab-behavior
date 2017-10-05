function trigT = camTrig(pp,pin,trigT);
%%% every 0.1sec set camera trigger high and record time
currentT = GetSecs;
if isempty(trigT) | (currentT- trigT(end)) >(0.1 -0.5/60)
    trigT = [trigT currentT];
    setPPpin(pp,pin.camtrig,1);
    WaitSecs(0.001);
    setPPpin(pp,pin.camtrig,0);
end
    