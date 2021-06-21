function trigT = camTrig(pp,pin,trigT);
%%% every 0.1sec set camera trigger high and record time
global frameCount
currentT = GetSecs;
if isempty(trigT) | (currentT- trigT(end)) >(0.1 -0.5/60)
    
    %%% control LED pints to have 3 blue frames, then one green frame
    frameCount = frameCount+1;
    if mod(frameCount,4) ==0   %%% every 4th frameset green pin on, blue pin off
        setPPpin(pp, pin.green,1)
        setPPpin(pp, pin.blue,0)
    else
        setPPpin(pp, pin.green,0)
        setPPpin(pp, pin.blue,1)
    end
    WaitSecs(0.001);  %%% give 1msec before triggering camera frame (just to be safe)
     
    trigT = [trigT currentT];
    setPPpin(pp,pin.camtrig,1);
    WaitSecs(0.001);
    setPPpin(pp,pin.camtrig,0);
end
    