function trigT = camTrig(pp,pin,trigT);
%%% every 0.1sec set camera trigger high and record time
global pinState
currentT = GetSecs;
if isempty(trigT) | (currentT- trigT(end)) >(0.1 -0.5/60)
    pinState = pinState+1;
    if mod(pinState,4) ==0
        setPPpin(pp, pin.green,1)
        setPPpin(pp, pin.blue,0)
    else
        setPPpin(pp, pin.green,0)
        setPPpin(pp, pin.blue,1)
    end
    
    trigT = [trigT currentT];
    setPPpin(pp,pin.camtrig,1);
    WaitSecs(0.001);
    setPPpin(pp,pin.camtrig,0);
end
    