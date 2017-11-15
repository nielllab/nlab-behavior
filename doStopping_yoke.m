function [details trigT] = doStopping(stopImg,framerate, subj,win,pp,label,pin,trigT, meanStop)

%%% set flags
t= 0;       %%% current frame
stopped =0;  %%% stopped yet?
done=0;      %%% finished stop period

pinDefs; %%% read in pin definitions

%%% center position for optical mosue
xcenter = 1920/2; ycenter = 1080/2;

start = GetSecs

while ~done
    t=t+1;
    
    %%% read and reset mouse
    [x y] = GetMouse;
    SetMouse(xcenter,ycenter);
    details.dx(t) = x-xcenter;
    details.dy(t)=y-ycenter;
    d(t) = sqrt(details.dx(t)^2 + details.dy(t)^2);
    
    %%% set up screen with stopping image (dependent on movement
    %     tex=Screen('MakeTexture', win, stopImg);
    %     Screen('DrawTexture', win, tex);
    %     if d(t)<subj.stopThresh
    %         Screen('DrawText',win,sprintf('stop'),10,30);
    %     else
    %         Screen('DrawText',win,sprintf('move'),10,30);
    %     end
    Screen('DrawText',win,label,10,30);
    Screen('Flip',win);
    trigT= camTrig(pp,pin,trigT);
    %Screen('Close',tex)
    details.frameT(t)=GetSecs;
    %%% check whether the last "duration" frames were less than threshold
    if ~stopped&& rand<(1/(meanStop*60))
        stopped=1;
        stopTime=t;
        details.stopSecs = GetSecs-start;
    end
    
    %%% if no stopReward, we're done
    %%% otherwise, open valve and then wait to close it
    if stopped&~subj.stopReward
        done=1;
    elseif stopped&subj.stopReward&stopTime==t;
        setPPpin(pp,pin.valve,1);
        %display('opened valve')
        openTime= GetSecs;
        t
    elseif stopped & subj.stopReward&(GetSecs-openTime)>subj.stopReward
        setPPpin(pp,pin.valve,0);
        %display('closed valve')
        t
        done=1;
    end
    c=keyboardCommand(win,pp);
    if strcmp(c,'q')
        details.quit=1;
        return
    end
end

details.stopTime= stopTime/framerate;
details.start =start;
details.quit = 0;

