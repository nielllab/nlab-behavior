function details = doStimPeriod(stimulus,stimDetails,framerate,subj,win)

%%% get time and set flags
start= GetSecs;
done = 0;
t=0; %%% number of frames

xcenter = 1920/2; ycenter = 1080/2;
while ~done
    t=t+1
    if stimDetails.static ==1
        %%% display stimulus image
        tex=Screen('MakeTexture', win, stimulus); Screen('DrawTexture', win, tex);
        vbl = Screen('Flip', win);
        
    elseif stimDetails.static==0
        %%% wait and display next frame
    end
    %pco;
    details.frameT(t) = GetSecs-start;
    %%% get mouse position and reset to center
    [x y] = GetMouse
    SetMouse(xcenter,ycenter);
    if t==1
        details.xpos(t)=0; details.ypos(t) = 0;
    else
        details.xpos(t) = details.xpos(t-1)+x-xcenter;
        details.ypos(t) = details.ypos(t-1)+y-ycenter;
    end
    
    %%% if position has crossed threshold, check to see if it's correct or not
    if abs(details.xpos(t))>subj.respThresh
        responded =1;
        respFrame= t;
        details.respTime = GetSecs-start;
        details.response = sign(details.xpos(t));
        %%% correct response
        if stimDetails.correctResp*sign(details.xpos(t))==1
            details.correct=1;
            io64(ioObj,valveAddr,255);
        else
            details.correct=0;
        end
        done=1;
    end
    %%% check for timeout
    if t>(subj.maxStimduration)*framerate
        done=1;
        details.correct=0;
        details.timeout=1;
        respFrame=t;
    end
    
    c=keyboardCommand(win);
    if strcmp(c,'q')
        details.quit=1;
        return
    end
    
end
done = 0;

details
%%% post-response period - give reward and change screen?

while ~done
    t=t+1;
    %%% correct response
    if details.correct
        %%% show correct image
        tex=Screen('MakeTexture', win, stimulus); Screen('DrawTexture', win, tex);
        Screen('DrawText',win,sprintf('correct'),10,30);
        %%% close valve?
        if t-respFrame>=(subj.rewardDuration)*framerate
          io64(ioObj,valveAddr,0);
        end
        %%% when finished
        if t-respFrame>(subj.correctDuration)*framerate
            done=1;
        end
        
        %%% incorrect response
    elseif ~details.correct
        %%% show error image
        Screen('FillRect',win,255);
        Screen('DrawText',win,sprintf('incorrect'),10,30);
        %%% when finished
        if t-respFrame>(subj.errorDuration)*framerate
            Screen('FillRect',win,128);
            done=1;
        end
    end
    
    Screen('Flip',win);
    
    %%% keep recording mouse position
    details.frameT(t) = GetSecs-start;
    [x y] = GetMouse;
    SetMouse(xcenter,ycenter);
    details.xpos(t) = details.xpos(t-1)+x-xcenter;
    details.ypos(t) = details.ypos(t-1)+y-ycenter;
    
    c=keyboardCommand(win);
    if strcmp(c,'q')
        details.quit=1;
        return
    end
    
end
details.quit=0;
%%% close valve (just in case)



