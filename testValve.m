display('type q to quit')
d =input('open duration secs : ');
w =input('wait duration : ');
n= input('number : ');

setupPP

Screen('Preference', 'SkipSyncTests', 1);
win = Screen('OpenWindow',0,128);
framerate = Screen('FrameRate',win);
ListenChar(2);
for i = 1:n
    c=keyboardCommand(win);
    if strcmp(c,'q')
        display('quitting')
        break
    end
    Screen('DrawText',win,num2str(i),10,30);
    Screen('Flip',win);
    t = GetSecs;
    setPP(pp,255);
    
    while GetSecs-t < d
        Screen('DrawText',win,num2str(i),10,30);
        Screen('Flip',win);
    end
    setPP(pp,0);
    WaitSecs(w);
end
setPP(pp,0);

Priority(0);
ListenChar(1);
Screen('CloseAll');