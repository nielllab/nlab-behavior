function c = keyboardCommand(win,pp,pin)
%%% check keyboard and take appropriate action
[ keyIsDown, seconds, keyCode ] = KbCheck;
if keyIsDown;
    c = KbName(keyCode);
    if strcmp(c,'o')
          Screen('DrawText',win,sprintf('water'),100,300);
          Screen('Flip',win);
          setPPpin(pp,pin.valve,1);
          WaitSecs(0.5)
           setPPpin(pp,pin.valve,0);
    end
else
    c = '';
end
