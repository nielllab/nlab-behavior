function c = keyboardCommand(win)
%%% check keyboard and take appropriate action
if CharAvail;
    c = GetChar;
    if strcmp(c,'o')
          Screen('DrawText',win,sprintf('water'),10,30);
          Screen('Flip',win);
          WaitSecs(0.5)
    end
else
    c = '';
end
