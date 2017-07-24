function c = keyboardCommand(win,pp)
%%% check keyboard and take appropriate action
if CharAvail;
    c = GetChar;
    if strcmp(c,'o')
          Screen('DrawText',win,sprintf('water'),100,300);
          Screen('Flip',win);
          setPP(pp,255);
          WaitSecs(0.5)
          setPP(pp,0);
    end
else
    c = '';
end
