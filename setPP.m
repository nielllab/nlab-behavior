function setPP(pp,data);
if strcmp(computer,'PCWIN')
    io32(pp.ioObj,pp.addr,data);
elseif strcmp(computer,'PCWIN64')
   io64(pp.ioObj,pp.addr,data);
end

%%% note - for widefield behav rigs

%%% 2 = valve
%%% 8 = green LED
%%% 16 = blue
%%% 32 = camera?
