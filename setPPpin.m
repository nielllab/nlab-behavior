function setPP(pp,pin,value);
if strcmp(computer,'PCWIN')
    current =  io32(pp.ioObj,pp.addr);
    update = bitset(current,pin,value);
    io32(pp.ioObj,pp.addr,update);
elseif strcmp(computer,'PCWIN64')
    current =  io64(pp.ioObj,pp.addr);
    update = bitset(current,pin,value);
    io64(pp.ioObj,pp.addr,update);
end

%%% note - for widefield behav rigs

%%% 2 = valve
%%% 8 = green LED
%%% 16 = blue
%%% 32 = camera?
