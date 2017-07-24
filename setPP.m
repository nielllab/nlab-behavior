function setPP(pp,data);
if strcmp(computer,'PCWIN')
    io32(pp.ioObj,pp.addr,data);
elseif strcmp(computer,'PCWIN64')
   io64(pp.ioObj,pp.addr,data);
end