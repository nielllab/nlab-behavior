addr='0378';
if strcmp(computer,'PCWIN')
    pp.ioObj = io32; status = io32(pp.ioObj);
elseif strcmp(computer,'PCWIN64')
    pp.ioObj = io64; status = io64(pp.ioObj);
end

if status~=0
    status, error('driver installation not successful')
end
pp.addr = hex2dec(addr);
setPP(pp,0);