%% testLeeValve

ljasm = NET.addAssembly('LJUDDotNet');
ljudObj = LabJack.LabJackUD.LJUD;

try
    % Read and display the UD version.
    disp(['UD Driver Version = ' num2str(ljudObj.GetDriverVersion())])

    % Open the first found LabJack U3.
    [ljerror, ljhandle] = ljudObj.OpenLabJackS('LJ_dtU3', 'LJ_ctUSB', '0', true, 0);

    % Start by using the pin_configuration_reset IOType so that all pin
    % assignments are in the factory default condition.
    ljudObj.ePutS(ljhandle, 'LJ_ioPIN_CONFIGURATION_RESET', 0, 0, 0);
    disp('LabJack bootup and configuration successful')
catch
    disp('error with LabJack device')
end

again = 1;
while again
    i =input('0 left, 1 right: ');
    d =input('open duration secs : ');
    
    % Set DAC 'i' to 3.0 volts for 'd' secs
    channel = i;
    voltage = 3.0;
    binary = 0;
    ljudObj.eDAC(ljhandle, channel, voltage, binary, 0, 0);
    pause(d)
    ljudObj.eDAC(ljhandle, channel, 0, binary, 0, 0);
    
    repeat = 0;
    repeat = input('repeat? 0=no, 1=yes: ')
    while repeat
        channel = i;
        voltage = 3.0;
        binary = 0;
        ljudObj.eDAC(ljhandle, channel, voltage, binary, 0, 0);
        pause(d)
        ljudObj.eDAC(ljhandle, channel, 0, binary, 0, 0)
        
        repeat = input('repeat? 0=no, 1=yes: ')
    end    
    
    again = input('do another? 0=no, 1=yes: ')
end

disp('evacuation complete')
