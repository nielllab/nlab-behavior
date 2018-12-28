%% doPokeShaping
%%%started by Phil P. 12/28/2018, based on PokeBehavior.m
%%%used to train mice on poking
%%%rewards for a left or right poke and displays stimulus to be rewarded in
%%%the future (big or small) at that poke

close all
clear all

%%% execute subject file to get parameters
cd('C:\Users\nlab\Desktop\GitHubCode\nlab-behavior\subjects\phil2poke')
[f,p] = uigetfile('*.m','subject file');
run(fullfile(p,f));

%%% load in taskFile, contains stimulus and details
load(subj.taskFile);

%%% save out session information
sessionfile = [subj.dataLocation '\' datestr(now,30) '_' num2str(sessions)];
fileList{sessions} = sessionfile;
save(subj.subjFile,'fileList','-append')
save(sessionfile,'sessions','subj');

%% %Ch0 = left solenoid, Ch1 = right solenoid %Ch4 = left poke, Ch5 = right, Ch6 = middle

% Make the UD .NET assembly visible in MATLAB.
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

if strcmp(subj.bigsmall,'big')
    disp('big rewarded')
    big=squeeze(stimdata(:,:,1));small=squeeze(stimdata(:,:,2));
    stimdata(:,:,1)=small;stimdata(:,:,2)=big;
elseif strcmp(subj.bigsmall,'small')
    disp('small rewarded')
else
    disp('check subj.bigsmall in subject file')
    return
end

trlb = {'incorrect','correct'};
done = 0;
trial = 0;
blankim = ones(size(stimdata,1),size(stimdata,2))*128; %ITI screen
whim = zeros(size(stimdata,1),size(stimdata,2)); %white for flash
blim = ones(size(stimdata,1),size(stimdata,2))*256; %black for flash

f1 = figure('units','normalized','outerposition',[0 0 1 1])
imshow(blankim)
trtype=0; %start w/right poke rewarded

while ~done
    
    trial = trial + 1;
    disp(sprintf('trial %d',trial))
    
    figure(f1)
    imshow(blankim)
    
    if trtype
        img = stimdata(:,:,1);
        figure(f1);imshow(img)
    else
        img = stimdata(:,:,2);
        figure(f1);imshow(img)
    end
    
    tic
    trdone = 0;
    while ~trdone
        [ljerror, chan4] = ljudObj.eDI(ljhandle, 4, 0);
        [ljerror, chan5] = ljudObj.eDI(ljhandle, 5, 0);
        
        if chan4==0 & chan5==0 %no pokes
            continue
        elseif chan4==1 && trtype %left poke
            channel = 0;
            voltage = 3.0;
            binary = 0;
            ljudObj.eDAC(ljhandle, channel, voltage, binary, 0, 0);
            pause(subj.opendur0)
            voltage=0;
            ljudObj.eDAC(ljhandle, channel, voltage, binary, 0, 0);
            trdone = 1;b=toc;trtype=0;
        elseif chan5==1 && ~trtype %right poke
            channel = 1;
            voltage = 3.0;
            binary = 0;
            ljudObj.eDAC(ljhandle, channel, voltage, binary, 0, 0);
            pause(subj.opendur1)
            voltage=0;
            ljudObj.eDAC(ljhandle, channel, voltage, binary, 0, 0);
            trdone = 1;b=toc;trtype=1;
        end
    end
    pause(2)
    imshow(blankim)
    pause(2)
    
    trialCond(trial) = trtype;
    trdur(trial) = b;
    disp(sprintf('%0.1fsec to respond',...
        b))
        
    save(sessionfile,'trialCond','trdur','-append');
end

%% plot preliminary results (manually run this after ctl+c exit of behavior
ntrials = 1:length(trialCond);

figure
plot(ntrials,trdur,'ko')
xlabel('trial #')
ylabel('trial duration (s)')
axis([0 length(ntrials) 0 10])
