%% doPokeBehavior
%%%started by Phil P. 12/11/2018, based on doBehavior.m
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

while ~done
    
    trial = trial + 1;
    disp(sprintf('trial %d',trial))
    tic
    
    figure(f1)
    imshow(blankim)
    
    strt=0;
    while ~strt
        [ljerror, chan6] = ljudObj.eDI(ljhandle, 6, 0);
        if chan6==1
            strt=1;
            a = toc;
        else
            continue
        end
    end
    
    trtype = round(rand(1));
    img = stimdata(:,:,trtype+1);
    figure(f1)
    imshow(img)

    tic
    
    corchan = 4+trtype;
    disp(sprintf('corchan = %d',corchan))
    
    trdone = 0;
    while ~trdone
        [ljerror, chan4] = ljudObj.eDI(ljhandle, 4, 0);
        [ljerror, chan5] = ljudObj.eDI(ljhandle, 5, 0);
        
        if chan4==0 & chan5==0 & toc<=subj.maxdur %no pokes, under time lim
            continue
        elseif chan4==1 & corchan==4 %correct left poke
            channel = 0;
            voltage = 3.0;
            binary = 0;
            ljudObj.eDAC(ljhandle, channel, voltage, binary, 0, 0);
            pause(subj.opendur0)
            voltage=0;
            ljudObj.eDAC(ljhandle, channel, voltage, binary, 0, 0);
            trdone = 1;cor=1;b=toc;
        elseif chan5==1 & corchan==5 %correct right poke
            channel = 1;
            voltage = 3.0;
            binary = 0;
            ljudObj.eDAC(ljhandle, channel, voltage, binary, 0, 0);
            pause(subj.opendur1)
            voltage=0;
            ljudObj.eDAC(ljhandle, channel, voltage, binary, 0, 0);
            trdone = 1;cor=1;b=toc;
        else %incorrect poke or no poke
            trdone = 1;cor=0;b=toc;
            figure(f1)
            for fr = 1:5 %do screen flash
                imshow(whim)
                imshow(blim)
            end
            imshow(blankim)
            pause(5)
        end
    end
    trialCond(trial) = trtype;
    allResp(trial) = cor;
    itint(trial) = a;
    trdur(trial) = b;
    disp(sprintf('%s, %0.1fsec to initiate, %0.1fsec to respond, %0.1f%% correct',...
        trlb{cor+1},a,b,100*sum(allResp)/length(allResp)))
    if trial>=10;
        disp(sprintf('%d/last ten correct',sum(allResp(end-9:end))))
    end
    
    save(sessionfile,'allResp','trialCond','itint','trdur','-append');
end

%% plot preliminary results (manually run this after ctl+c exit of behavior
ntrials = 1:length(trialCond);

figure
subplot(2,2,1)
hold on
plot(ntrials(allResp==1),trialCond(allResp==1),'go')
plot(ntrials(allResp==0),trialCond(allResp==0),'ro')
xlabel('trial #')
ylabel('trial type')
legend('correct','incorrect')
axis([0 length(ntrials) 0 1])
set(gca,'ytick',[0 1],'yticklabel',{'left','right'})

subplot(2,2,2)
plot(ntrials,itint,'ko')
xlabel('trial #')
ylabel('intertrial interval (s)')
axis([0 length(ntrials) 0 20])

subplot(2,2,3)
plot(ntrials,trdur,'ko')
xlabel('trial #')
ylabel('trial duration (s)')
axis([0 length(ntrials) 0 20])

subplot(2,2,4)
hold on
for i = 5:length(ntrials)-5
    plot(i,mean(allResp(i-4:i+5)),'ko')
end
plot([0 length(ntrials)],[0.5 0.5],'k:')
axis([0 length(ntrials) 0 1])
xlabel('trial #')
ylabel('avg perf +/- 5 trials')
