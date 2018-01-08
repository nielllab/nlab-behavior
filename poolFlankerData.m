close all
clear all
[f p] = uigetfile('*.mat','subj data file');
load(fullfile(p,f));

sess=0;
for i=1:length(subjData);
    allResp=[];
    try
        load(fileList{i})
    catch
        try
            fn = [fileList{i}(1:11) '\' fileList{i}(12:end)]
            load(fn);
        catch
            try
                fn = [fileList{i}(1:11) '\' fileList{i}(13:end)]
                load(fn);
            catch
                
                sprintf('couldnt load file %s',fileList{i})
            end
        end
    end
    if length(allResp)>200 & strcmp(subjData{i}.taskFile(1:9),'FullFlank')
        sess =sess+1;
        filenum(sess)=i;
        correct =field2array(allResp,'correct');
        bias = field2array(allResp,'response')>0;
        r= field2array(allResp,'respTime');
        s = field2array(allStop,'stopSecs');
        
        clear label flankResp flankBias biasLower biasUpper respLower respUpper
        
        clear label flankResp flankBias biasLower biasUpper respLower respUpper
        if isfield(stimDetails,'flankContrast')
            flankC = field2array(stimDetails(trialCond),'flankContrast');
            targC = field2array(stimDetails(trialCond),'targContrast');
            fc = unique(flankC);
            tc = unique(targC);
            for f = 1: length(fc);
                for t = 1:length(tc)
                    label{f} = num2str(fc(f));
                    tlabel{t} = num2str(tc(t));
                    use = flankC==fc(f) & targC==tc(t);
                    [mn ci] = binofit( sum(correct(use)),sum(use));
                    flankResp(f,t) = mn; respLower(f,t) = mn-ci(1); respUpper(f,t) = ci(2)-mn;
                    [mn ci] = binofit(sum(bias(use)),sum(use));
                    flankBias(f,t) = mn; biasLower(f,t) = mn-ci(1); biasUpper(f,t) = ci(2)-mn;
                    
                    flankRT(f,t) = median(r(use)); RTerr(f,t)=std(r(use))/sqrt(length(r(use)));
                    
                    
                end
            end
            
            col = 'rgbcmy';
            figure
            for c = 1:length(tc);
                errorbar((1:length(fc))+0.1,flankRT(:,c),RTerr(:,c),[col(c) '-o']);hold on;
            end
            title('RT'); set(gca,'Xtick',1:length(fc));set(gca,'XTickLabel',label); xlabel('contrast'); ylim([0 2]);
            legend(tlabel);
            
            col = 'rgbcmy';
            figure
            for c = 1:length(tc);
                errorbar((1:length(fc))+0.1,flankResp(:,c),respLower(:,c),respUpper(:,c),[col(c) '-o']);hold on;
            end
            title(sprintf('correct sess %d',i)); set(gca,'Xtick',1:length(fc));set(gca,'XTickLabel',label); xlabel('contrast'); ylim([0 1]);
            legend(tlabel);
        end
        
        flankRespAll(:,:,sess) = flankResp; flankBiasAll(:,:,sess)=flankBias; flankRTall(:,:,sess)=flankRT;
        nAll(sess) = length(correct);
        correctAll(sess)=mean(correct); biasAll(sess) = mean(bias); rtAll(sess)= median(r);
    end
end

figure
plot(correctAll,'g'); hold on; plot(biasAll,'b'); ylim([0 1]); plot([1 length(correctAll)],[0.5 0.5],'r:')
legend('correct','bias');
figure
plot(nAll); ylabel('ntrials')

meanResp = nanmean(flankRespAll,3); seResp=nanstd(flankRespAll,[],3)/sqrt(size(flankRespAll,3));
col = 'rgbcmy';
figure
for i = 1:length(tc);
    errorbar((1:length(fc))+0.1,meanResp(:,i),seResp(:,i),[col(i) '-o']);hold on;
end
title('correct'); set(gca,'Xtick',1:length(fc));set(gca,'XTickLabel',label); xlabel('contrast'); ylim([0 1]);
legend(tlabel);


meanRT = nanmedian(flankRTall,3); seRT=nanstd(flankRTall,[],3)/sqrt(size(flankRTall,3));
col = 'rgbcmy';
figure
for i = 1:length(tc);
    errorbar((1:length(fc))+0.1,meanRT(:,i),seRT(:,i),[col(i) '-o']);hold on;
end
title('RT'); set(gca,'Xtick',1:length(fc));set(gca,'XTickLabel',label); xlabel('contrast');
legend(tlabel);


