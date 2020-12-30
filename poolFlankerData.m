close all
clear all
useBatch = 1;

if useBatch
    batchFlankBehav_bk
    nsubj = length(files);
else
    nsubj = 1;
end

for subj = 1:nsubj
    subj
   
    
    if useBatch
        sfile =files{subj}.subjfile;
        p = [datapath filesep files{subj}.subjdir];
        
    else
        [sfile p] = uigetfile('*.mat','subj data file');
    end
    
    load(fullfile(p,sfile));
    
    clear flankRespAll flankBiasAll flankRTall nAll nAllSessions correctAll biasAll rtAll
    
    sess=0;
    for i=1:length(subjData);
        i
        allResp=[];
        
        try
            filename = fileList{i};
            if ismac
                filename(filename=='\') = '/';
            end
            [pn fn] = fileparts(filename);
            
            load(fullfile(p,fn)); %%%% use path from subject file, since path in subjFile isn't always accurate
            goodData = 1;
            nAllSessions(i) = length(allResp);
        catch
            goodData = 0;
        end
        
        subjData{i}.taskFile
        
        if goodData & length(allResp)>200 & strcmp(subjData{i}.taskFile(1:9),'FullFlank') & mean(field2array(allResp,'correct'))>0.7
            sess =sess+1;
            filenum(sess)=i;
            correct =field2array(allResp,'correct');
            bias = field2array(allResp,'response')>0;
            r= field2array(allResp,'respTime');
            s = field2array(allStop,'stopSecs');
            
%             dur = 50
%             figure
%             data = conv(correct,ones(dur,1),'valid')/dur;
%             plot(1:length(data), data); ylim([0 1]);
%             hold on; plot([1 length(data)], [0.5 0.5]);
            
            
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
                
                %                  col = 'rgbcmy';
                %                 figure
                %                 for c = 1:length(tc);
                %                     errorbar((1:length(fc))+0.1,flankRT(:,c),RTerr(:,c),[col(c) '-o']);hold on;
                %                 end
                %                 title('RT'); set(gca,'Xtick',1:length(fc));set(gca,'XTickLabel',label); xlabel('contrast'); ylim([0 2]);
                %                 legend(tlabel);
                %
                %                 col = 'rgbcmy';
                %                 figure
                %                 for c = 1:length(tc);
                %                     errorbar((1:length(fc))+0.1,flankResp(:,c),respLower(:,c),respUpper(:,c),[col(c) '-o']);hold on;
                %                 end
                %                 title(sprintf('correct sess %d',i)); set(gca,'Xtick',1:length(fc));set(gca,'XTickLabel',label); xlabel('contrast'); ylim([0 1]);
                %                 legend(tlabel);
            end
            
            flankRespAll(:,:,sess) = flankResp; flankBiasAll(:,:,sess)=flankBias; flankRTall(:,:,sess)=flankRT;
            nAll(sess) = length(correct);
            correctAll(sess)=mean(correct); biasAll(sess) = mean(bias); rtAll(sess)= median(r);
        end
    end
    
    figure
    plot(correctAll,'g'); hold on; plot(biasAll,'b'); ylim([0 1]); plot([1 length(correctAll)],[0.5 0.5],'r:')
    legend('correct','bias');
%     figure
%     plot(nAll); ylabel('ntrials in used sessions')
%     figure
%     plot(nAllSessions); ylabel('ntrials in all sessions')
    
    meanResp = nanmean(flankRespAll,3); seResp=nanstd(flankRespAll,[],3)/sqrt(size(flankRespAll,3));
    col = 'rgbcmy';
    figure
    for i = 1:length(tc);
        errorbar((1:length(fc))+0.1,meanResp(:,i),seResp(:,i),[col(i) '-o']);hold on;
    end
    title(sprintf('%s correct',subj.name)); set(gca,'Xtick',1:length(fc));set(gca,'XTickLabel',label); xlabel('contrast'); ylim([0 1]);
    legend(tlabel);
    
    saveas(gcf,fullfile(p, [sfile(1:end-8) 'correct']),'png')
    
    meanRT = nanmedian(flankRTall,3); seRT=nanstd(flankRTall,[],3)/sqrt(size(flankRTall,3));
%     col = 'rgbcmy';
%     figure
%     for i = 1:length(tc);
%         errorbar((1:length(fc))+0.1,meanRT(:,i),seRT(:,i),[col(i) '-o']);hold on;
%     end
%    title(sprintf('%s RT',subj.name)); set(gca,'Xtick',1:length(fc));set(gca,'XTickLabel',label); xlabel('contrast');
%     legend(tlabel);
%     saveas(gcf,fullfile(p, [sfile(1:end-8) 'RT']),'png')
    
    
    save(fullfile(p, [sfile(1:end-8) 'out']))
    
end

if useBatch
    clear meanRespAll meanRTAll
    for s = 1:length(files);
        f =files{s}.subjfile;
        p = [datapath filesep files{s}.subjdir];
        load(fullfile(p,  [f(1:end-8) 'out']),'meanResp','meanRT');
        meanRespAll(:,:,s) = meanResp;
        meanRTAll(:,:,s) = meanRT;
    end
    
    meanResp = nanmean(meanRespAll,3); seResp=nanstd(meanRespAll,[],3)/sqrt(size(flankRespAll,3));
    col = 'rgbcmy';
    figure
    for i = 1:length(tc);
        errorbar((1:length(fc))-0.025*(i-2),meanResp(:,i),seResp(:,i),[col(i) '-o']);hold on;
    end
    title('correct'); set(gca,'Xtick',1:length(fc));set(gca,'XTickLabel',label); xlabel('contrast'); ylim([0 1]);
    legend(tlabel);
    ylim([0.4 1]); plot([1 5], [0.5 0.5],':')
    
    
    meanRT = nanmean(meanRTAll,3); seRT=nanstd(meanRTAll,[],3)/sqrt(size(meanRTAll,3));
    col = 'rgbcmy';
    figure
    for i = 1:length(tc);
        errorbar((1:length(fc))+0.1,meanRT(:,i),seRT(:,i),[col(i) '-o']);hold on;
    end
    title('RT'); set(gca,'Xtick',1:length(fc));set(gca,'XTickLabel',label); xlabel('contrast');
    legend(tlabel);
    
end


figure
errorbar(1:3, meanResp(3,:), seResp(3,:)); ylim([0.5 1])
figure; hold on
for i = 3:-1:1;
    data = [0  mean(meanResp([2 4],i))- meanResp(3,i)  mean(meanResp([1 5],i))-meanResp(3,i)];
    err = [0 mean(seResp([2 4],i)) mean(seResp([1 5],i))];
    xpos = [1 (2:3)-0.025*(i-2)];
    errorbar (xpos, data, err,[col(i) '-o']);
end

legend({'1','0.25','0.0625'})
xlabel('distractor contrast'); ylabel('relative performance')
set(gca,'Xtick',[1 2 3]); set(gca,'Xticklabel',{'0', '0.25', '1'})
xlim([0.75 3.25])

figure
errorbar(1:3, meanResp(3,:), seResp(3,:)); ylim([0.5 1])
figure; hold on
for i = 3:-1:1;
    data = [meanResp(3,i)  mean(meanResp([2 4],i))  mean(meanResp([1 5],i))];
    err = [seResp(3,i)  mean(seResp([2 4],i)) mean(seResp([1 5],i))];
    xpos = [ (1:3)-0.025*(i-2)];
    errorbar (xpos, data, err,[ '-o']);
end
legend({'1','0.25','0.0625'})
xlabel('distractor contrast'); ylabel('correct')
set(gca,'Xtick',[1 2 3]); set(gca,'Xticklabel',{'0', '0.25', '1'})
xlim([0.75 3.25]); ylim([0.45 1])
plot([1 3],[0.5 0.5],':')



figure; hold on
for i = 1:3;
    data = meanResp(:,i) - meanResp(3,i);

    errorbar (1:5, data', seResp(:,i));
end