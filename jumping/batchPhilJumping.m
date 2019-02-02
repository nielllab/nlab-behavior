%% batchPhilJumping contains data from jumping experiments
%jumptime in seconds

datapathname = '\\langevin\backup\Phil\Behavior\K99 prelim\';
outpathname = '\\langevin\backup\Phil\Behavior\K99 prelim\';

datapathname = 'G:\Phil_Jumping_Behavior\';
outpathname = 'G:\Phil_Jumping_Behavior\';

trialdata = {};

%% animal #1
trialdata(1).name = 'G6H13p4TT';
n=0;

% n=n+1;
% trialdata(1).expt(n).date = '012119';
% trialdata(1).expt(n).trials = 30;
% trialdata(1).expt(n).platform = [1 2 3 2 3 1 3 2 2 1 3 3 1 2 3 1 2 2 1 1 3 3 1 2 3 1 2 2 2 2];
% trialdata(1).expt(n).distance = [5 8 4 6 8 7 7 6 9 8 3 5 8 10 10 10 7 5 6 4 9 6 11 11 11 3 3 4 5 11];
% trialdata(1).expt(n).success = [1 0 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 0 1 1 0 1 1 1 1 1 1];
% trialdata(1).expt(n).jumptime = {};
% trialdata(1).expt(n).vidnames = {''};
% 
% n=n+1;
% trialdata(1).expt(n).date = '012219';
% trialdata(1).expt(n).trials = 31;
% trialdata(1).expt(n).platform = [2 1 3 2 1 2 3 3 2 1 3 2 3 2 1 1 2 1 3 2 3 1 3 1 2 2 2 2 3 1 2];
% trialdata(1).expt(n).distance = [4 3 5 4 4 5 6 4 6 7 7 8 3 3 5 6 7 8 8 10 10 9 9 10 10 9 10 10 11 11 11];
% trialdata(1).expt(n).success = [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 0 1 0 1 1 1 1];
% trialdata(1).expt(n).jumptime = {};
% trialdata(1).expt(n).vidnames = {'012219_G6H13p4TT.MOV'};

n=n+1;
trialdata(1).expt(n).date = '012319';
trialdata(1).expt(n).trials = 37;
trialdata(1).expt(n).platform = [1 2 3 1 2 3 1 2 1 2 2 3 3 3 2 1 2 1 3 1 2 2 1 1 3 3 3 2 1 3 3 2 1 3 1 2 3];
trialdata(1).expt(n).distance = [3 2 3 4 5 4 5 6 6 4 3 4 5 8 8 7 7 9 9 8 10 9 11 10 10 7 11 11 11 11 11 8 9 9 8 9 8];
trialdata(1).expt(n).success = [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 0 1 1 0 1 1 1 1 1 1 1];
trialdata(1).expt(n).jumptime = {[37 79 104 131 157 190 220 246 285 319 342 369 400 450 487 553 590 621 672 712 742 775 827 857 898 941 1001 1019 1078 1135 1149 1213 1260 1296 1362 1449 1638]};
trialdata(1).expt(n).vidnames = {'012319_G6H13p4TT.MOV'};

% n=n+1; % first two trials not recorded so left them out
% trialdata(1).expt(n).date = '012419';
% trialdata(1).expt(n).trials = 47;
% trialdata(1).expt(n).platform = [3 3 2 1 2 1 3 3 1 2 3 2 1 2 2 1 1 3 2 3 1 3 3 3 2 1 3 2 3 2 2 3 3 1 2 3 3 1 2 3 2 1 3 1 1 2 3];
% trialdata(1).expt(n).distance = [5 3 6 5 5 4 7 6 6 3 4 8 8 8 9 7 9 9 7 11 10 11 9 8 11 11 11 11 10 10 11 11 11 8 9 8 8 3 4 5 8 9 10 10 10 10 10];
% trialdata(1).expt(n).success = [1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 0 1 0 1 0 1 1 0 1 0 0 1 1 1 0 1 1 1 0 1 1 1 1 1 1 0 0 1 1 1];
% trialdata(1).expt(n).jumptime = {};
% trialdata(1).expt(n).vidnames = {'012419_G6H13p4TT.MOV'};
% 
% n=n+1;
% trialdata(1).expt(n).date = '012519';
% trialdata(1).expt(n).trials = 45;
% trialdata(1).expt(n).platform = [3 2 3 1 1 3 1 2 2 1 3 3 2 1 2 1 3 1 2 2 3 1 2 3 1 2 3 1 3 2 3 2 1 2 3 1 2 3 1 1 3 3 2 2 1];
% trialdata(1).expt(n).distance = [4 3 3 5 3 5 4 4 5 6 6 7 6 7 7 8 8 9 8 9 9 10 10 10 11 11 10 11 11 11 11 8 9 6 10 9 9 7 10 7 6 9 10 10 8];
% trialdata(1).expt(n).success = [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 1 1 1 0 0 1 1 1 1 1 1 1 1 0 1 1 1 0 1 1];
% trialdata(1).expt(n).jumptime = {};
% trialdata(1).expt(n).vidnames = {'012519_G6H13p4TT.MOV'};
% 
% n=n+1;
% trialdata(1).expt(n).date = '012619';
% trialdata(1).expt(n).trials = 33;
% trialdata(1).expt(n).platform = [2 3 1 2 1 3 2 1 1 2 3 2 1 3 2 3 1 3 1 1 3 3 2 3 2 2 1 2 1 2 2 3 2];
% trialdata(1).expt(n).distance = [3 4 3 5 5 3 4 4 7 6 5 8 6 6 7 7 8 10 11 9 8 9 10 10 9 9 10 11 11 11 10 11 11];
% trialdata(1).expt(n).success = [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 1 1 1 0 1 0 1 1 0 1 0 1 1 1];
% trialdata(1).expt(n).jumptime = {};
% trialdata(1).expt(n).vidnames = {'012619_G6H13p4TT.MOV'};
% 
% n=n+1;
% trialdata(1).expt(n).date = '012719';
% trialdata(1).expt(n).trials = 29;
% trialdata(1).expt(n).platform = [3 2 1 3 1 2 3 2 3 1 1 2 3 1 1 2 3 3 2 3 2 1 3 2 1 1 2 2 1];
% trialdata(1).expt(n).distance = [4 4 5 3 3 5 6 7 8 6 7 6 7 9 8 9 5 10 8 9 3 11 11 10 11 10 11 11 4];
% trialdata(1).expt(n).success = [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 0 1 1];
% trialdata(1).expt(n).jumptime = {};
% trialdata(1).expt(n).vidnames = {'012719_G6H13p4TT.MOV'};
% 
% % n=n+1;
% % trialdata(1).expt(n).date = '012819';
% % trialdata(1).expt(n).trials = 29;
% % trialdata(1).expt(n).platform = [2 3 1 2 1 3 1 2 1 3 2 2 1 3 2 1 3 2 3 1 2 3 1 3 2 3 1 2 1];
% % trialdata(1).expt(n).distance = [3 4 5 4 3 6 7 5 4 3 6 7 8 5 10 11 8 9 11 6 8 10 9 7 5 9 10 11 12];
% % trialdata(1).expt(n).success = [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1];
% % trialdata(1).expt(n).jumptime = {};
% % trialdata(1).expt(n).vidnames = {'012819_G6H13p4TT.avi'};
% % 
% % n=n+1;
% % trialdata(1).expt(n).date = '012919';
% % trialdata(1).expt(n).trials = 37;
% % trialdata(1).expt(n).platform = [2 1 3 3 1 2 2 3 1 2 2 2 3 3 2 1 3 1 3 3 2 3 3 2 1 1 1 2 3 1 2 2 2 1 3 3 3];
% % trialdata(1).expt(n).distance = [8 6 3 7 5 7 4 9 8 6 5 10 11 9 9 10 11 11 6 8 3 10 5 9 4 7 9 11 4 3 11 5 6 8 9 4 7];
% % trialdata(1).expt(n).success = [1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1];
% % trialdata(1).expt(n).jumptime = {};
% % trialdata(1).expt(n).vidnames = {'012919_G6H13p4TTa.avi','012919_G6H13p4TTb.avi','012919_G6H13p4TTc.avi'};
% 
% %% animal #2
% trialdata(2).name = 'G6H13p4LT';
% n=0;
% 
% n=n+1;
% trialdata(2).expt(n).date = '012119';
% trialdata(2).expt(n).trials = 25;
% trialdata(2).expt(n).platform = [3 2 1 3 2 3 1 2 3 1 2 3 3 1 2 2 1 1 2 1 3 2 3 9 9];
% trialdata(2).expt(n).distance = [2 2 2 3 4 5 5 6 7 7 8 6 4 6 5 3 3 4 7 8 8 8 9 9 9];
% trialdata(2).expt(n).success = [1 1 1 1 1 1 1 1 1 1 0 0 1 1 1 1 1 1 1 1 0 1 0 1 1];
% trialdata(2).expt(n).jumptime = {};
% trialdata(2).expt(n).vidnames = {''};
% 
% n=n+1;
% trialdata(2).expt(n).date = '012219';
% trialdata(2).expt(n).trials = 25;
% trialdata(2).expt(n).platform = [2 1 3 2 1 2 3 1 3 2 1 3 3 2 2 1 1 3 3 3 2 2 1 3 2];
% trialdata(2).expt(n).distance = [3 3 4 5 4 4 3 6 6 8 7 8 7 7 6 8 5 8 5 9 9 9 9 10 10];
% trialdata(2).expt(n).success = [1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 0 1 1 1 1];
% trialdata(2).expt(n).jumptime = {};
% trialdata(2).expt(n).vidnames = {'012219_G6H13p4LTa.MOV','012219_G6H13p4LTa.MOV'};
% 
% n=n+1;
% trialdata(2).expt(n).date = '012319';
% trialdata(2).expt(n).trials = 27;
% trialdata(2).expt(n).platform = [3 2 1 3 2 1 1 3 2 3 3 1 2 3 2 1 3 2 1 3 2 3 1 3 1 2 3];
% trialdata(2).expt(n).distance = [3 3 3 5 5 4 5 6 4 5 4 7 6 8 7 6 8 9 8 7 8 7 9 9 10 10 10];
% trialdata(2).expt(n).success = [1 1 1 0 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 0 1 1 1 1 1 0 1];
% trialdata(2).expt(n).jumptime = {};
% trialdata(2).expt(n).vidnames = {'012319_G6H13p4LT.MOV'};
% 
% n=n+1; %last trial not recorded
% trialdata(2).expt(n).date = '012419';
% trialdata(2).expt(n).trials = 31;
% trialdata(2).expt(n).platform = [2 3 1 3 1 2 3 1 2 3 1 2 2 1 3 1 2 1 2 3 1 3 1 3 3 2 3 1 1 2 1];
% trialdata(2).expt(n).distance = [3 3 4 5 3 4 4 6 5 6 5 6 8 7 8 8 9 8 7 7 8 9 9 8 9 10 10 10 10 11 11];
% trialdata(2).expt(n).success = [1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 1 0 1 1 1 0 1 1 1 1 1 0 1 0 1];
% trialdata(2).expt(n).jumptime = {};
% trialdata(2).expt(n).vidnames = {'012419_G6H13p4LTa.MOV','012419_G6H13p4LTb.MOV'};
% 
% n=n+1;
% trialdata(2).expt(n).date = '012519';
% trialdata(2).expt(n).trials = 20;
% trialdata(2).expt(n).platform = [1 3 1 3 2 1 2 2 3 1 2 3 2 1 3 3 1 2 1 3];
% trialdata(2).expt(n).distance = [2 3 3 3 3 4 5 5 4 5 4 6 6 6 5 8 7 7 8 7];
% trialdata(2).expt(n).success = [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1];
% trialdata(2).expt(n).jumptime = {};
% trialdata(2).expt(n).vidnames = {'012519_G6H13p4LT.MOVa','012519_G6H13p4LTb.MOV'};
% 
% n=n+1;
% trialdata(2).expt(n).date = '012619';
% trialdata(2).expt(n).trials = 32;
% trialdata(2).expt(n).platform = [2 1 3 2 3 1 2 3 3 1 2 2 3 1 2 2 1 3 1 2 2 3 1 3 2 3 2 2 1 2 3 1];
% trialdata(2).expt(n).distance = [3 4 3 5 6 6 4 5 8 7 7 6 8 9 10 8 5 7 10 10 10 4 8 9 10 10 9 10 3 11 11 11];
% trialdata(2).expt(n).success = [1 1 1 1 1 1 1 1 0 1 1 1 1 1 0 1 1 1 1 0 0 1 1 1 0 1 1 1 1 1 1 1];
% trialdata(2).expt(n).jumptime = {};
% trialdata(2).expt(n).vidnames = {'012619_G6H13p4LT.MOV'};
% 
% n=n+1;
% trialdata(2).expt(n).date = '012719';
% trialdata(2).expt(n).trials = 29;
% trialdata(2).expt(n).platform = [2 3 2 1 3 2 1 2 1 3 2 1 3 3 2 2 3 2 1 3 1 2 1 3 2 3 1 3 1];
% trialdata(2).expt(n).distance = [3 4 4 4 6 6 5 5 7 8 7 8 9 7 9 8 10 11 9 11 11 10 6 11 10 5 10 3 3];
% trialdata(2).expt(n).success = [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 0 1 1 1 1 1 1 1];
% trialdata(2).expt(n).jumptime = {};
% trialdata(2).expt(n).vidnames = {'012719_G6H13p4LT.MOV'};
% 
% % n=n+1;
% % trialdata(2).expt(n).date = '012819';
% % trialdata(2).expt(n).trials = 32;
% % trialdata(2).expt(n).platform = [1 3 2 3 2 1 2 1 2 3 2 1 2 2 1 3 2 1 1 3 2 3 3 1 1 3 2 2 1 3 3 2];
% % trialdata(2).expt(n).distance = [3 4 5 8 6 5 8 7 4 6 10 8 5 9 9 7 7 11 6 10 11 9 11 10 4 10 3 11 10 5 3 11];
% % trialdata(2).expt(n).success = [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 0 1 1 1 0 1 1 1 1];
% % trialdata(2).expt(n).jumptime = {};
% % trialdata(2).expt(n).vidnames = {'012819_G6H13p4LT.avi'};
% % 
% % n=n+1;
% % trialdata(1).expt(n).date = '012919';
% % trialdata(1).expt(n).trials = 29;
% % trialdata(1).expt(n).platform = [2 3 1 2 1 3 2 1 3 2 1 2 3 2 1 3 1 2 3 2 2 3 3 1 1 2 3 2 1];
% % trialdata(1).expt(n).distance = [3 4 5 8 6 5 8 7 4 6 10 8 5 9 9 7 7 11 6 10 11 9 11 10 4 10 3 11 10 5 3 11];
% % trialdata(1).expt(n).success = [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 0 1 1 1 1 1 1 1];
% % trialdata(1).expt(n).jumptime = {};
% % trialdata(1).expt(n).vidnames = {'012919_G6H13p4LTa.avi','012919_G6H13p4LTb.avi'};
% 
% %% animal #3
% trialdata(3).name = 'G6H13p4RT';
% n=0;
% 
% n=n+1;
% trialdata(3).expt(n).date = '012619';
% trialdata(3).expt(n).trials = 33;
% trialdata(3).expt(n).platform = [3 1 2 3 2 3 1 3 2 1 3 1 3 2 1 2 3 2 1 1 1 3 2 2 1 3 2 1 1 3 3 1 2];
% trialdata(3).expt(n).distance = [3 3 3 4 3 3 3 4 5 6 5 5 6 7 8 6 7 8 7 4 9 8 9 4 9 9 10 11 10 10 11 11 11];
% trialdata(3).expt(n).success = [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 0 1 1 1 1 1];
% trialdata(3).expt(n).jumptime = {};
% trialdata(3).expt(n).vidnames = {'012619_G6H13p4RTa.MOV','012619_G6H13p4RTb.MOV'};
% 
% trialdata(3).expt(n).trials = 29;
% trialdata(3).expt(n).platform = [2 3 1 2 1 2 1 3 2 3 1 1 2 3 2 3 1 3 2 2 2 3 3 1 1 2 1 3 1];
% trialdata(3).expt(n).distance = [3 4 6 6 7 5 5 6 7 8 9 8 8 7 10 5 10 9 9 4 11 10 11 11 11 11 4 3 3];
% trialdata(3).expt(n).success = [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 0 1 1 1 1 1];
% trialdata(3).expt(n).jumptime = {};
% trialdata(3).expt(n).vidnames = {'012719_G6H13p4RT.MOV'};
% 
% % n=n+1;
% % trialdata(3).expt(n).date = '012819';
% % trialdata(3).expt(n).trials = 32;
% % trialdata(3).expt(n).platform = [1 3 2 1 2 3 2 1 3 1 1 2 3 1 2 3 2 1 1 3 2 3 2 1 3 3 2 3 2 1 1 1];
% % trialdata(3).expt(n).distance = [3 4 4 7 5 7 8 10 10 9 6 9 9 7 11 11 6 5 8 6 11 8 7 10 5 3 10 5 3 4 11 12];
% % trialdata(3).expt(n).success = [1 1 1 1 1 1 1 0 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1];
% % trialdata(3).expt(n).jumptime = {};
% % trialdata(3).expt(n).vidnames = {'012819_G6H13p4RT.avi'};
% % 
% % n=n+1;
% % trialdata(1).expt(n).date = '012919';
% % trialdata(1).expt(n).trials = 30;
% % trialdata(1).expt(n).platform = [3 2 3 1 2 1 3 1 2 3 2 3 1 1 2 3 1 2 2 1 3 1 1 1 3 2 1 2 3 3];
% % trialdata(1).expt(n).distance = [4 8 5 4 5 10 10 7 6 8 4 7 8 11 9 6 3 10 7 6 3 9 4 5 11 11 10 3 11 9];
% % trialdata(1).expt(n).success = [1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1];
% % trialdata(1).expt(n).jumptime = {};
% % trialdata(1).expt(n).vidnames = {'012919_G6H13p4RT.avi'};
