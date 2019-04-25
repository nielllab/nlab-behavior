%% batchPhilJumping contains data from jumping experiments
%jumptime in seconds
%note all 012919 experiments have weird glitchy video

% datapathname = '\\langevin\backup\Phil\Behavior\K99 prelim\';
% outpathname = '\\langevin\backup\Phil\Behavior\K99 prelim\';

datapathname = 'D:\Jumping Analysis\Jumping-Phil-2019-04-01_AVI\videos\';
outpathname = 'D:\Jumping Analysis\Jumping-Phil-2019-04-01_AVI\videos\';

trialdata = {};

%% animal #1
trialdata(1).name = 'G6H13p4TT';
n=0;

% n=n+1;
% trialdata(1).expt(n).date = '012119';
% trialdata(1).expt(n).cond = 'control';
% trialdata(1).expt(n).trials = 30;
% trialdata(1).expt(n).platform = [1 2 3 2 3 1 3 2 2 1 3 3 1 2 3 1 2 2 1 1 3 3 1 2 3 1 2 2 2 2];
% trialdata(1).expt(n).distance = [5 8 4 6 8 7 7 6 9 8 3 5 8 10 10 10 7 5 6 4 9 6 11 11 11 3 3 4 5 11];
% trialdata(1).expt(n).success = [1 0 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 0 1 1 0 1 1 1 1 1 1];
% trialdata(1).expt(n).jumptime = {};
% trialdata(1).expt(n).vidnames = {''};
% 
% n=n+1;
% trialdata(1).expt(n).date = '012219';
% trialdata(1).expt(n).cond = 'control';
% trialdata(1).expt(n).trials = 31;
% trialdata(1).expt(n).platform = [2 1 3 2 1 2 3 3 2 1 3 2 3 2 1 1 2 1 3 2 3 1 3 1 2 2 2 2 3 1 2];
% trialdata(1).expt(n).distance = [4 3 5 4 4 5 6 4 6 7 7 8 3 3 5 6 7 8 8 10 10 9 9 10 10 9 10 10 11 11 11];
% trialdata(1).expt(n).success = [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 0 1 0 1 1 1 1];
% trialdata(1).expt(n).jumptime = {};
% trialdata(1).expt(n).vidnames = {'012219_G6H13p4TT.MOV'};

% n=n+1;
% trialdata(1).expt(n).date = '012319';
% trialdata(1).expt(n).cond = 'control';
% trialdata(1).expt(n).trials = 37;
% trialdata(1).expt(n).platform = [1 2 3 1 2 3 1 2 1 2 2 3 3 3 2 1 2 1 3 1 2 2 1 1 3 3 3 2 1 3 3 2 1 3 1 2 3];
% trialdata(1).expt(n).distance = [3 2 3 4 5 4 5 6 6 4 3 4 5 8 8 7 7 9 9 8 10 9 11 10 10 7 11 11 11 11 11 8 9 9 8 9 8];
% trialdata(1).expt(n).success = [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 0 1 1 0 1 1 1 1 1 1 1];
% trialdata(1).expt(n).jumptime = {[41.5 80.6 106.6 132.8 159.6 192.5 223.3 247.5 286.8 321.1 342.9 371.1 402.7 452.8,...
%     489.3 554.7 591.7 622.3 673.0 712.9 743.0 776.3 828.3 858.1 898.5 942.9 1002.3 1017.7 1079.0 1136.1 1150.9 1215.0,...
%     1262.5 1297.5 1364.5 1451.1 1638.6]};
% trialdata(1).expt(n).vidnames = {'012319_G6H13p4TT.MOV'};

% n=n+1; % first two trials not recorded so left them out
% trialdata(1).expt(n).date = '012419';
% trialdata(1).expt(n).cond = 'control';
% trialdata(1).expt(n).trials = 47;
% trialdata(1).expt(n).platform = [3 3 2 1 2 1 3 3 1 2 3 2 1 2 2 1 1 3 2 3 1 3 3 3 2 1 3 2 3 2 2 3 3 1 2 3 3 1 2 3 2 1 3 1 1 2 3];
% trialdata(1).expt(n).distance = [5 3 6 5 5 4 7 6 6 3 4 8 8 8 9 7 9 9 7 11 10 11 9 8 11 11 11 11 10 10 11 11 11 8 9 8 8 3 4 5 8 9 10 10 10 10 10];
% trialdata(1).expt(n).success = [1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 0 1 0 1 0 1 1 0 1 0 0 1 1 1 0 1 1 1 0 1 1 1 1 1 1 0 0 1 1 1];
% trialdata(1).expt(n).jumptime = {[10.7 46.3 80.5 103.7 133.8 161.4 183.1 205.6 233.4 259.1 286.0 312.9 331.7 370.3,...
%     392.3 422.0 512.4 547.0 566.1 609.3 623.3 663.6 678.2 716.0 756.7 768.6 822.8 837.1 854.2 887.8 940.1 1003.9,...
%     1015.9 1080.2 1112.8 1144.1 1157.3 1212.5 1260.3 1312.9 1352.9 1391.1 1429.7 1453.5 1468.4 1533.5 1676.4]};
% trialdata(1).expt(n).vidnames = {'012419_G6H13p4TT.MOV'};
% 
% n=n+1;
% trialdata(1).expt(n).date = '012519';
% trialdata(1).expt(n).cond = 'control';
% trialdata(1).expt(n).trials = 45;
% trialdata(1).expt(n).platform = [3 2 3 1 1 3 1 2 2 1 3 3 2 1 2 1 3 1 2 2 3 1 2 3 1 2 3 1 3 2 3 2 1 2 3 1 2 3 1 1 3 3 2 2 1];
% trialdata(1).expt(n).distance = [4 3 3 5 3 5 4 4 5 6 6 7 6 7 7 8 8 9 8 9 9 10 10 10 11 11 10 11 11 11 11 8 9 6 10 9 9 7 10 7 6 9 10 10 8];
% trialdata(1).expt(n).success = [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 1 1 1 0 0 1 1 1 1 1 1 1 1 0 1 1 1 0 1 1];
% trialdata(1).expt(n).jumptime = {[147.7 181.5 217.0 258.5 293.4 334.6 370.7 400.4 447.8 488.3 521.3 561.6 584.1 609.1,...
%     629.1 654.6 679.0 716.1 744.2 796.7 833.2 892.8 925.7 969.2 999.3 1023.5 1056.5 1111.1 1150.3 1167.9 1178.9 1219.3,...
%     1268.3 1314.5 1371.4 1407.3 1451.6 1489.3 1551.7 1577.7 1619.3 1681.9 1725.6 1743.7 1778.9]};
% trialdata(1).expt(n).vidnames = {'012519_G6H13p4TT.MOV'};
% 
% n=n+1;
% trialdata(1).expt(n).date = '012619';
% trialdata(1).expt(n).cond = 'control';
% trialdata(1).expt(n).trials = 33;
% trialdata(1).expt(n).platform = [2 3 1 2 1 3 2 1 1 2 3 2 1 3 2 3 1 3 1 1 3 3 2 3 2 2 1 2 1 2 2 3 2];
% trialdata(1).expt(n).distance = [3 4 3 5 5 3 4 4 7 6 5 8 6 6 7 7 8 10 11 9 8 9 10 10 9 9 10 11 11 11 10 11 11];
% trialdata(1).expt(n).success = [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 1 1 1 0 1 0 1 1 0 1 0 1 1 1];
% trialdata(1).expt(n).jumptime = {[48.6 79.5 103.2 133.7 169.9 209.0 238.1 266.6 298.1 325.5 353.6 393.5 435.0 487.2,...
%     526.2 559.1 594.0 630.7 649.5 666.1 718.8 769.5 832.7 871.0 908.2 916.7 964.3 1018.3 1044.2 1106.9 1123.9 1181.0 1226.7]};
% trialdata(1).expt(n).vidnames = {'012619_G6H13p4TT.MOV'};
% 
% n=n+1;
% trialdata(1).expt(n).date = '012719';
% trialdata(1).expt(n).cond = 'control';
% trialdata(1).expt(n).trials = 29;
% trialdata(1).expt(n).platform = [3 2 1 3 1 2 3 2 3 1 1 2 3 1 1 2 3 3 2 3 2 1 3 2 1 1 2 2 1];
% trialdata(1).expt(n).distance = [4 4 5 3 3 5 6 7 8 6 7 6 7 9 8 9 5 10 8 9 3 11 11 10 11 10 11 11 4];
% trialdata(1).expt(n).success = [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 0 1 1];
% trialdata(1).expt(n).jumptime = {[32.9 72.1 96.2 118.7 157.5 194.2 224.1 268.6 298.6 325.5 366.0 402.8 441.1 480.0 515.0,...
%     542.2 571.3 613.8 650.2 675.2 710.7 742.1 783.7 840.3 873.1 915.8 942.8 963.6 998.1]};
% trialdata(1).expt(n).vidnames = {'012719_G6H13p4TT.MOV'};

n=n+1;
trialdata(1).expt(n).date = '012819';
trialdata(1).expt(n).cond = 'control';
trialdata(1).expt(n).trials = 29;
trialdata(1).expt(n).platform = [2 3 1 2 1 3 1 2 1 3 2 2 1 3 2 1 3 2 3 1 2 3 1 3 2 3 1 2 1];
trialdata(1).expt(n).distance = [3 4 5 4 3 6 7 5 4 3 6 7 8 5 10 11 8 9 11 6 8 10 9 7 5 9 10 11 12];
trialdata(1).expt(n).success = [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1];
trialdata(1).expt(n).jumptime = {[25 51 75 96 124 160 193 225 241 263 287 324 343 372 413 439 466,...
    509 555 580 609 637 673 713 755 823 882 937 1008]};
trialdata(1).expt(n).vidnames = {'012819_G6H13p4TT.avi'};

% n=n+1;
% trialdata(1).expt(n).date = '012919';
% trialdata(1).expt(n).cond = 'control';
% trialdata(1).expt(n).trials = 37;
% trialdata(1).expt(n).platform = [2 1 3 3 1 2 2 3 1 2 2 2 3 3 2 1 3 1 3 3 2 3 3 2 1 1 1 2 3 1 2 2 2 1 3 3 3];
% trialdata(1).expt(n).distance = [8 6 3 7 5 7 4 9 8 6 5 10 11 9 9 10 11 11 6 8 3 10 5 9 4 7 9 11 4 3 11 5 6 8 9 4 7];
% trialdata(1).expt(n).success = [1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1];
% trialdata(1).expt(n).jumptime = {[16 55 94 127 163 200 232 266 293 320 352],[28 69 107 147 225],...
%     [26 74 110 137 172 211 247 287 344 369 393 425 437 479 509 638 665 702 740 789 830]};
% trialdata(1).expt(n).vidnames = {'012919_G6H13p4TTa.avi','012919_G6H13p4TTb.avi','012919_G6H13p4TTc.avi'};

% n=n+1;
% trialdata(1).expt(n).date = '020319';
% trialdata(1).expt(n).cond = 'suture';
% trialdata(1).expt(n).trials = 31;
% trialdata(1).expt(n).platform = [1 2 3 1 3 2 1 2 3 1 2 3 2 1 1 2 1 1 2 3 3 3 2 1 3 2 3 1 2 2 3];
% trialdata(1).expt(n).distance = [3 4 5 6 8 10 8 6 4 4 9 7 7 9 9 5 5 7 8 6 9 9 3 10 11 9 10 11 11 10 3];
% trialdata(1).expt(n).success = [1 1 1 1 1 0 1 1 1 1 0 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1];
% trialdata(1).expt(n).jumptime = {[7 55 88 142 218 291 308 365 712 459 530 553 605 674 724 767 812 888 945,...
%     991 1047 1082 1138 1290 1413 1479 1649 1986 2081 2153 2196]};
% trialdata(1).expt(n).vidnames = {'020319_G6H13p4TT.avi'};

% n=n+1;
% trialdata(1).expt(n).date = '020619';
% trialdata(1).expt(n).cond = 'suture';
% trialdata(1).expt(n).trials = 25;
% trialdata(1).expt(n).platform = [3 1 2 1 2 3 2 3 3 1 2 3 1 2 3 3 1 3 1 2 3 1 3 1 2];
% trialdata(1).expt(n).distance = [4 6 5 8 7 8 4 6 10 9 10 10 5 8 5 7 9 9 7 6 9 4 3 3 3];
% trialdata(1).expt(n).success = [1 1 1 1 1 1 1 1 0 0 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1];
% trialdata(1).expt(n).jumptime = {[30 60 115 198 244 303 359 403 522 735 764 831 882 944 1029 1080 1152,...
%     1280 1594 1680 1958 2007 2157 2252 2353]};
% trialdata(1).expt(n).vidnames = {'020619_G6H13p4TT.avi'};

%% animal #2
trialdata(2).name = 'G6H13p4LT';
n=0;

% n=n+1;
% trialdata(2).expt(n).date = '012119';
% trialdata(2).expt(n).cond = 'control';
% trialdata(2).expt(n).trials = 25;
% trialdata(2).expt(n).platform = [3 2 1 3 2 3 1 2 3 1 2 3 3 1 2 2 1 1 2 1 3 2 3 9 9];
% trialdata(2).expt(n).distance = [2 2 2 3 4 5 5 6 7 7 8 6 4 6 5 3 3 4 7 8 8 8 9 9 9];
% trialdata(2).expt(n).success = [1 1 1 1 1 1 1 1 1 1 0 0 1 1 1 1 1 1 1 1 0 1 0 1 1];
% trialdata(2).expt(n).jumptime = {};
% trialdata(2).expt(n).vidnames = {''};
% 
% n=n+1;
% trialdata(2).expt(n).date = '012219';
% trialdata(2).expt(n).cond = 'control';
% trialdata(2).expt(n).trials = 25;
% trialdata(2).expt(n).platform = [2 1 3 2 1 2 3 1 3 2 1 3 3 2 2 1 1 3 3 3 2 2 1 3 2];
% trialdata(2).expt(n).distance = [3 3 4 5 4 4 3 6 6 8 7 8 7 7 6 8 5 8 5 9 9 9 9 10 10];
% trialdata(2).expt(n).success = [1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 0 1 1 1 1];
% trialdata(2).expt(n).jumptime = {};
% trialdata(2).expt(n).vidnames = {'012219_G6H13p4LTa.MOV','012219_G6H13p4LTa.MOV'};
% 
% n=n+1;
% trialdata(2).expt(n).date = '012319';
% trialdata(2).expt(n).cond = 'control';
% trialdata(2).expt(n).trials = 27;
% trialdata(2).expt(n).platform = [3 2 1 3 2 1 1 3 2 3 3 1 2 3 2 1 3 2 1 3 2 3 1 3 1 2 3];
% trialdata(2).expt(n).distance = [3 3 3 5 5 4 5 6 4 5 4 7 6 8 7 6 8 9 8 7 8 7 9 9 10 10 10];
% trialdata(2).expt(n).success = [1 1 1 0 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 0 1 1 1 1 1 0 1];
% trialdata(2).expt(n).jumptime = {[79,132,161,201,222,262,300,400,441,500,557,596,673,...
%     710,808,863,935,1020,1070,1145,1168,1222,1293,1428,1607,1669,1690]};
% trialdata(2).expt(n).vidnames = {'012319_G6H13p4LT.MOV'};

% n=n+1; %last trial not recorded
% trialdata(2).expt(n).date = '012419';
% trialdata(2).expt(n).cond = 'control';
% trialdata(2).expt(n).trials = 31;
% trialdata(2).expt(n).platform = [2 3 1 3 1 2 3 1 2 3 1 2 2 1 3 1 2 1 2 3 1 3 1 3 3 2 3 1 1 2 1];
% trialdata(2).expt(n).distance = [3 3 4 5 3 4 4 6 5 6 5 6 8 7 8 8 9 8 7 7 8 9 9 8 9 10 10 10 10 11 11];
% trialdata(2).expt(n).success = [1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 1 0 1 1 1 0 1 1 1 1 1 0 1 0 1];
% trialdata(2).expt(n).jumptime = {[76,131,218,258,311,366,403,448,483,523,580,629,715,...
%     824,892,913,923,1012,1040,1111,1168,1238,1264,1327,1380,1437,1605,1647,1668,1763]};
% trialdata(2).expt(n).vidnames = {'012419_G6H13p4LTa.MOV','012419_G6H13p4LTb.MOV'};
% 
% n=n+1;
% trialdata(2).expt(n).date = '012519';
% trialdata(2).expt(n).cond = 'control';
% trialdata(2).expt(n).trials = 20;
% trialdata(2).expt(n).platform = [1 3 1 3 2 1 2 2 3 1 2 3 2 1 3 3 1 2 1 3];
% trialdata(2).expt(n).distance = [2 3 3 3 3 4 5 5 4 5 4 6 6 6 5 8 7 7 8 7];
% trialdata(2).expt(n).success = [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1];
% trialdata(2).expt(n).jumptime = {[45,75,191,241,283,377,584,688,729,792,...
%     840,920,1095,1220,1261,1330,1378,1571,1713],[21]};
% trialdata(2).expt(n).vidnames = {'012519_G6H13p4LTa.MOV','012519_G6H13p4LTb.MOV'};
% 
% n=n+1;
% trialdata(2).expt(n).date = '012619';
% trialdata(2).expt(n).cond = 'control';
% trialdata(2).expt(n).trials = 32;
% trialdata(2).expt(n).platform = [2 1 3 2 3 1 2 3 3 1 2 2 3 1 2 2 1 3 1 2 2 3 1 3 2 3 2 2 1 2 3 1];
% trialdata(2).expt(n).distance = [3 4 3 5 6 6 4 5 8 7 7 6 8 9 10 8 5 7 10 10 10 4 8 9 10 10 9 10 3 11 11 11];
% trialdata(2).expt(n).success = [1 1 1 1 1 1 1 1 0 1 1 1 1 1 0 1 1 1 1 0 0 1 1 1 0 1 1 1 1 1 1 1];
% trialdata(2).expt(n).jumptime = {[103,146,189,236,284,342,390,442,515,535,596,644,692,755,810,835,877,929,...
%     1010,1060,1077,1109,1160,1242,1283,1324,1415,1482,1522,1598,1670,1746]};
% trialdata(2).expt(n).vidnames = {'012619_G6H13p4LT.MOV'};
% 
% n=n+1;
% trialdata(2).expt(n).date = '012719';
% trialdata(2).expt(n).cond = 'control';
% trialdata(2).expt(n).trials = 29;
% trialdata(2).expt(n).platform = [2 3 2 1 3 2 1 2 1 3 2 1 3 3 2 2 3 2 1 3 1 2 1 3 2 3 1 3 1];
% trialdata(2).expt(n).distance = [3 4 4 4 6 6 5 5 7 8 7 8 9 7 9 8 10 11 9 11 11 10 6 11 10 5 10 3 3];
% trialdata(2).expt(n).success = [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 0 1 1 1 1 1 1 1];
% trialdata(2).expt(n).jumptime = {[87,162,203,244,292,333,360,419,456,509,592,636,679,715,793,845,...
%     911,1030,1067,1152,1173,1242,1288,1376,1429,1467,1530,1604,1658]};
% trialdata(2).expt(n).vidnames = {'012719_G6H13p4LT.MOV'};
% 
n=n+1;
trialdata(2).expt(n).date = '012819';
trialdata(2).expt(n).cond = 'control';
trialdata(2).expt(n).trials = 32;
trialdata(2).expt(n).platform = [1 3 2 3 2 1 2 1 2 3 2 1 2 2 1 3 2 1 1 3 2 3 3 1 1 3 2 2 1 3 3 2];
trialdata(2).expt(n).distance = [3 4 5 8 6 5 8 7 4 6 10 8 5 9 9 7 7 11 6 10 11 9 11 10 4 10 3 11 10 5 3 11];
trialdata(2).expt(n).success = [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 0 1 1 1 0 1 1 1 1];
trialdata(2).expt(n).jumptime = {[56,88,123,154,190,225,298,334,389,434,480,538,602,662,719,776,817,894,...
    939,982,1002,1059,1117,1155,1177,1218,1263,1323,1347,1387,1428,1503]};
trialdata(2).expt(n).vidnames = {'012819_G6H13p4LT.avi'};

% n=n+1;
% trialdata(2).expt(n).date = '012919';
% trialdata(2).expt(n).cond = 'control';
% trialdata(2).expt(n).trials = 29;
% trialdata(2).expt(n).platform = [2 3 1 2 1 3 2 1 3 2 1 2 3 2 1 3 1 2 3 2 2 3 3 1 1 2 3 2 1];
% trialdata(2).expt(n).distance = [7 5 7 4 5 11 11 9 8 8 6 9 9 5 3 7 10 9 4 6 10 6 3 4 8 3 10 11 11];
% trialdata(2).expt(n).success = [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 0 1 1 1 1 1 1 1];
% trialdata(2).expt(n).jumptime = {[72,104,153,183,227,303,334,375,435,472,496,540,557,592,626,662,701,749,...
%     800,836,901,945],[25,141,233,261,375,460,578]};
% trialdata(2).expt(n).vidnames = {'012919_G6H13p4LTa.avi','012919_G6H13p4LTb.avi'};

% n=n+1;
% trialdata(2).expt(n).date = '020319';
% trialdata(2).expt(n).cond = 'suture';
% trialdata(2).expt(n).trials = 12;
% trialdata(2).expt(n).platform = [3 2 1 3 2 3 2 1 1 3 2 2];
% trialdata(2).expt(n).distance = [3 5 6 7 8 8 9 10 5 5 11 6];
% trialdata(2).expt(n).success = [1 1 1 1 0 1 1 1 1 1 0 1];
% trialdata(2).expt(n).jumptime = {[227,309,362,502,766,802,886,1034,1133,1284,1505,1804]};
% trialdata(2).expt(n).vidnames = {'020319_G6H13p4LT.avi'};
% 
% n=n+1;
% trialdata(2).expt(n).date = '020719';
% trialdata(2).expt(n).cond = 'suture';
% trialdata(2).expt(n).trials = 29;
% trialdata(2).expt(n).platform = [3 1 2 2 2 1 3 1 2 2 1 3 3 1 3 3 1 1 3 3 2 1 1 2 2 2 2 3 1];
% trialdata(2).expt(n).distance = [3 4 5 7 9 8 6 10 4 6 5 11 7 9 9 8 6 9 5 4 3 3 7 8 10 10 11 10 11];
% trialdata(2).expt(n).success = [1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1];
% trialdata(2).expt(n).jumptime = {[48,117,230,309,690,897,981,1450,1510,1554,1621,1876,1976,2140,...
%     2227,2354,2454,2727,2772,2868,2958,3042,3201,3360,3558,3599,3727,3800,4834]};
% trialdata(2).expt(n).vidnames = {'020719_G6H13p4LT.avi'};

%% animal #3
trialdata(3).name = 'G6H13p4RT';
n=0;

% n=n+1; %missing trial 4 on video
% trialdata(3).expt(n).date = '012619';
% trialdata(3).expt(n).cond = 'control';
% trialdata(3).expt(n).trials = 33;
% trialdata(3).expt(n).platform = [3 1 2 2 3 1 3 2 1 3 1 3 2 1 2 3 2 1 1 1 3 2 2 1 3 2 1 1 3 3 1 2];%3 
% trialdata(3).expt(n).distance = [3 3 3 3 3 3 4 5 6 5 5 6 7 8 6 7 8 7 4 9 8 9 4 9 9 10 11 10 10 11 11 11];%4 
% trialdata(3).expt(n).success = [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 0 1 1 1 1 1];%1 
% trialdata(3).expt(n).jumptime = {[40 111 184],[70,102,150,189,222,275,314,355,384,419,456,506,550,584,...
%     612,672,723,741,776,822,861,894,925,956,970,1019,1064,1112,1141]};
% trialdata(3).expt(n).vidnames = {'012619_G6H13p4RTa.MOV','012619_G6H13p4RTb.MOV'};
% 
% n=n+1;
% trialdata(3).expt(n).date = '012719';
% trialdata(3).expt(n).cond = 'control';
% trialdata(3).expt(n).trials = 29;
% trialdata(3).expt(n).platform = [2 3 1 2 1 2 1 3 2 3 1 1 2 3 2 3 1 3 2 2 2 3 3 1 1 2 1 3 1];
% trialdata(3).expt(n).distance = [3 4 6 6 7 5 5 6 7 8 9 8 8 7 10 5 10 9 9 4 11 10 11 11 11 11 4 3 3];
% trialdata(3).expt(n).success = [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 0 1 1 1 1 1];
% trialdata(3).expt(n).jumptime = {[25,63,91,122,178,264,305,351,384,421,472,525,581,636,665,744,...
%     792,829,874,926,1004,1024,1061,1100,1172,1205,1246,1302,1351]};
% trialdata(3).expt(n).vidnames = {'012719_G6H13p4RT.MOV'};

n=n+1;
trialdata(3).expt(n).date = '012819';
trialdata(3).expt(n).cond = 'control';
trialdata(3).expt(n).trials = 32;
trialdata(3).expt(n).platform = [1 3 2 1 2 3 2 1 3 1 1 2 3 1 2 3 2 1 1 3 2 3 2 1 3 3 2 3 2 1 1 1];
trialdata(3).expt(n).distance = [3 4 4 7 5 7 8 10 10 9 6 9 9 7 11 11 6 5 8 6 11 8 7 10 5 3 10 5 3 4 11 12];
trialdata(3).expt(n).success = [1 1 1 1 1 1 1 0 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1];
trialdata(3).expt(n).jumptime = {[65,125,160,194,254,307,339,371,407,438,473,520,555,605,654,673,733,784,...
    813,844,885,925,971,1019,1083,1127,1160,1196,1230,1261,1303,1336]};
trialdata(3).expt(n).vidnames = {'012819_G6H13p4RT.avi'};

% n=n+1;
% trialdata(3).expt(n).date = '012919';
% trialdata(3).expt(n).cond = 'control';
% trialdata(3).expt(n).trials = 30;
% trialdata(3).expt(n).platform = [3 2 3 1 2 1 3 1 2 3 2 3 1 1 2 3 1 2 2 1 3 1 1 1 3 2 1 2 3 3];
% trialdata(3).expt(n).distance = [4 8 5 4 5 10 10 7 6 8 4 7 8 11 9 6 3 10 7 6 3 9 4 5 11 11 10 3 11 9];
% trialdata(3).expt(n).success = [1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1];
% trialdata(3).expt(n).jumptime = {[9,81,130,159,195,251,271,307,331,366,402,435,465,501,525,579,649,...
%     708,750,777,824,860,890,926,1001,1027,1101,1159,1229,1291]};
% trialdata(3).expt(n).vidnames = {'012919_G6H13p4RT.avi'};

% n=n+1;
% trialdata(3).expt(n).date = '020719';
% trialdata(3).expt(n).cond = 'suture';
% trialdata(3).expt(n).trials = 14;
% trialdata(3).expt(n).platform = [3 2 1 2 3 1 2 1 3 1 2 2 3 2];
% trialdata(3).expt(n).distance = [3 5 6 4 8 7 9 10 11 5 6 8 7 4];
% trialdata(3).expt(n).success = [1 1 1 1 1 1 1 1 1 1 1 1 0 1];
% trialdata(3).expt(n).jumptime = {[162,221,427,475,580,679,775,828,1233,1408,1486,1906,2060,2175]};
% trialdata(3).expt(n).vidnames = {'020719_G6H13p4RT.avi'};
