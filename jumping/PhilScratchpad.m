%%
colors = jet(size(xcent,1));
scatter(xcent,ycent,100,colors)

%% get kate's data to tif
for i = 1:9
    fname = sprintf('B6_5_001_00%d',i)
    sbx2tif(fname)
    disp('done')
end