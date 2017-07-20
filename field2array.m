function y = field2array(data,f);
y = zeros(1,length(data));
for i=1:length(data)
    y(i) = getfield(data(i),f);
end