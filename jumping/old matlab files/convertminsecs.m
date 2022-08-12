function [newtimes] = convertminsecs(times)
%%%feed in an array of minutes/seconds as minutes.seconds and get out
%%%seconds, e.g. 1.40 for one minutes 40 seconds returns 100 seconds

mins = floor(times);
secs = 100*(times-mins);
newtimes = round(mins*60+secs)