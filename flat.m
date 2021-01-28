load('weights_folded.mat');
load('bias_folded.mat');
flatw = [];
flatb = [];
for i=1:13
    tempw=weights_folded{i};
    flatw=vertcat(flatw,tempw(:));
    tempb=bias_folded{i};
    flatb=vertcat(flatb,tempb(:));
end
 H=NumericTypeScope;
step(H,flatw)