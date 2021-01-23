% Yikai Mao 1/15/2021
% error calculation for each layer

function [output_error, mean_error, max_error] = kc_error(exprimental, theoretical)

output_error = abs(theoretical - exprimental);
mean_error = mean(output_error(:));
max_error = max(output_error(:));

end