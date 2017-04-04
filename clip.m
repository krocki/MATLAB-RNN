function [ out ] = clip( x, r1, r2 )

%
% Clip x value to [r1, r2]
%
% Author: Kamil Rocki <kmrocki@us.ibm.com>
% Created on: 02/02/2016
%

    
    out = x;
    
    out = max(out, r1);
    out = min(out, r2);
    
end

