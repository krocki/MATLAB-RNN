function [data] = read_raw(filename)

%
% read byte stream into var data
%
% Author: Kamil Rocki <kmrocki@us.ibm.com>
% Created on: 02/02/2016
%

    fid = fopen(filename, 'r');
    data = fread(fid);
    
end