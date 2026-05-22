clear;clc

outpath = '';
[FC_trend, R] = readgeoraster('');
info = geotiffinfo('');
SCFI_trend = readgeoraster('');


[nRows, nCols] = size(FC_trend);
landscape_type = nan(nRows, nCols,'single');
% fc_trend=nan(nRows, nCols,'single');
scfi_trend = nan(nRows, nCols,'single');

MyPar = parpool('local',15);
parfor i = 1:nRows
    for j = 1:nCols
        fc = FC_trend(i,j);
        scfi = SCFI_trend(i,j);

        if isnan(fc) || isnan(scfi)
            continue;
        end

        
        if  fc == 2   
            fc_trend = 1;
        elseif fc == -2  
            fc_trend = -1;
        elseif fc == -1 || fc == 1 || fc ==0 
            fc_trend = 0;
        else
            continue;
        end

        if scfi == 2  
            scfi_trend = 1;
        elseif scfi == -2  
            scfi_trend = -1;
        elseif scfi == -1|| scfi == 1  || scfi == 0 
            scfi_trend = 0;
        else
            continue;
        end

        if fc_trend == 1 && scfi_trend == -1  % CA↑CLFI↓
            landscape_type(i,j) = 1;   
        elseif fc_trend == 1 && scfi_trend == 1  % CA↑CLFI↑
            landscape_type(i,j) = 2; 
        elseif fc_trend == -1 && scfi_trend == 1 %  CA↓CLFI↑
            landscape_type(i,j) = 3; 
        elseif fc_trend == -1 && scfi_trend == -1 % CA↓CLFI↓
            landscape_type(i,j) = 4; 
        elseif fc_trend == 0 && scfi_trend == 1 % CA→CLFI↑
            landscape_type(i,j) = 5; 
        elseif fc_trend == 0 && scfi_trend == -1 % CA→CLFI↓
            landscape_type(i,j) = 6; 
        elseif fc_trend == 1 && scfi_trend == 0 % CA↑CLFI→
            landscape_type(i,j) = 7; 
        elseif fc_trend == -1 && scfi_trend == 0 % CA↓CLFI→
            landscape_type(i,j) = 8; 
        elseif fc_trend == 0 && scfi_trend == 0 % CA→CLFI→
            landscape_type(i,j) = 9; 
        end 
    end
end
delete(MyPar);

geotiffwrite(fullfile(outpath, 'landscape_dynamic_class_Maize_2024.tif'), ...
    int8(landscape_type), R, 'GeoKeyDirectoryTag', info.GeoTIFFTags.GeoKeyDirectoryTag);

valid_mask = ~isnan(landscape_type);
total_pixels = sum(valid_mask(:));

type_labels = {
    'CA↑CLFI↓', 'CA↑CLFI↑',...
    'CA↓CLFI↑', 'CA↓CLFI↓', ...
    'CA→CLFI↑', 'CA→CLFI↓', ...
    'CA↑CLFI→', 'CA↓CLFI→', 'CA→CLFI→'
};

pixel_count = zeros(9,1);
percentage = zeros(9,1);

for k = 1:9
    num_k = sum(landscape_type(:) == k);
    pixel_count(k) = num_k;
    percentage(k) = num_k / total_pixels * 100;
end

% -------------------------------
T_all = table((1:9)', type_labels', pixel_count, percentage,  ...
    'VariableNames', {'value', 'type', 'pixel_num', 'proportion'});

outputExcel = fullfile(outpath, 'landscape_dynamic_2024.xlsx');
writetable(T_all, outputExcel, 'Sheet', 'Maize', 'WriteRowNames', true);
