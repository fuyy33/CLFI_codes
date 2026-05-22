
clear; clc;

datapath = ' ';
outpath = ' ';

list = dir(fullfile(datapath, '*.tif'));
[~,idx] = sort({list.name});
list = list(idx);

[img, R] = readgeoraster(fullfile(datapath, list(1).name));
info = geotiffinfo(fullfile(datapath, list(1).name));
[rows, cols] = size(img);
years = 2001:2024;
n_years = length(years);


cfi_stack = zeros(rows, cols, n_years);
for y = 1:n_years
    cfi_stack(:,:,y) = readgeoraster(fullfile(datapath, list(y).name));   

nonzero_thresholds = 3:15  

for t = nonzero_thresholds
    
    sen_slope = nan(rows, cols);
    z_value = nan(rows, cols);
    trend = nan(rows, cols);
    MyPar = parpool('local', 20);
    parfor r = 1:rows
        for c = 1:cols
            
            ts = squeeze(cfi_stack(r,c,:));

            if all(isnan(ts)) || all(ts == 0)
                continue;
            end

            valid_ts = ts(~isnan(ts));

            if all(valid_ts == 0)
                continue;
            end

            nonzero_years = sum(valid_ts > 0)
            if nonzero_years < t
                sen_slope(r,c) = 0;
                z_value(r,c) = 0;
                trend(r,c) = 3;
                continue;
            end

            slopes = [];
            for i = 2:n_years
                for j = 1:(i-1)
                    if ~isnan(ts(i)) && ~isnan(ts(j))
                        slopes = [slopes; (ts(i) - ts(j)) / (i - j)];
                    end
                end
            end
            sen_slope(r,c) = median(slopes);

            S = 0;
            for i = 2:n_years
                for j = 1:(i-1)
                    if ~isnan(ts(i)) && ~isnan(ts(j))
                        S = S + sign(ts(i) - ts(j));
                    end
                end
            end
            varS = (n_years*(n_years-1)*(2*n_years+5))/18;
            if S > 0
                Z = (S - 1)/sqrt(varS);
            elseif S < 0
                Z = (S + 1)/sqrt(varS);
            else
                Z = 0;
            end
            z_value(r,c) = Z;

            if sen_slope(r,c) > 0
                if abs(Z) > 1.96
                    trend(r,c) = 2;  
                else
                    trend(r,c) = 1;   
                end
            elseif sen_slope(r,c) < 0
                if abs(Z) > 1.96
                    trend(r,c) = -2; 
                else
                    trend(r,c) = -1;  
                end
            else
                trend(r,c) = 0;
            end
        end
    end
    delete(MyPar);


    tag = sprintf('nonzero_lt%d', t);
    geotiffwrite(fullfile(outpath, [tag '_Maize_SCFI_Sen_2024.tif']), sen_slope, R, 'GeoKeyDirectoryTag', info.GeoTIFFTags.GeoKeyDirectoryTag);
    geotiffwrite(fullfile(outpath, [tag '_Maize_SCFI_Z_2024.tif']), z_value, R, 'GeoKeyDirectoryTag', info.GeoTIFFTags.GeoKeyDirectoryTag);
    geotiffwrite(fullfile(outpath, [tag '_Maize_SCFI_Trend_int16_2024.tif']), int16(trend), R, 'GeoKeyDirectoryTag', info.GeoTIFFTags.GeoKeyDirectoryTag);
    geotiffwrite(fullfile(outpath, [tag '_Maize_SCFI_Trend_2024.tif']), single(trend), R, 'GeoKeyDirectoryTag', info.GeoTIFFTags.GeoKeyDirectoryTag);

    sig_class_values = [-2, -1, 0, 1, 2];
    sig_class_names = {
        'sig_decrease',
        'nonsig_decrease',
        'nochange',
        'nonsig_increase',
        'sig_increase'
        };
    sig_counts = zeros(length(sig_class_values), 1);
    for k = 1:length(sig_class_values)
        sig_counts(k) = sum(trend(:) == sig_class_values(k), 'omitnan');   
    end
    sig_valid_pixels = sum(sig_counts);
    sig_percentages = sig_counts / sig_valid_pixels * 100;
    sig_result_table = table( ...
        sig_class_names(:), ...
        sig_counts(:), ...
        sig_percentages(:), ...
        'VariableNames', {'Type', 'pixel_num', 'proportion(%)'});
    outputExcel = fullfile(outpath,  '3', sprintf('SCFI_Trend_Proportion_lt%d_sig2_2024.xlsx', t));
    writetable(sig_result_table, outputExcel, 'Sheet', 'Maize');

end


