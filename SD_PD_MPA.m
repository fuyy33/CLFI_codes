clear;clc

inputFolder = '';
outputFolder = '';
gridFile = '';
years = 2001:2024

list = dir([inputFolder,'*.tif']);

if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);
end

[grid1000, R_1000m] = readgeoraster(gridFile);
info = geotiffinfo (gridFile);
[gridRows, gridCols] = size(grid1000);

cellSize = 30;
winSize = round(1000 / cellSize); 


for yIdx = 1:length(years)
    y = years(yIdx);
    fprintf('Processing year %d...\n', y);

    try

        inputFile = list(y-2000).name
        [map30, ~] = readgeoraster([inputFolder,list(y-2000).name]);
        
        map30(map30==1)=0; 
        map30(map30==6|map30==9)=1;
        map30(map30~=1)=0; 
        max(map30(:))
        min(map30(:))
  
        ED_map = nan(gridRows, gridCols,'single');
        PD_map = nan(gridRows, gridCols,'single');
        MPA_map = nan(gridRows, gridCols,'single');
       
        MyPar = parpool('local',20);
        tic

        parfor i = 1:gridRows
            for j = 1:gridCols
                rowStart = (i-1)*winSize + 1;
                colStart = (j-1)*winSize + 1;

                if rowStart + winSize - 1 > size(map30,1) || colStart + winSize - 1 > size(map30,2)
                    continue;
                end

                patch = map30(rowStart:rowStart+winSize-1, colStart:colStart+winSize-1);

                if all(patch(:)==0) || all(isnan(patch(:)))
                    continue;
                end

                A = (winSize * cellSize)^2;

                L = bwlabel(patch==1, 8);
                stats = regionprops(L, 'Area', 'Perimeter');  

                if isempty(stats)
                    continue;
                end

                % Edge Density
                totalEdge = sum([stats.Perimeter]) * cellSize;  
                ED = (totalEdge / A) * 10000;  
                % Patch Density
                n_i = numel(stats);
                PD = (n_i / A) * 10000 * 100;  % patches/km²
                % Mean Patch Area
                areas = [stats.Area] * (cellSize^2) / 10000;  % m² → ha
                MPA = mean(areas);

                ED_map(i,j) = ED;
                PD_map(i,j) = PD;
                MPA_map(i,j) = MPA;
            end
        end
        toc
        delete(MyPar);
        clear map30 patch L stats
    
        geotiffwrite(fullfile(outputFolder, sprintf('Gansu_Maize_ED_%d.tif', y)), ED_map, R_1000m, ...
            'GeoKeyDirectoryTag',info.GeoTIFFTags.GeoKeyDirectoryTag,'TiffType','bigtiff');
        geotiffwrite(fullfile(outputFolder, sprintf('Gansu_Maize_PD_%d.tif', y)), PD_map, R_1000m, ...
            'GeoKeyDirectoryTag',info.GeoTIFFTags.GeoKeyDirectoryTag,'TiffType','bigtiff');
        geotiffwrite(fullfile(outputFolder, sprintf('Gansu_Maize_MPA_%d.tif', y)), MPA_map, R_1000m, ...
            'GeoKeyDirectoryTag',info.GeoTIFFTags.GeoKeyDirectoryTag,'TiffType','bigtiff');
    clear ED_map PD_map MPA_map;
    catch ME
        warning('Failed to process year %d: %s', y, ME.message);
    end

end




