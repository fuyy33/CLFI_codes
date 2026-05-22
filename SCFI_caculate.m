clear;clc

years = 2001:2024;

inputED = '';   
inputPD = '';   
inputMPA = '';   
outputFolder = '';
list_ED = dir([inputED,'*.tif']);
list_PD = dir([inputPD,'*.tif']);
list_MPA = dir([inputMPA,'*.tif']);


[data, R] = readgeoraster(fullfile(inputED, sprintf('*.tif', years(1))));
info = geotiffinfo (fullfile(inputED, sprintf('*.tif', years(1))));


globalMinED = Inf;
globalMaxED = -Inf;
globalMinPD = Inf;
globalMaxPD = -Inf;
globalMinMPA = Inf;
globalMaxMPA = -Inf;

for yIdx = 1:length(years)
    y = years(yIdx);
    
    ED = double(readgeoraster(fullfile(inputED, sprintf('*.tif', y))));
    PD = double(readgeoraster(fullfile(inputPD, sprintf('*.tif', y))));
    MPA = double(readgeoraster(fullfile(inputMPA, sprintf('*.tif', y))));
    
    validMask = ~isnan(ED) & ~isnan(PD) & ~isnan(MPA) & (PD > 0);
    
    globalMinED = min(globalMinED, min(ED(validMask)));
    globalMaxED = max(globalMaxED, max(ED(validMask)));
    
    globalMinPD = min(globalMinPD, min(PD(validMask)));
    globalMaxPD = max(globalMaxPD, max(PD(validMask)));
    
    globalMinMPA = min(globalMinMPA, min(MPA(validMask)));
    globalMaxMPA = max(globalMaxMPA, max(MPA(validMask)));
    disp(y)
end


for yIdx = 1:length(years)
    y = years(yIdx);

    ED = double(readgeoraster(fullfile(inputED, sprintf('*.tif', y))));
    PD = double(readgeoraster(fullfile(inputPD, sprintf('*.tif', y))));
    MPA = double(readgeoraster(fullfile(inputMPA, sprintf('*.tif', y))));

    baseMask = ~isnan(ED) & ~isnan(PD) & ~isnan(MPA);  
    cropMask = baseMask & (PD > 0); 

    ED_nor = nan(size(ED));
    PD_nor = nan(size(PD));
    MPA_nor = nan(size(MPA));

    if globalMaxED > globalMinED
        ED_nor(cropMask) = (ED(cropMask) - globalMinED) / (globalMaxED - globalMinED);
    else
        ED_nor(cropMask) = 0; 
    end

    if globalMaxPD > globalMinPD
        PD_nor(cropMask) = (PD(cropMask) - globalMinPD) / (globalMaxPD - globalMinPD);
    else
        PD_nor(cropMask) = 0;
    end

    if globalMaxMPA > globalMinMPA
        MPA_nor(cropMask) = (MPA(cropMask) - globalMinMPA) / (globalMaxMPA - globalMinMPA);
    else
        MPA_nor(cropMask) = 0;
    end

    SCFI = nan(size(ED));
    SCFI(cropMask) = (ED_nor(cropMask) + PD_nor(cropMask) + (1 - MPA_nor(cropMask))) / 3;

    outFile = fullfile(outputFolder, sprintf('CN_Sugarcane_SCFI_%d.tif', y));
    geotiffwrite(outFile, SCFI, R, 'GeoKeyDirectoryTag',info.GeoTIFFTags.GeoKeyDirectoryTag);
    clear ED PD MPA ED_nor PD_nor MPA_nor SCFI baseMask cropMask outFile
end
