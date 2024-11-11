function [Xtrain,Xval] = create_datastores()
%Creates combined datastores of images and labels for training and validation

%% Locations
comp_loc1 = "../1_Alloy_Data/compositions.xls";
ele_loc1 = "../1_Alloy_Data/elemental_features.xlsx";
lab_loc1 = "../1_Alloy_Data/compositions.xls";
im_loc1 = "../1_Alloy_Data/images_augmented/";

%% Import compositions from spreadsheet
opts = spreadsheetImportOptions("NumVariables", 21);
opts.Sheet = "Sheet1";
opts.DataRange = "B2:V110";
opts.VariableNames = ["Al", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zr", "Nb", "Mo", "Hf", "Ta", "W", "Re", "C", "B", "Si", "P", "S"];
opts.VariableTypes = ["double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double"];
compositions = readtable(comp_loc1, opts, "UseExcel", false);
compositions = table2array(compositions);
clear opts

%% Import elemental features from spreadsheet
opts = spreadsheetImportOptions("NumVariables", 21);
opts.Sheet = "Sheet1";
opts.DataRange = "B2:V12";
opts.VariableNames = [ "Al", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zr", "Nb", "Mo", "Hf", "Ta", "W", "Re", "C", "B", "Si", "P", "S"];
opts.VariableTypes = [ "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double"];
elementalfeatures = readtable(ele_loc1, opts, "UseExcel", false);
elementalfeatures = table2array(elementalfeatures);
clear opts

%% Import target property
[xldat]=xlsread(lab_loc1);

%% Import Images
imds = imageDatastore(im_loc1,"FileExtensions",[".png"]);

%% Normalize features 
compositions = normalize(compositions,1,'range',[0,1]);
elementalfeatures = normalize(elementalfeatures,2,'range',[0,1]);

%% Assign labels & feature vector according to the filename
imagefiles = dir(strcat(im_loc1,'*.png'));      
nfiles = length(imagefiles);    % Number of files found
for ii=1:nfiles
   currentfilename = imagefiles(ii).name;
   currentimage = imread(strcat(im_loc1,currentfilename));
   num = extractBefore(currentfilename,"_");
   for j=1:109
       if string(xldat(j,1))== num
         hardness(ii)= (xldat(j,23));
         for x=1:21 
             for y=1:11
             fv(ii,y,x)=compositions(j,x)*elementalfeatures(y,x);
             end
         end
       end
   end
end

%% flatten composition-elemental features
for i=1:1962
    fv1 = zeros([11,21]);
    fv1(:,:)=fv(i,:,:);
    fvflat(i,:) = fv1(:);
end

%% Make datastores and subset them into train & val randomly (fixed random state for all tests)
fds = arrayDatastore(fvflat);
hardness = hardness';
hds = arrayDatastore(hardness);

% Load fixed random state 
split = load('../Data/data_split_for_tests.mat');
ntrain = split.ntrain;
nval = split.nval;

% Split datastores
Yval = subset(hds,nval);
Ytrain = subset(hds,ntrain);

ival = subset(imds,nval);
itrain = subset(imds,ntrain);

fval = subset(fds,nval);
ftrain = subset(fds,ntrain);

% Composition-elemental features will not be used for image network design
Xtrain = combine(itrain,Ytrain);
Xval = combine(ival,Yval);
