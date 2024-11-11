function [Xtrain,Xval]=create_datastores()
%% Locations
comp_loc1 = "../1_Alloy_Data/compositions.xls";
ele_loc1 = "../1_Alloy_Data/elemental_features.xlsx";
labDir1 = comp_loc1;
im_loc1 = "../1_Alloy_Data/images_original/";
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
[xldat]=xlsread(labDir1);
%% Import Images
imds = imageDatastore(im_loc1,"FileExtensions",[".png"]);
%% Normalize input itself
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
%% Scale labels 
% for i=1:nfiles
%       fv(i,:,:) = normalize(fv(i,:,:),3,'scale');
%     for y=1:11
%         mi = min(fv(i,:,y));
%         ma = max(fv(i,:,y));
%         for x=1:21
%             fv2(i,y,x)=fv(i,y,x)/(ma-mi);
%         end
%     end
% end
%% flatten elemental features
%for i=1:1962
for i=1:109
    fv1 = zeros([11,21]);
    fv1(:,:)=fv(i,:,:);
    %reshape(fv1.',1,[]);
    fvflat(i,:) = fv1(:);
end

%fvflat = normalize(fvflat,1,'range',[0,1]);
%% Array Datastore
fds = arrayDatastore(fvflat);
%% Make datastores and subset them into train test val randomly
hardness = hardness';
n = size(hardness,1);
pv = 20;
hds = arrayDatastore(hardness);
%[ntrain,nval] = gen_rand_indices(n,pv);
split = load("datasplit_comp.mat");
ntrain= split.ntrain;
nval = split.nval;
Yval = subset(hds,nval);
Ytrain = subset(hds,ntrain);
ival = subset(imds,nval);
itrain = subset(imds,ntrain);
fval = subset(fds,nval);
ftrain = subset(fds,ntrain);
Xtrain = combine(ftrain,Ytrain);
Xval = combine(fval,Yval);
Xfull = combine(imds,fds,hds);
