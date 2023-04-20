% Decoder for behavioral clusters and freezing states
% 
% Data sources
%   - Event.nEvents.ByChunk    : calcium event data of every 2s for each neuron
%   - Video.Cluster            : behavior cluster obtained with behavior classification (tSNE.m)
%   - Freezing.Obs.ByEpoch     : Freezing rate of observers every 2s

%% Multiclass SVM and 10-fold cross-validation for decoding behavioral cluster

Decode_WT.Multi.MCReps = 15; % Number of Monte-carlo shufflings
Decode_WT.Multi.nPerms = 15; % Number of permutation for statistical significance

Decode_WT.Multi.Real = struct();
Decode_WT.Multi.Perm = struct();

Decode_WT.Multi.pval = zeros(Exp.nRecordings,1);

opts = struct();
opts.Coding     = 'onevsall';
opts.Learners   = 'svm';
opts.ClassNames = 1:8;
opts.Prior      = 'uniform';
opts.Options    = statset('UseParallel',true);

rng(0) % for reproducivity

for ii=1:4
    fprintf('Processing %d/%d recordings...\n',ii,Exp.nRecordings);
    % Set predictors and class labels
    N = Event.nEvents(ii).ByChunk;
    N = reshape(N,Event.nChunksPerEpoch,Event.nEpochs,Exp.nCells(ii));
    N = permute(N,[2 1 3]);
    N = reshape(N,Event.nEpochs,Event.nChunksPerEpoch*Exp.nCells(ii));
    C = Video.Cluster(ii,:);
    % Define a random partition for 10-fold cross-validation
    cvp = cvpartition(numel(C),'KFold',10);
    % Ten-fold cross-validation and prediction
    [ModelR, Decode_WT.Multi.Real(ii).label, ...
     Decode_WT.Multi.Real(ii).confmat, ...
     Decode_WT.Multi.Real(ii).accuracy] = ...
        SVM(@fitcecoc,N,C,cvp,opts,Decode_WT.Multi.MCReps,'real');
    % Permutation test
    [ModelL, Decode_WT.Multi.Perm(ii).label, ...
     Decode_WT.Multi.Perm(ii).confmat, ...
     Decode_WT.Multi.Perm(ii).accuracy] = ...
        SVM(@fitcecoc,N,C,cvp,opts,Decode_WT.Multi.nPerms,'circshift');
    % Evaluate significance of decoder accuracy
    myp = mean(Decode_WT.Multi.Perm(ii).accuracy > mean(Decode_WT.Multi.Real(ii).accuracy));
    Decode_WT.Multi.pval(ii) = min(myp,1-myp)*2; % two-way significance test
end

clear optis ii N C cvp my*

%% Binary SVM and 10-fold cross-validation for decoding freezing states

Decode_WT.Binary.MCReps = 1000; % Number of Monte-carlo shufflings
Decode_WT.Binary.nPerms = 1000; % Number of permutation for statistical significance

Decode_WT.Binary.Real = struct();
Decode_WT.Binary.Perm = struct();

Decode_WT.Binary.pval = zeros(Exp.nRecordings,1);

gtmed = @(x)(x > median(x));

opts.Regularization = 'ridge';
opts.ClassNames = [0 1]; % for determining the size of confusion matrix

rng(0) % for reproducibility

for ii=1:4
    fprintf('Processing %d/%d recordings...\n',ii,Exp.nRecordings);
    % Set predictors and class labels
    N = Event.nEvents(ii).ByChunk;
    N = reshape(N,Event.nChunksPerEpoch,Event.nEpochs,Exp.nCells(ii));
    N = permute(N,[2 1 3]);
    N = reshape(N,Event.nEpochs,Event.nChunksPerEpoch*Exp.nCells(ii));
    C = gtmed(Freezing.Obs.ByEpoch(ii,:)); % binarized freezing
    C = double(C);
    % Define a random partition for 10-fold cross-validation
    cvp = cvpartition(numel(C),'KFold',10);
    % Ten-fold cross-validation and prediction
    [Decode_WT.Binary.Real(ii).label, ...
     Decode_WT.Binary.Real(ii).confmat, ...
     Decode_WT.Binary.Real(ii).accuracy] = ...
        SVM(@fitclinear,N,C,cvp,opts,Decode_WT.Binary.MCReps,'real');
    % Permutation test
    [Decode_WT.Binary.Perm(ii).label, ...
     Decode_WT.Binary.Perm(ii).confmat, ...
     Decode_WT.Binary.Perm(ii).accuracy] = ...
        SVM(@fitclinear,N,C,cvp,opts,Decode_WT.Binary.nPerms,'circshift');
    % Evaluate significance of decoder accuracy
    myp = mean(Decode_WT.Binary.Perm(ii).accuracy > mean(Decode_WT.Binary.Real(ii).accuracy));
    Decode_WT.Binary.pval(ii) = min(myp,1-myp)*2; % two-way significance test
end

clear N C opts cvp ii

%% Functions

function [Mdl, label,confmat,accuracy] = SVM(func,N,C,cvp,opts,nReps,method)

nClasses = numel(opts.ClassNames);

label    = zeros(numel(C),nReps);
confmat  = zeros(nClasses,nClasses,nReps);
accuracy = zeros(nReps,1);

cellopts = [fieldnames(opts) struct2cell(opts)]';

timerVal = tic; % for displaying elapsed time

for ii=1:nReps
    if mod(ii,100)==0
        fprintf('Processing %d/%d iterations...elapsed %.3f seconds.\n', ...
            ii,nReps,toc(timerVal));
    end
    switch method
        case 'randperm'
            C = C(randperm(numel(C)));
        case 'circshift'
            C = circshift(C,randi(numel(C)));
    end
    Mdl = func(N,C,'CVPartition',cvp.repartition,cellopts{:});
    % Class labels and scores predicted by the cross-validated classifier
    label(:,ii) = Mdl.kfoldPredict;
    % Confution matrix for classification
    confmat(:,:,ii) = confusionmat(C,label(:,ii));
    % Calculate decoder accuracy
    accuracy(ii) = trace(confmat(:,:,ii)) / numel(C);
end
 
end % END OF SVM()
