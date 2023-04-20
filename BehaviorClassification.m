% Behavioral classification of DeepLabCut data for Huang et al. (2023)
%
% Data sources
%   - tmpframe.csv         : frame number file extracted from ffii file in FreezeFrame (for detection of dropped frame(s))
%   - freezing2s.csv       : freezing of observer mice every 2s
%
% Depencency
%   - 'clusterWaterShed.m' from https://github.com/murthylab/pulseTypePipeline
%   - Statistics and Machine Learning Toolbox

%% t-SNE analysis for 10s bouts and 2s bouts of DeepLabCut data 

load('tmpframe.csv')
frame = tmpframe + 1;
path = 'WT'; % DeepLabCut csv files under this folder

% 10s (MATLAB 2020a)
fpb = 75; % 10s: 75 (7.5Hz x 10s), 2s: 15 (7.5Hz x 2s)
ff45 = 9; % FreezeFrame version 4/5 
[dataBout, node, centerMov, centerPosi, nFrame] = preprocess(path, fpb,ff45, frame, 1);

rng default % for reproducibility
tsneWT = tsne(dataBout,'Algorithm','barneshut', 'Perplexity', 30);
[cluWTRaw10s, XX, boundaryY, boundaryX, Z0, L] = clustering(tsneWT,0.5);
cluWT10s = bijection(cluWTRaw10s, [4 5 3 6 7 1 8 2]);

cmap = [ 242 78 98; 241 129 78; 252 234 30; 191 211 108; 91 181 231; 71 102 133; 148 65 149; 219 103 172]/255.0;
drawCluster(tsneWT, cluWT10s, XX, boundaryX, boundaryY, Z0, cmap)

% 2s (MATLAB 2022a)
[movId, freezId] = foundMFId('freezingWT2s.csv');
fpb = 15; % 2s: 15 (7.5Hz x 2s)
ff45 = 9; % FreezeFrame version 4/5 
[movBout, nodeD, centMovD, centPosD, freezBout, nodeR, centMovR, centPosR] = ...
    preprocess2s(movId, freezId, path, fpb, ff45, frame);

[cluWTRaw2s, cluWTM, cluWTF, tsneWTM, tsneWTF, XXM, boundaryYM, boundaryXM, LM, XXF, boundaryYF, boundaryXF, LF] = ...
    tSNE2s(movBout, freezBout, movId, freezId);
cluWT2s = bijection(reshape(cluWTRaw2s,[],1), [4 5 8 1 2 3 6 7 9 10 12 14 16 17 18 11 13 15]);

%% Positioning new data on t-SNE map (MATLAB 2020a)

% ex. mPFC inhibition
load('tmpframe_opt.csv')
frame_opt = tmpframe_opt + 1;
path = 'opto mPFC';
fpb = 75; % 10s: 75 (7.5Hz x 10s)
ff45 = 8; % FreezeFrame version 4/5 
[dataBoutmPFC, ~, centerMovmPFC,  ~, ~] = preprocess(path, fpb, ff45, frame_opt, 1);
posOpto = posiOnTsne(dataBout, dataBoutmPFC, tsneWT, 10, 100);

cluOptoRaw = NaN(size(posOpto,1),1);
for n = 1:size(posOpto,1)
    cluOptoRaw(n) = L(findnearest(posOpto(n,2), XX{2}), findnearest(posOpto(n,1), XX{1}));
end
cluOpto = bijection(cluOptoRaw, [4 5 3 6 7 1 8 2]);


%% functions
function [sumTsneVector, sumNode, sumCenterPosi, sumMovingDis, nFrames] = preprocess(path, Bout, first, mydroppedframe, mode)
    % transform the DLC data into tsne vector
    cd(path)
    list = dir('*.csv');
    sumNode = [];
    sumTsneVector = [];
    sumCenterPosi = [];
    sumMovingDis = [];
    nFrames = zeros(1,length(list));
    nFrame = 6750;
    
    for n = 1:length(list)
        disp(extractBefore(list(n).name,'DLC'))
        dlcRaw= readmatrix(fullfile(pwd,list(n).name),'Range',[4 2]);
        nFrames(n) = size(dlcRaw,1);
        nPoints = size(dlcRaw,2)/3;
        dlc_re = reshape(dlcRaw,[nFrames(n) 3 nPoints]);
        disp(size(dlc_re))
        dlc = zeros(nFrame, 3, nPoints);
        
        % use the mean value for the dropped frames
        nDropped = sum(isnan(mydroppedframe(n,:)));
        if nDropped > 0
            for k = 1:6750-nDropped
                dlc(mydroppedframe(n,k),:,:) = dlc_re(k,:,:);
            end
            dropped_frame = find(ismember(1:6750,mydroppedframe(n,:))==0);
            for k = 1:nDropped
                dlc(dropped_frame(k),:,:) = NaN;
            end
            for i = 1:nPoints
                dlc(:,:,i) = fillmissing(dlc(:,:,i),'linear');
            end
        else
            dlc = dlc_re;
        end
        
        dlc = dlc(1:nFrame,:,:);

        % linear interpolation with low probablity points 
        thresh = 0.9999;
        dlcFilt = zeros(nFrame,2,nPoints);
        dlcFiltFilled = zeros(nFrame,2,nPoints);
        for i=1:nPoints
            myframe = dlc(:,3,i) < thresh; 
            dlcFilt(:,:,i) = dlc(:,1:2,i);
            % resize to the previous size
            if n > first
                dlcFilt(:,:,i) = dlcFilt(:,:,i)/2;
            end
            dlcFilt(myframe,:,i) = NaN;
            dlcFiltFilled(:,:,i) = fillmissing(dlcFilt(:,:,i),'linear');
        end
       
        % change coordinate to vector from body center
        dlcVec = zeros(nFrame,2,nPoints);
        for i=1:nPoints
            dlcVec(:,:,i) = dlcFiltFilled(:,:,i)-dlcFiltFilled(:,:,9);
        end

        % geometric restriction : y coordiante of tail root is larger than that of body center 
        dlcVecRe = dlcVec;
        if dlcVecRe(1,2,12) < dlcVecRe(1,2,9)
            dlcVecRe(1,2,12) = dlcVecRe(1,2,9);
        end
        for k = 1:nFrame-1
            if dlcVecRe(k+1,2,12) < dlcVecRe(k+1,2,9)
                dlcVecRe(k+1,:,12) = dlcVecRe(k,:,12);
            end
        end

        nBout = floor(nFrame/Bout);
        centerPosi = dlcVecRe + dlcFiltFilled(:,:,9);
        sumCenterPosi = vertcat(sumCenterPosi, centerPosi(1:Bout*nBout,:,9));            
        
        % dlcVecBout : transform coordinate to vector from the back center of the first frame in 10 s
        % dlcVecFrame  : transform coordinate to vector from the back center of the each frame in 10 s
        dlcVecBout = zeros(nFrame,2,nPoints);
        dlcVecFrame = zeros(nFrame,2,nPoints);
        for i = 1:nPoints
            for j = 1:nBout
                t = (j-1)*Bout + 1;
                if mode == 1
                    % difference from back center
                    dlcVecBout(t:t+Bout-1,:,i) = centerPosi(t:t+Bout-1,:,i)-centerPosi(t,:,9);
                elseif mode == 2
                    % raw data
                    dlcVecBout(t:t+Bout-1,:,i) = centerPosi(t:t+Bout-1,:,i);
                end
                dlcVecFrame(t:t+Bout-1,:,i) = dlcFilt(t:t+Bout-1,:,i)-dlcFilt(t:t+Bout-1,:,9);
            end    
        end
        sumMovingDis = vertcat(sumMovingDis, dlcVecBout(1:Bout*nBout,:,9));

        % for illustration
        node = zeros(Bout*nBout,2,5);
        node(:,:,1) = dlcVecFrame(1:Bout*nBout,:,4);       % earL
        node(:,:,2) = dlcVecFrame(1:Bout*nBout,:,5);       % earR
        node(:,:,3) = mean(node(1:Bout*nBout,:,1:2),3);    % neck
        node(:,:,4) = dlcVecFrame(1:Bout*nBout,:,9);       % body
        node(:,:,5) = dlcVecFrame(1:Bout*nBout,:,12);      % tail
        sumNode = vertcat(sumNode, node(1:Bout*nBout,:,:));

        tsneVector = zeros(nFrame,nPoints*2);
        for i = 1:nPoints
            tsneVector(:,i) = dlcVecBout(:,1,i);
            tsneVector(:,i+nPoints) = dlcVecBout(:,2,i);
        end
        tmpTsneVector = reshape(tsneVector(1:Bout*nBout,:)',[nPoints*2*Bout,nBout])';
        sumTsneVector = vertcat(sumTsneVector,tmpTsneVector);
    end
    cd ..
end % END OF preprocess()

function [movingBoutsId, freezingBoutsId] = foundMFId(file)
    freezingBout2s = readmatrix(file);
    block2s = reshape(freezingBout2s',[],1);
    movingBoutsId = find(block2s<50);
    freezingBoutsId = find(block2s>=50);
end % END OF foundMFId()

function [movingBout, nodeD, centerMovD, centerPosiD, freezingBout, nodeR, centerMovR, centerPosiR] = preprocess2s(movingBoutsId, freezingBoutsId, path, frameperbout, ff4to5, tmpframe)   
    % mode 1: Difference from the first frame (for mobile bouts)
    % mode 2: Raw number (for immobile bouts)
    [dataBoutD, nodeD, centerMovD, centerPosiD] = preprocess(path, frameperbout,ff4to5, tmpframe, 1);
    movingBout = dataBoutD(movingBoutsId,:);
    [dataBoutR, nodeR, centerMovR, centerPosiR] = preprocess(path, frameperbout,ff4to5, tmpframe, 2);
    freezingBout = dataBoutR(freezingBoutsId,:);
end % END OF preprocess2s()

function [cluster, XX, boundaryY, boundaryX, Z0, L] = clustering(tsneRes, resolution)
    [cluster, ~, Z0, L, L0, XX] = clusterWaterShed(tsneRes, 100, resolution);
    [boundaryY,boundaryX] = find(L0==0);
end % END OF clustering()

function [clu, cluM, cluF, tsneM, tsneF, XXM, boundaryYM, boundaryXM, LM, XXF, boundaryYF, boundaryXF, LF] = tSNE2s(movBout, freezBout, movId, freezId)
    rng default % for reproducibility
    tsneM = tsne(movBout,'Algorithm','barneshut', 'Perplexity', 30);
    [cluM, XXM, boundaryYM, boundaryXM, Z0M, LM] = clustering(tsneM,0.64);
    rng default
    tsneF = tsne(freezBout,'Algorithm','barneshut', 'Perplexity', 30);
    [cluF, XXF, boundaryYF, boundaryXF, Z0F, LF] = clustering(tsneF,0.8);

    cmap1 = [ 230 231 232; 230 231 232; 230 231 232; 255 32 14; 255 119 1; 230 231 232; 230 231 232; 253 246 52; 230 231 232]/255;
    cmap2 = [ 188 190 192; 28 187 185; 188 190 192; 51 120 238; 188 190 192; 13 71 215; 188 190 192; 188 190 192; 188 190 192]/255;

    drawCluster(tsneM, cluM, XXM, boundaryXM, boundaryYM, Z0M, cmap1)
    drawCluster(tsneF, cluF, XXF, boundaryXF, boundaryYF, Z0F, cmap2)

    clu(movId) = cluM;
    clu(freezId) = cluF+9;
    clu = transpose(reshape(clu,450,[])); 
end

function A = bijection(A,order)
    S = sparse(A,1:numel(A),1);
    S = S(order,:);
    [A,~] = find(full(S));
end % END OF bijection() 

function drawCluster(tSNE, cluster, XX, boundaryX, boundaryY, Z0, cmap)
    figure
    subplot(1,2,1)
    gscatter(tSNE(:,1),tSNE(:,2),cluster,cmap)
    hold on
    plot(XX{1}(boundaryX),XX{2}(boundaryY),'.k', 'Markersize', 6)
    subplot(1,2,2)
    imagesc(XX{1}, XX{2}, Z0)
    hold on
    plot(XX{1}(boundaryX),XX{2}(boundaryY),'.w', 'Markersize', 6)
    colormap(jet)
    axis(gcas, 'square', 'tight', 'off', 'xy')
end % END OF drawCluster()

function newPosi = posiOnTsne(refData, newData, refPosi, knn, batchsize)
    % map new point on to the previous atlas by calculating correlations
    newPosi = zeros(size(newData,1), 2);
    nBatch = ceil(size(newData,1)/batchsize);
    for b = 1:nBatch
        batch = int64(linspace((b-1)*batchsize+1, min(b*batchsize, size(newData,1)), min(b*batchsize, size(newData,1))-(b-1)*batchsize));
        C = corr(newData(batch,:)', refData');
        for i = 1:length(batch)
            [~,ind] = maxk(C(i,:),knn);
            newPosi(batch(i),:) = median(refPosi(ind,:));
        end
    end
end % END OF posiOnTsne()