% Detection of cluster and component neurons
%
% Data sources
%   - Event.Data        : calcium event data
%   - Video.Cluster     : behavior cluster (components) obtained with behavior classification (tSNE.m)

data = Event.Data{1}';
cluster = Video.Cluster(1,:);
cluN = 8; % 18 for 2s components

data(data == 0) = NaN;

nNeuron = size(data,1);
data_mat = zeros(nNeuron, 22500);
Ca_sec = zeros(nNeuron,900);
Ca_bout = zeros(nNeuron,90); % 450 for 2s components

for n = 1:nNeuron
    len = sum(~isnan(data(n,:)));
    for k = 1:len
        data_mat(n, data(n,k)) = 1;
    end
end

for n = 1:nNeuron
    for sec = 1:900
        Ca_sec(n, sec) = sum(data_mat(n, (sec-1)*25+1:sec*25));
    end
    for bout = 1:90 % 450 for 2s components
        Ca_bout(n, bout) = sum(Ca_sec(n, (bout-1)*10+1:bout*10)); % 2 for 2s components
    end
end
 

cluster_event = zeros(nNeuron,cluN);
for clu = 1:cluN
    ind = find(cluster==clu);
    for n = 1:nNeuron
        cluster_event(n,clu) = sum(Ca_bout(n,ind));
    end
end

data_shuffled = zeros(nNeuron*1000, 1000);
for n = 1:nNeuron
    for m = 1:1000
        rand_tmp = randi(22500);
        data_shuffled((n-1)*1000+m,:) = mod(data(n,:)+rand_tmp,22500)+1;
    end
end

cluster_event_shuffled = zeros(nNeuron*1000,cluN);
for n = 1:nNeuron*1000
    data_mat_shuffled = zeros(1, 22500);
    len = sum(~isnan(data_shuffled(n,:)));
    for k = 1:len
        data_mat_shuffled(1, data_shuffled(n,k)) = 1;
    end

    Ca_bout_shuffled = zeros(1,90); % 450 for 2s components
    for bout = 1:90 % 450 for 2s components
        Ca_bout_shuffled(1, bout) = sum(data_mat_shuffled(1, (bout-1)*250+1:bout*250)); % 50 for 2s components
    end

    for clu = 1:cluN
        ind = find(cluster == clu);
        cluster_event_shuffled(n,clu) = sum(Ca_bout_shuffled(1,ind));
    end
end

alpha = 0.05;
pval = zeros(nNeuron,2);
for n = 1:nNeuron
    for k = 1:cluN
        pval(n,k) = sum(cluster_event_shuffled(:,k)>cluster_event(n,k))/nNeuron/1000;
    end
end

use_ind = unique(cluster);
pval_use = pval(:,use_ind);
clu_used = length(use_ind);
res = zeros(nNeuron,clu_used);
for n = 1:nNeuron
    [pval_sorted, pval_order] = sort(pval_use(n,:));
    for k = 1:clu_used
        if pval_sorted(k)>= alpha/clu_used
            break
        else
            res(n,pval_order(k)) = 1;
        end
    end
end
sig = zeros(nNeuron,clulN);
for k = 1:clu_used
    sig(:,use_ind(k)) = res(:,k);
end