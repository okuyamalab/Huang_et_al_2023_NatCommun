% Markov Chain Analysis
%
% Data sources
%   - cluWT10s       : behavior cluster obtained with behavior classification (tSNE.m)
%
% Dependencies
%   - 'shadedErrorBar.m' from https://jp.mathworks.com/matlabcentral/fileexchange/26311-raacampbell-shadederrorbar

cluN = 8;
clu = transpose(reshape(cluWT10s,90,[]));
len_bout = size(clu,2);

state_counts = zeros(cluN,1);
state_probabilities = zeros(cluN,1);
for s = 1:cluN
    state_counts(s) = sum(length(find(cluWT10s==s)));
    state_probabilities(s) = state_counts(s) / (length(cluWT10s)-1);
end

for state = 1:cluN
    state_chains_aligned = NaN(state_counts(state), 6);
    i = 1;
    for n = 1:size(clu,1)
        positions = find(clu(n,1:len_bout-1) == state);
        for pos = 1:length(positions)
            tmp_max = len_bout;
            tmp_start = max(1, positions(pos) - 4);
            tmp_end = min(positions(pos) + 1, tmp_max);
            state_chains_aligned(i, tmp_start-positions(pos)+5:5+tmp_end-positions(pos)) = clu(n,tmp_start:tmp_end);
            i = i +1;
        end
    end

    state_chains_aligned = state_chains_aligned(find(~isnan(state_chains_aligned(:, end))),:);
    len_st = size(state_chains_aligned,1);
    idxs = 1:len_st;
    prediction_probabilities = NaN(size(state_chains_aligned));

    for ii = 1:len_st
        % T0 probability predicted correctly under null
        p0 = state_probabilities(state_chains_aligned(ii, end));
        prediction_probabilities(ii, end) = p0;
        % T-1 first order markov prediction
        selected = [1:ii-1 ii+1:len_st];
        leave_one_out = state_chains_aligned(selected,:);
        p1 = sum(leave_one_out(:, end) == state_chains_aligned(ii, end)) / (len_st-1);
        prediction_probabilities(ii, end-1) = p1;
        % T-2~5 higher order markov predictions
        for t = 2:5
            if ~isnan(state_chains_aligned(ii, end-t))
                leave_one_out = leave_one_out(find(~isnan(leave_one_out(:, end-t))),:);
                match = [];
                for m = 1:size(leave_one_out,1)
                    if leave_one_out(m, end-t:end-1) == state_chains_aligned(ii, end-t:end-1)
                        match = [match; leave_one_out(m, :)];
                    end
                end
                if size(match,1) > 0
                    p = sum(match(:, end) == state_chains_aligned(ii, end)) / size(match,1);
                    prediction_probabilities(ii, end-t) = p;
                end
            end
        end
    end
    std_sqrt = zeros(1,6);
    for k = 1:6
        tmp_size = length(prediction_probabilities(:,7-k)) - sum(isnan(prediction_probabilities(:,7-k)));
        std_sqrt(k) = std(prediction_probabilities(:,7-k),'omitnan')/sqrt(tmp_size);
    end
    for s = 1:5
        [h,p] = ttest2(prediction_probabilities(:,end-s+1), prediction_probabilities(:,end-s));
        if h == 1
            fprintf('%d vs %d : state = %d\n', s-1, s, state)
            fprintf('p = %d\n', p)
        end
    end
    figure
    plot(flip(nanmean(prediction_probabilities)), 'ok', 'MarkerFaceColor', 'k')
    % shadedErrorBar(1:6, flip(nanmean(prediction_probabilities)), std_sqrt,'lineProps','k')
    ylim([0 1])
    xlim([0 7])
    xticks([1 2 3 4 5 6])
end