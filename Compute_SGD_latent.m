function [f,g] =  Compute_SGD_latent(V0,indexSet, Q, all_phi_full_matrix,TrainSet,videosToIndex,Lambda)
V0 = reshape(V0,60,length(V0)/60);
f = 0;
g = zeros(size(V0));


scores = V0*all_phi_full_matrix';
%%scores(a,s) means the score of concept a for concept s


for query =1:length(Q),
    q = Q{query};
    %% First Pairs of Positives and negatives
    better=1;
    worse = 3;
    for p1=1:length(videosToIndex),
        if(~ismember(videosToIndex(p1),TrainSet)),
            continue; %this video is not in the training set
        end
        if (isempty(indexSet{query}{better,p1})),
            continue; % this video is not a positive for this query
        end
        
        positive_indexes = cell2mat(indexSet{query}(better,p1));
        positive_scores = scores(q,positive_indexes);
        for p2=1:length(videosToIndex),
            if(~ismember(videosToIndex(p2),TrainSet)),
                continue; %this video is not in the training set
            end
            if (isempty(indexSet{query}{worse,p2})),
                continue; % this video is not a negative for this query
            end
            
            negative_indexes = cell2mat(indexSet{query}(worse,p2));
            try
                negative_scores = scores(q,negative_indexes);
            catch
                keyboard
            end
            %%compute the loss for this pair
            
            %%hard max is used here for loss function
            s_positive = sum(max(positive_scores'));
            s_negative = sum(max(negative_scores'));
            
            loss = max(0,s_negative+1-s_positive);
            if (loss<=0),
                continue;
            else
                %%we compute the gradient
                f_positive = all_phi_full_matrix(positive_indexes,:);
                f_negative = all_phi_full_matrix(negative_indexes,:);
                %%softmax is used to get derivatives
                
                pos_part = zeros(length(q),size(f_positive,2));
                neg_part = pos_part;
                %%derivative for positive part
                sumOnH = sum(exp(positive_scores)');%% 1 by length q
                sumOnH = sumOnH';
                
                coeffs = exp(positive_scores)./repmat(sumOnH,1,size(positive_scores,2));%%length Q by number of shots
                
                
                pos_part = pos_part + coeffs*f_positive;
                
                
                %%derivative for positive par
                sumOnH = sum(exp(negative_scores)');%% 1 by length q
                sumOnH = sumOnH';
                coeffs = exp(negative_scores)./repmat(sumOnH,1,size(negative_scores,2));%%length Q by number of shots
                
                neg_part = neg_part + coeffs*f_negative;
                
                g(q,:) = g(q,:) + neg_part-pos_part;
            end
            
            
            
            f = f+ loss;
        end
    end
    keyboard
     %% second pairs of middle and negatives
    better=2;
    worse = 3;
    for p1=1:length(videosToIndex),
        if(~ismember(videosToIndex(p1),TrainSet)),
            continue; %this video is not in the training set
        end
        if (isempty(indexSet{query}{better,p1})),
            continue; % this video is not a positive for this query
        end
        
        positive_indexes = cell2mat(indexSet{query}(better,p1));
        positive_scores = scores(q,positive_indexes);
        for p2=1:length(videosToIndex),
            if(~ismember(videosToIndex(p2),TrainSet)),
                continue; %this video is not in the training set
            end
            if (isempty(indexSet{query}{worse,p2})),
                continue; % this video is not a negative for this query
            end
            
            negative_indexes = cell2mat(indexSet{query}(worse,p2));
            try
                negative_scores = scores(q,negative_indexes);
            catch
                keyboard
            end
            %%compute the loss for this pair
            
            %%hard max is used here for loss function
            s_positive = sum(max(positive_scores'));
            s_negative = sum(max(negative_scores'));
            
            loss = max(0,s_negative+1-s_positive);
            if (loss<=0),
                continue;
            else
                %%we compute the gradient
                f_positive = all_phi_full_matrix(positive_indexes,:);
                f_negative = all_phi_full_matrix(negative_indexes,:);
                %%softmax is used to get derivatives
                
                pos_part = zeros(length(q),size(f_positive,2));
                neg_part = pos_part;
                %%derivative for positive part
                sumOnH = sum(exp(positive_scores)');%% 1 by length q
                sumOnH = sumOnH';
                
                coeffs = exp(positive_scores)./repmat(sumOnH,1,size(positive_scores,2));%%length Q by number of shots
                
                
                pos_part = pos_part + coeffs*f_positive;
                
                
                %%derivative for positive par
                sumOnH = sum(exp(negative_scores)');%% 1 by length q
                sumOnH = sumOnH';
                coeffs = exp(negative_scores)./repmat(sumOnH,1,size(negative_scores,2));%%length Q by number of shots
                
                neg_part = neg_part + coeffs*f_negative;
                
                g(q,:) = g(q,:) + neg_part-pos_part;
            end
            
            
            
            f = f+ loss;
        end
    end
    
     %% Third Pairs of Positives and middle
    better=1;
    worse = 2;
    for p1=1:length(videosToIndex),
        if(~ismember(videosToIndex(p1),TrainSet)),
            continue; %this video is not in the training set
        end
        if (isempty(indexSet{query}{better,p1})),
            continue; % this video is not a positive for this query
        end
        
        positive_indexes = cell2mat(indexSet{query}(better,p1));
        positive_scores = scores(q,positive_indexes);
        for p2=1:length(videosToIndex),
            if(~ismember(videosToIndex(p2),TrainSet)),
                continue; %this video is not in the training set
            end
            if (isempty(indexSet{query}{worse,p2})),
                continue; % this video is not a negative for this query
            end
            
            negative_indexes = cell2mat(indexSet{query}(worse,p2));
            try
                negative_scores = scores(q,negative_indexes);
            catch
                keyboard
            end
            %%compute the loss for this pair
            
            %%hard max is used here for loss function
            s_positive = sum(max(positive_scores'));
            s_negative = sum(max(negative_scores'));
            
            loss = max(0,s_negative+1-s_positive);
            if (loss<=0),
                continue;
            else
                %%we compute the gradient
                f_positive = all_phi_full_matrix(positive_indexes,:);
                f_negative = all_phi_full_matrix(negative_indexes,:);
                %%softmax is used to get derivatives
                
                pos_part = zeros(length(q),size(f_positive,2));
                neg_part = pos_part;
                %%derivative for positive part
                sumOnH = sum(exp(positive_scores)');%% 1 by length q
                sumOnH = sumOnH';
                
                coeffs = exp(positive_scores)./repmat(sumOnH,1,size(positive_scores,2));%%length Q by number of shots
                
                
                pos_part = pos_part + coeffs*f_positive;
                
                
                %%derivative for positive par
                sumOnH = sum(exp(negative_scores)');%% 1 by length q
                sumOnH = sumOnH';
                coeffs = exp(negative_scores)./repmat(sumOnH,1,size(negative_scores,2));%%length Q by number of shots
                
                neg_part = neg_part + coeffs*f_negative;
                
                g(q,:) = g(q,:) + neg_part-pos_part;
            end
            
            
            
            f = f+ loss;
        end
    end
end

V0(:,61) = zeros(60,1);
V0 = V0(:);
g = g(:);
f = f + Lambda*(V0'*V0)/2;
g = g +  Lambda*V0;
end