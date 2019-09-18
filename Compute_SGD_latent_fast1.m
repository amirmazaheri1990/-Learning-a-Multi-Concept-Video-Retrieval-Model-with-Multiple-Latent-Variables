function [f,g] =  Compute_SGD_latent_fast2(V0,indexSet,TrainSamples, Q, all_phi_full_matrix,TrainSet,videosToIndex,Lambda)
V0 = reshape(V0,60,length(V0)/60);
f = 0;
g = zeros(size(V0));

tic
scores = V0*all_phi_full_matrix';
%%scores(a,s) means the score of concept a for concept s

s_g = indexSet;
s_matrix = indexSet;

for i=1:length(Q),%% is is on query
    q = Q{i};
    for j=1:size(indexSet{i},1),%% j is on the class
        for k=1:size(indexSet{i},2),%% k is on the videos
            %             if(i== 1&& j==  3&& k==15),
            %                 keyboard;
            %             end
            if(isempty(indexSet{i}{j,k})),
                continue;
            end
            
            Tindexes = indexSet{i}{j,k};
            Tscores = scores(q,Tindexes);
            %%using hard max to get the s
            s_matrix{i}{j,k} = sum(max(Tscores'));
            
            %%% find the gradient using the softmax
            Tfeatures = all_phi_full_matrix(Tindexes,:);
            
            sumOnH = sum(exp(Tscores)');%% 1 by length q
            sumOnH = sumOnH';
            
            coeffs = exp(Tscores)./repmat(sumOnH,1,size(Tscores,2));%%length Q by number of shots
            
            
            s_g{i}{j,k} = coeffs*Tfeatures;
            
            
        end
    end
end
all_f = cell(1,length(Q));
all_g = cell(1,length(Q));


parfor query =1:length(Q),
    
    
    q = Q{query};
    all_g{query} = zeros(length(q),size(g,2));
    all_f{query} = 0;
    %% First Pairs of Positives and negatives
    better=1;
    worse = 3;
    for v1=1:length(TrainSamples{query}{better}),
        p1 = TrainSamples{query}{better}(v1);
        for v2=1:length(TrainSamples{query}{worse}),
            p2 = TrainSamples{query}{worse}(v2);
            loss = max(0,s_matrix{query}{worse,p2}+1-s_matrix{query}{better,p1});
            if(loss>0),
                all_g{query} = all_g{query} + s_g{query}{worse,p2}-s_g{query}{better,p1};
                all_f{query} = all_f{query}+ loss;
            end
        end
    end
    
    
    %% second pairs of middle and negatives
    better =2;
    worse =3;
    
    for v1=1:length(TrainSamples{query}{better}),
        p1 = TrainSamples{query}{better}(v1);
        for v2=1:length(TrainSamples{query}{worse}),
            p2 = TrainSamples{query}{worse}(v2);
            loss = max(0,s_matrix{query}{worse,p2}+1-s_matrix{query}{better,p1});
            if(loss>0),
                all_g{query} = all_g{query} + s_g{query}{worse,p2}-s_g{query}{better,p1};
                all_f{query} = all_f{query}+ loss;
            end
        end
    end
    
    
    
    
    %% Third Pairs of Positives and middle
    better=1;
    worse = 2;
    
    for v1=1:length(TrainSamples{query}{better}),
        p1 = TrainSamples{query}{better}(v1);
        for v2=1:length(TrainSamples{query}{worse}),
            p2 = TrainSamples{query}{worse}(v2);
            loss = max(0,s_matrix{query}{worse,p2}+1-s_matrix{query}{better,p1});
            if(loss>0),
                all_g{query} = all_g{query} + s_g{query}{worse,p2}-s_g{query}{better,p1};
                all_f{query} = all_f{query}+ loss;
            end
        end
    end
    
    
    
    
    
    
end

for i=1:length(Q),
    f =  f + all_f{i};
    g(Q{i},:) = g(Q{i},:) + all_g{i};
end


V0(:,61) = zeros(60,1);
V0 = V0(:);
g = g(:);
f = f + Lambda*(V0'*V0)/2;
g = g +  Lambda*V0;
end