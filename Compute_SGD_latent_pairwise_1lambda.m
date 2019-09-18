function [f,g] =  Compute_SGD_latent_pairwise_1lambda(V0,indexSet,TrainSamples, Q, all_phi_full_matrix,TrainSet,videosToIndex,Lambda_V)
windows_size = 2; %% time windows size for second term

V0 = reshape(V0,60,length(V0)/60);
f = 0;
g= zeros(size(V0));

V0_W = V0(:,1:61);
V0_V = V0(:,62:end);
g_W = zeros(size(V0_W));
g_V = g_W;

scores_W = V0_W*all_phi_full_matrix';
V_scores = cell(1,size(V0,1));
for i=1:length(V_scores),
    V_scores{i} = repmat(V0_V(i,:),size(all_phi_full_matrix,1),1).*all_phi_full_matrix;
end
%scores_V = V0_V*all_phi_full_matrix';
%%scores(a,s) means the score of concept a for concept s

s_g_W = indexSet;
s_matrix = indexSet;

s_g_V = indexSet;


parfor i=1:length(Q),%% is is on query
    q = Q{i};
    for j=1:size(indexSet{i},1),%% j is on the class
        if(j==2),
            continue;
        end
        for k=1:size(indexSet{i},2),%% k is on the videos
            %             if(i== 1&& j==  3&& k==15),
            %                 keyboard;
            %             end
            if(isempty(indexSet{i}{j,k})),
                continue;
            end
            
            %%write a function to get indexSet cell and features
            %%give out the a
            
            Tindexes = indexSet{i}{j,k};
            Tscores = scores_W(q,Tindexes);
            %%using hard max to get the s
            
            Tfeatures = all_phi_full_matrix(Tindexes,:);
            V_scores_tmp = cell(1,length(q));
            for a=1:length(q),
                V_scores_tmp{a} = V_scores{q(a)}(Tindexes,:);
            end
            [s_matrix{i}{j,k},s_g_W{i}{j,k},s_g_V{i}{j,k}] = combination(Tscores,V_scores_tmp,Tfeatures,windows_size);   
        end
    end
end

all_f = cell(1,length(Q));
all_g_W = cell(1,length(Q));
all_g_V = cell(1,length(Q));

parfor query =1:length(Q),
    q = Q{query};
    all_g_W{query} = zeros(length(q),size(g_W,2));
    all_g_V{query} = all_g_W{query};
    all_f{query} = 0;
    %% First Pairs of Positives and negatives
    better=1;
    worse = 3;
    for v1=1:length(TrainSamples{query}{better}),
        p1 = TrainSamples{query}{better}(v1);
        for v2=1:length(TrainSamples{query}{worse}),
            N = 1/(length(TrainSamples{query}{better})*length(TrainSamples{query}{worse}));
            p2 = TrainSamples{query}{worse}(v2);
            loss = max(0,s_matrix{query}{worse,p2}-s_matrix{query}{better,p1});
            if(loss>0),
                loss = N*1;
                all_g_W{query} = all_g_W{query} + N*(s_g_W{query}{worse,p2}-s_g_W{query}{better,p1});
                all_g_V{query} = all_g_V{query} + N*(s_g_V{query}{worse,p2}-s_g_V{query}{better,p1});
                all_f{query} = all_f{query}+ loss;
                
            end
        end
    end
end

for i=1:length(Q),
    f =  f + all_f{i};
    g_W(Q{i},:) = g_W(Q{i},:) + all_g_W{i};
    g_V(Q{i},:) = g_V(Q{i},:) + all_g_V{i};
end


V0_W(:,61) = zeros(60,1);
V0_V(:,61) = zeros(60,1);
% 
% g_W = g_W +  Lambda_W.*V0_W;
% g_V = g_V +  Lambda_V.*V0_V;
V0 = [V0_W,V0_V];
V0 = V0(:);
g = [g_W,g_V];
g = g(:);


V0_W = V0_W(:);
V0_V = V0_V(:);

f= f + Lambda_V*(V0'*V0)/2;

%V0 = V0(:);
%g = g(:);
%f = f + Lambda*(V0'*V0)/2;
g = g +  Lambda_V*V0;

end