function [max_score,Tgradient_W,Tgradient_V] = combination(scores_W,V_scores,full_matrix,Window_size)
number_of_shots = size(scores_W,2);
coeffs = zeros(size(scores_W,1),number_of_shots);
%coeffs_V = zeros(size(scores_W,1),number_of_shots);
all_phi_v = zeros(number_of_shots,size(full_matrix,2));
 b_h = zeros(length(V_scores),number_of_shots);
for h=1:number_of_shots,
   
    v_bundle = max(1,h-Window_size):1:min(number_of_shots,h+Window_size);
    if(length(v_bundle)>1),
       v_bundle(v_bundle==h)=[]; 
    end
    for i=1:length(V_scores),%% 2
        if(~isvector(V_scores{i}(v_bundle,:))),
            [tmp, indexes_max_bh]= max(V_scores{i}(v_bundle,:));
        else
            tmp = V_scores{i}(v_bundle,:);
            indexes_max_bh = ones(size(tmp));
        end
        
        indexes_max_bh = v_bundle(indexes_max_bh);
        
        inds = sub2ind(size(full_matrix),indexes_max_bh,1:1:length(indexes_max_bh));
        all_phi_v(h,:) = full_matrix(inds);
        b_h(i,h) = sum(tmp);
    end
end
coeffs = scores_W + b_h;
%sum_coeffs = sum(coeffs);%%% 1 by number of possible h
%total_sum = sum(sum_coeffs);
%%%%finding the best_score
max_score = sum(max(coeffs,[],2));
%max_coeffs = max(coeffs,[],2);

%%%%finding the gradients

%%

Tgradient_W = (coeffs./repmat(sum(coeffs,2),1,size(coeffs,2)))*full_matrix;% gonna be 2*61

%%
Tgradient_V = (coeffs./repmat(sum(coeffs,2),1,size(coeffs,2)))*all_phi_v;
%Tgradient_V = all_phi_v;

end