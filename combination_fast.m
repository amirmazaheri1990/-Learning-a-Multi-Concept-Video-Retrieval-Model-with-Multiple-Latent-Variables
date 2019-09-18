function [max_score,Tgradient_W,Tgradient_V] = combination_fast(scores_W,V_scores,full_matrix,Window_size)
number_of_shots = size(scores_W,2);
coeffs = zeros(size(scores_W,1),number_of_shots);
%coeffs_V = zeros(size(scores_W,1),number_of_shots);
all_phi_v = zeros(length(V_scores),size(full_matrix,2));

 b_h = zeros(length(V_scores),1);
for i=1:length(V_scores),%% 2
    if(~isvector(V_scores{i})),
        [tmp, indexes_max_bh]= max(V_scores{i});
    else
        tmp = V_scores{i};
        indexes_max_bh = ones(size(tmp));
    end
    
    %indexes_max_bh = v_bundle(indexes_max_bh);
    
    inds = sub2ind(size(full_matrix),indexes_max_bh,1:1:length(indexes_max_bh));
    all_phi_v(i,:) = full_matrix(inds);
    b_h(i,1) = sum(tmp);
end



coeffs = scores_W+repmat(b_h,1,size(scores_W,2));


sum_coeffs = sum(coeffs);%%% 1 by number of possible h
%total_sum = sum(sum_coeffs);
%%%%finding the best_score
max_score = sum(max(coeffs,[],2));
%max_coeffs = max(coeffs,[],2);

%%%%finding the gradients

%%

Tgradient_W = (coeffs./repmat(sum(coeffs,2),1,size(coeffs,2)))*full_matrix;% gonna be 2*61

%%
%Tgradient_V = (coeffs./repmat(sum(coeffs,2),1,size(coeffs,2)))*all_phi_v;
Tgradient_V = all_phi_v;

end