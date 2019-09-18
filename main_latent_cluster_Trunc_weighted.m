function[] = main_latent_cluster1(id)
id = str2num(id);


% data_path = '/Users/amirmazaheri/Google Drive/Research/Project-with-Amir/Codes/ranksvm/data/';
% output_path = './';
data_path = '/home/amirmaz/Research/ranksvm/data/';
output_path = '/home/amirmaz/cluster_ranksvm/output/';
    %load('../../data/Binary_queries.mat');
    load([data_path,'indexset14.mat']);
    load([data_path,'Q_len2.mat']);
    load([data_path,'all_phi_full_matrix14.mat']);
    %load([data_path,'GT14.mat']);
    load([data_path,'AllSets.mat']);
    load([data_path,'scores_devided.mat']);


addpath(genpath('../../../minFunc_2012'));

%%%
disp('loading is completed');
if(exist('options','var')),
    clear options;
end
options.Display = 'iter'; %'off';
options.Method = 'lbfgs'; %'newton0';
options.optTol = 1e-10;
options.progTol = 1e-10;
options.MaxIter = 500;
options.MaxFunEvals = 500;
%options.DerivativeCheck = 'on';
W0 = eye(60);




%% Building Lambdas
Lambdas = -5:1:15;
Lambdas = 2.^Lambdas;
Lambdas = [0,Lambdas];

%%
%  Q2{1} = Q{1};
%  %Q2{2} = Q{2};
%  Q = Q2;

%% 
usingTrainingSample = TrainSet_Train;
TrainSamples = TrainSampleBuilder(indexSet,usingTrainingSample,videosToIndex);


%% optimization
%%%% Addin a bias column to the end of the feature vectors
all_phi_full_matrix = [  all_phi_full_matrix,ones(size(all_phi_full_matrix,1),1)];
W0 = [W0,zeros(60,1)];
%% delete the above line if no need for bias function also last argument is Lambda
%%%Lambdas[12] is zero.
funObj = @(V0)Compute_SGD_latent_Trunc_weighted(V0,indexSet,TrainSamples, Q, all_phi_full_matrix,usingTrainingSample,videosToIndex,Lambdas(id));
disp('start');
V0 = W0;

%V0 = [zeros(60,5)];
[W0, ~, ~,~] = minFunc(funObj, V0(:),options);
W_MSSV = W0;

W_MSSV = reshape(W_MSSV,60,length(W_MSSV)/60);
save([output_path,'Trunc_withBias_',num2str(id)],'W_MSSV');

end