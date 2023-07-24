% input: 
%       data: a cell array, view_num*1, each array is n*d_v
% 
% result = EvaluationMetrics(labels, y);
% res = [acc, nmi, Pu, Fscore, Precision, Recall, ARI];
%
close all; clear all; clc
warning off

currentFolder = pwd;
rng(12)
betas = 1e-2;
gammas = 1e-1;
ITERS = 1;
percentDels = 0.1;
is_missing = 1;
Dataname = 'BBCSport';
load(Dataname);
data = fea';
labels = gt;
clear fea gt
num_cluster = length(unique(labels));
knns = [5*num_cluster];
[num_views,~] = size(data);
    
for i_perDel = 1:length(percentDels)
    X = data;
    percentDel = percentDels(i_perDel);
    Datafold = [Dataname,'_percentDel_',num2str(percentDel),'.mat'];
    load(Datafold);
    filename = strcat('TTGL-',Dataname,'-',num2str(percentDel),'.txt');
    iter_folds = 1;
    ind_folds = folds{iter_folds};
    W = cell(1,num_views);
    Y = cell(num_views,1);
    X_exist = cell(num_views,1);
    for iv = 1:num_views
        X1 = X{iv};
        ind_0{iv} = find(ind_folds(:,iv) == 0);
        ind_1{iv} = find(ind_folds(:,iv) == 1);
        X1(ind_0{iv},:) = [];  
        Y{iv} = X1;         
    end
    clear X1 ind_0
    X = Y;
    clear Y

    for i_b = 1:length(betas)
        beta = betas(i_b);
        for i_g = 1:length(gammas)
            gamma = gammas(i_g);
             res = zeros(ITERS,7);
            for i_knn = 1:length(knns)
                knn = knns(i_knn);
                num_data = size(X{1},1);
                for j =1:ITERS
                    [result,S,Tim] = TTGL(X,labels,beta,knn,gamma,is_missing,ind_folds);
                    t = toc;
                    res(j,:)=result;
                    fprintf('beta = %d, gamma = %d\n',beta,gamma);
                    result
                    dlmwrite(filename,[beta,gamma,knn, result t],'-append','delimiter','\t','newline','pc');
                end
            end
        end
    end
end

% clear data labels
% save result.mat   