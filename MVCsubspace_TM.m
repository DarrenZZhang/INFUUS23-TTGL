% Multi-view Subspace Clustering on Topological Manifold
function [result, S, Tim] = MVCsubspace_TM(data,labels, alpha, beta, knn, lambda)
% data: cell array, view_num by 1, each array is num_samp by d_v
% num_clus: number of clusters
% num_view: number of views
% num_samp
% k: number of adaptive neighbours
% labels: groundtruth of the data, num by 1
%
if nargin < 3
    alpha = 10;
end
if nargin < 4
    beta = 10;
end
if nargin < 5
    knn = 25;
end
if nargin < 6
    lambda = 1000;
end
num_view = size(data,1);
num_samp = size(labels,1);
num_clus = length(unique(labels));
Wv = 1/num_view*ones(1,num_view);
NITER = 30;
zr = 1e-20;

% =====================   Normalization =====================
normData = 1;
% if normData == 1
%     for i = 1 :num_view
%         for  j = 1:num_samp
%             normItem = std(data{i}(j,:));
%             if (0 == normItem)
%                 normItem = eps;
%             end
%             data{i}(j,:) = (data{i}(j,:) - mean(data{i}(j,:)))/normItem;
%         end
%     end
% end

if normData == 1
    for iv = 1:num_view
        data{iv} = NormalizeFea(data{iv},1);
    end
end

%  ====== Initialization =======
% claculate G_v for all the views
Z = cell(num_view,1);
L = cell(num_view,1);
sumZ = zeros(num_samp);
for v = 1:num_view
%     Zv = constructW_PKN(data{v}',knn);
    Zv = calcSim(data{v},data{v},knn);
    Dv = diag(sum(Zv));
    Lv = Dv - Zv;
    Z{v} = Zv;
    L{v} = Lv;
    D{v} = Dv;
    sumZ = sumZ + Zv;
    clear Zv Lv
end 
 
% initialize S
S = sumZ/num_view;
% initialize F
S0 = S-diag(diag(S));
w0 = (S0+S0')/2;
D0 = diag(sum(w0));
L0 = D0 - w0;
[F0,~,~] = eig1(L0,num_clus,0);
F = F0(:,1:num_clus);
I = eye(num_samp);

% update ...
tic;
for iter = 1:NITER
    %
    % update Z_v
%     Z = cell(num_view,1);
%     L = cell(num_view,1);
%     for v = 1:num_view
%         K = data{v}*data{v}';
%         Z{v} = (2*K+2*alpha*I)\(2*K+Wv(v)*((D{v}^(-1/2)*S)*(D{v}^(-1/2)*S)'));
%         Z{v}(Z{v}<0) = 0; 
%         Z{v} = (Z{v} + Z{v}')/2;
%         Z{v} = Z{v} - diag(diag(Z{v}));
%         L{v} = diag(sum(Z{v})) - Z{v};
%     end
    %
    % update S
    iniS = S;
    S = zeros(num_samp);
    A = beta*I;
%     A = zeros(num_samp);
    for v = 1:num_view
        A = A + Wv(v)* L{v};
    end
    dist_u = L2_distance_1(F',F');
    for ni = 1:num_samp
        index = find(iniS(ni,:)>0);
        b = 2*beta*I(ni,index) - lambda*dist_u(ni,index);
%         b = - lambda*dist_u(ni,index);
        % solve z^T*A*z-z^T*b
        [si, ~] = fun_alm(A(index,index),b);
        S(ni,index) = si';
    end
    S = (S+S')/2; 
    %
    % update w_v
    for v = 1:num_view
        Wv(v) = 1/(2*sqrt(trace(S'*L{v}*S)));
        % Wv(v) = 1/(2*sqrt(trace(S'*L{v}*S))+1); % (0,1]
    end
    %
    % calculate objective value

    obj = beta*norm(S-I,'fro')^2;
    for v = 1:num_view
        obj = obj + Wv(v)*sqrt(trace(S'*L{v}*S));
        % obj = obj + Wv(v)*sqrt(trace(S'*L{v}*S)) + norm(data{v}'-data{v}'*Z{v},'fro')^2 + alpha*norm(Z{v},'fro')^2;
    end
    OBJ(iter) = obj;
    
    % update F                                                   
    LS = diag(sum(S)) - S;
    F_old = F;
    [F, ~, ev] = eig1(LS, num_clus, 0);
%     fn1 = sum(ev(1:num_clus));
%     fn2 = sum(ev(1:num_clus+1));
%     if fn1 > zr
%         lambda = 2*lambda;
%     elseif fn2 < zr
%         lambda = lambda/2;  
%         F = F_old;
%     else
%         fprintf('the %d -th iteration -> end ...\n',iter)
%         break;
%     end 
%     if i>5 &((norm(S-iniS)/norm(iniS))<1e-3)
%         break
%     end
end
fprintf('iter:%d\n',iter);
Tim = toc;
% =====================  result =====================
[clusternum, y]=graphconncomp(sparse(S)); 
y = y';
if clusternum ~= num_clus
    sprintf('Can not find the correct cluster number: %d', num_clus)
end
% y = litekmeans(F,num_clus);
result = EvaluationMetrics(labels, y);
end


function [v, obj] = fun_alm(A,b)
if size(b,1) == 1
    b = b';
end

% initialize
rho = 1.5;
mu = 30;
n = size(A,1);
alpha = ones(n,1);
v = ones(n,1)/n;
% obj_old = v'*A*v-v'*b;

obj = [v'*A*v-v'*b];
iter = 0;
while iter < 10
    % update z
    z = v-A'*v/mu+alpha/mu;

    % update v
    c = A*z-b;
    d = alpha/mu-z;
    mm = d+c/mu;
    v = EProjSimplex_new(-mm);

    % update alpha and mu
    alpha = alpha+mu*(v-z);
    mu = rho*mu;
    iter = iter+1;
    obj = [obj;v'*A*v-v'*b];
end
end


function [x] = EProjSimplex_new(v, k)
%
% Problem
%
%  min  1/2 || x - v||^2
%  s.t. x>=0, 1'x=k
%
if nargin < 2
    k = 1;
end;

ft=1;
n = length(v);

v0 = v-mean(v) + k/n;
%vmax = max(v0);
vmin = min(v0);
if vmin < 0
    f = 1;
    lambda_m = 0;
    while abs(f) > 10^-10
        v1 = v0 - lambda_m;
        posidx = v1>0;
        npos = sum(posidx);
        g = -npos;
        f = sum(v1(posidx)) - k;
        lambda_m = lambda_m - f/g;
        ft=ft+1;
        if ft > 100
            x = max(v1,0);
            break;
        end
    end
    x = max(v1,0);

else
    x = v0;
end
end
