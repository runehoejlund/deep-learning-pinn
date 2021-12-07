clc; clear all; close all;

%% Burger's Equation Solver
% Group 9: Maximiliam Zwicker (s161063) and Øyvind Winton (s160602)
% FEM 2020, week 3

%% Definition
L   = 1;
P   = 5;
c   = 0;
d   = 0;
M   = 60;   % Number of Elements
n   = M +1;
Me = M;

% Time stepping
time    = 0.5; % final time
init    = @(x) -sin(x*pi);

dt      = 0.008;
dt      = 0.0005;
eps     = 0.01/pi;
steps   = round(time/dt);


% Uniform mesh
uniform     = true; % uniform/non-uniform mesh
OD45calc    = false; % OD45 or time stepping with for loop (time step min set to 0.008)

%%
if uniform
    x  = linspace(-L,L,n);
else
    x1  = linspace(-L,- L/3,round(n*0.2));
    x2  = linspace(-L/4,L/4,round(n*0.6));
    x3  = linspace(L/3,L,round(n*0.2));
    x = [x1 x2 x3];
    if (dt>0.008)
        dt = 0.008
    end
end
% x           = JacobiGL(0,0,n-1);
EToV(:,1)   = 1:n-1;
EToV(:,2)   = 2:n;
hn          = diff(x);

[x,EToV]    = reordermesh1D(x,EToV);
[C]         = ConstructConnectivity1D(EToV,P);

r           = linspace(-1,1,P+1);
r           = JacobiGL(0,0,P);
[x1d]       = mesh1D(x,EToV,r);


% Build finite element matrix
[A]     = buildA(hn,P,r,C);
[M]     = buildM(hn,P,r,C);
[S]     = buildS(hn,P,r,C);

S(1,:)  = 0; S(end,:)   = 0;
A(1,:)  = 0; A(end,:)   = 0;
M(1,:)  = 0; M(end,:)   = 0;
M(:,1)  = 0; M(:,end)   = 0;
M(1,1)  = 1; M(end,end) = 1;


if OD45calc
    
    odeFun  = @(t,u) (-1/2 * (M\S) * u.^2 - eps*(M\A)*u);
    tspan   = linspace(0,0.5105,5);
    y0      = init(x1d);
    [t,u] = ode45( odeFun , tspan , y0);
    
else
    
    u = init(x1d);
    for t = 2:steps
        u = u + dt * (-1/2 * (M\S) * u.^2 - eps*(M\A)*u);
        
    end
    
end

plot(x1d,u)

%% functions

function [x,EToV] = reordermesh1D(x,EToV)

x           = sort(x);
n           = length(x);
EToV(:,1)   = 1:n-1;
EToV(:,2)   = 2:n;

end

function [C] = ConstructConnectivity1D(EToV,P)

C = zeros(size(EToV,1),P+1);

gdix = 2;

for n = 1:size(EToV,1)
    gdix = gdix -1;
    for i = 1:P+1
        C(n,i) = gdix;
        gdix = gdix+1;
    end
end
end

function [x1d] = mesh1D(x,EToV,r)

length_x1d = length(x)+(length(r)-2);
x1d = [];

r = (r(:)+1)/2;

for i = 1:size(EToV,1)
    
    x1 = x(EToV(i,1));
    x2 = x(EToV(i,2));
    rL = x1 + r*(x2-x1);
    
    x1d = [x1d; rL(1:length(rL)-1)];
    
end

x1d = [x1d; rL(end)];

end

function [Kn] = diffusionmatrix1D(hn,P,r)

V   = Vandermonde1D(P,r);
DVr = GradVandermonde1D(P,r);

M   = (V*transpose(V))^(-1);
Dr  = DVr *(V)^(-1);
Kn  = 2/hn * transpose(Dr) * M * Dr;

end

function [Mn] = massmatrix1D(hn,P,r)
V   = Vandermonde1D(P,r);
Mn  = hn/2 * inv(V*transpose(V));
end

function [Sn] = stiffnessmatrix1D(hn,P,r)
V   = Vandermonde1D(P,r);
DVr = GradVandermonde1D(P,r);

M   = inv(V*transpose(V));
Dr  = DVr *(V)^(-1);

Sn = M*Dr;
end

function [A] = buildA(hn,P,r,C)

N       = size(C,1);
Mp      = P + 1;
Nnodes  = length(unique(C(:)));
A       = sparse(Nnodes,Nnodes);

for n = 1:N
    
    [Kn]    = diffusionmatrix1D(hn(n),P,r);
    ke      = Kn;
    
    for j = 1:Mp
        for i = 1:Mp
            A(C(n,i),C(n,j)) = A(C(n,i),C(n,j)) + ke(i,j);
        end
    end
end
end

function [M] = buildM(hn,P,r,C)

N = size(C,1);
Mp = P + 1;
Nnodes = length(unique(C(:)));

M = sparse(Nnodes,Nnodes);

for n = 1:N
    
    [Mn]    = massmatrix1D(hn(n),P,r);
    ke      = Mn;
    
    for j = 1:Mp
        for i = 1:Mp
            M(C(n,i),C(n,j)) = M(C(n,i),C(n,j)) + ke(i,j);
        end
    end
end
end

function [S] = buildS(hn,P,r,C)

N = size(C,1);
Mp = P + 1;
Nnodes = length(unique(C(:)));

S = sparse(Nnodes,Nnodes);
[Sn]    = stiffnessmatrix1D(hn(1),P,r);
ke      = Sn;
for n = 1:N
    for j = 1:Mp
        for i = 1:Mp
            S(C(n,i),C(n,j)) = S(C(n,i),C(n,j)) + ke(i,j);
        end
    end
end
end

function [b] = buildB(hn,P,r,C,f)

N       = size(C,1);
Mp      = P + 1;
Nnodes  = length(unique(C(:)));

B = sparse(Nnodes,Nnodes);
b = zeros(Nnodes,1);

for n = 1:N
    
    [Mn]    = massmatrix1D(hn(n),P,r);
    ke      = Mn;
    
    for j = 1:Mp
        for i = 1:j
            B(C(n,i),C(n,j)) = B(C(n,i),C(n,j)) + ke(i,j);
        end
    end
end

b = B*f;

end

function [A,b] = imposeDC(A,C,c,d,P,b)

N = size(C,1);
Mp = P + 1;
Nnodes = length(unique(C(:)));

%b = zeros(Nnodes,1);

b(1,1) = c;
A(1,1) = 1;

for i = 2:Mp
    b(i,1) = b(i,1) - A(1,i)*c;
    A(1,i) = 0;
end

b(C(N,Mp),1) = d;

for i = 1:Mp-1
    b(C(N,i),1)         = b(C(N,i),1) - A(C(N,i),C(N,Mp)) * d;
    A(C(N,i),C(N,Mp))   = 0;
end
A(C(N,Mp),C(N,Mp)) = 1;
b(1,1) = c;
end

function [A,b] = imposeNM(A,C,c,d,P,b)

N = size(C,1);
Mp = P + 1;
Nnodes = length(unique(C(:)));


end


