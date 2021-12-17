

function [] = tsai_compare_ssa ();
% Victor Tsai, Andrew Stewart, Andrew Thompson, Apr 2014
%
% Solves the steady 1D ice flow model in from Schoof (2007). Uses a similar
% numerical discretization, but drops the time derivative and just finds 
% the least-squares numerical solution.

%%% Set initial grounding line position and time parameters.
%%% Cases: case_f=1 is unstable thin ice sheet (Schoof Fig 8)
%%% case_f=2 is thick ice sheet (Schoof Fig 9)

case_f = 1;
if case_f == 1
  Nx = 500;
  x_g = 1.4;  
  fric_powerlaw = true; %%% Power law drag rather than yield stress
  fric_coulomb = false; %%% Coulomb friction at the GL
else
  Nx = 500;
  x_g = 1.5; % x_g = 1.957 should be stable    
  fric_powerlaw = true; %%% Power law drag rather than yield stress
  fric_coulomb = false; %%% No Coulomb friction
end

% c_hat = 1, puts constrain: H^2/L=C*U^M/(rho*g). Oyvind

%%% Parameters
ds = 1/(Nx-0.5);      %%% Dimensionless grid spacing
n=3;                  %%% Rheology power law exponent
m=1/3;                %%% Basal drag exponent
A = 1e-25;            %%% Dimensional extensional stress coefficient
H = 1000;             %%% Dimensional height scale
L = 1e5;              %%% Dimensional length scale
acc = 0.3/365/86400;    %%% Dimensional accumulation in m/yr
U = acc*L/H;          %%% Dimensional velocity scale
rho = 900;            %%% Ice density, kg/m^3
rho_w = 1000;         %%% Water density, kg/m^3
g = 9.8;             %%% Gravitational constant
del = 1-rho/rho_w;    %%% Ice/ocean density contrast 
eps = (U/L/A)^(1/n) / (2*rho*g*H) %%% Extensional stress parameter 
f = 0.6*L/H;          %%% Dimensionless coulomb friction parameter
% del = 0.1;          %%% Ice/ocean density contrast 
% eps = 5*10^-5;      %%% Extensional stress parameter 
% f = 600;              %%% Dimensionless coulomb friction parameter
tau0 = 1e2;           %%% Dimensionless uniform drag coefficient

C = 7.624e6           %%% C^(1/m) denoted as C. Same for c_hat. Oyvind
c_hat = L*U^m/(rho*g*H^2)*C    %%% Scaling from Tsai (2015). Oyvind


%%% Theoretical grounding line thickness
if (case_f==1)
  ['Theoretical GL scale ',num2str(eps^(n/(n+2)))]
else
  ['Theoretical GL scale ',num2str(eps^(n/(n+m+3)))]
end

%%% Coordinates/grids
%%% s is from 0 to 1, x is from 0 to x_g
% h_ is defined from ds/2 to 1 (1:N)
% u_ is staggered with h, from 0 to 1-ds/2 (1:N), where u(1)=0
s_h = [1:2:2*Nx-1]/(2*Nx-1);
s_u = [0:2:2*Nx-2]/(2*Nx-1);
x_h = s_h*x_g;
x_u = s_u*x_g;
dx = ds*x_g;

%%% Accumulation on h-gridpoints
a = det_a(x_h,case_f); 

%%% Floatation thickness
hf = det_hf(x_g,case_f,n,m,eps,del);

%%% Initial guess
if case_f == 1 
  u_init = 0.2*s_u;
else  
  h_init = (hf^(m+2) + (m+2)/(m+1)*a.^m.*(x_g.^(m+1)-x_h.^(m+1))) .^ (1/(m+2));                 
  u_init = 1*x_u./interp1(x_h,h_init,x_u);
  u_init(1) = 0;
end

%%% Find least-squares steady solution
% options = optimoptions('lsqnonlin','TolFun',1e-8,'TolX',1e-8,'MaxFunEvals',50000);
% options = optimset('TolX',1e-20,'TolFun',1e-20,'MaxFunEvals',100000,'MaxIter',1000);
% options = optimset('TolX',1e-40,'TolFun',1e-20,'Algorithm','levenberg-marquardt','MaxFunEvals',100000);
options = optimset('TolX',1e-40,'TolFun',1e-20,'Algorithm','levenberg-marquardt');
% options = optimset('TolX',1e-20,'TolFun',1e-20);
optvar_init = [u_init x_g];
[optvar,resnorm,exitflag] ...
  = lsqnonlin(@(optvar) calc_resid(optvar,Nx,n,m,del,eps,f,tau0,fric_powerlaw,fric_coulomb,ds,s_u,s_h,case_f,c_hat),optvar_init,[],[],options); % Change in calc_resid input. Oyvind

%%% Extract optimized solution
u = optvar(1:Nx);
x_g = optvar(Nx+1)
x_h = x_g*s_h;
x_u = x_g*s_u;
dx = x_g*ds;

%%% Calculate layer thickness
h = calc_h(u,x_g,x_h,Nx,dx,n,m,eps,del,case_f);

hf = det_hf(x_g,case_f,n,m,eps,del)
['Extensional stress at GL = ',num2str((del*hf/(8*eps))^n)]
['h(Nx)-hf = ',num2str(h(Nx)-hf)]

%%% Uncomment for initial condition plots and consistency check
check_consistency(h,u,x_g,Nx,n,m,del,eps,f,tau0, ...
                              fric_powerlaw,fric_coulomb,dx,x_u,x_h,case_f, c_hat) % change in check_consistency input. Oyvind

%%% Plot velocity
figure(1); 
subplot(3,1,1)
plot(x_u*L/1000,u*U*365*24*60*60);
hold on;
hold off;
xlabel('x in km');
ylabel('u in m/yr');

%%% Plot layer thickness and topography
b = det_b(x_h,case_f,n,m,eps,del);
subplot(3,1,2)
plot(x_h*L/1000,(h-b)*H,'r');
hold on;
plot(x_h*L/1000,h-b*H,'b');
hold off;
xlabel('x in km');
ylabel('y in m')
legend('h-b','-b');

%%% Save dimensional profile and velocity for comparison
b_dim = b*H;
h_dim = h*H;
u_dim = u*U;
xh_dim = x_h*L;
xu_dim = x_u*L;
display("Tsai BC at grounding = " + num2str((del*hf/(8*eps))^n*U/L*365*24*60*60))
dlmwrite('tsai_bc_grounding.csv', [(del*hf/(8*eps))^n*U/L]); % Grounding BC in m/s
data = [b; h; u; x_h; x_u]';
fid = fopen('compare_data_output.csv', 'wt');
fprintf(fid, '%s\t %s\t  %s\t %s\t %s\n', 'b','h','u','x_h','x_u');
dlmwrite('compare_data_output.csv', data, 'delimiter', '\t', '-append', 'precision', 20);
fclose(fid);


%% FUNCTIONS
%%% Computes the terms in the momentum equation, and returns the residualXML
function r = calc_resid (optvar,Nx,n,m,del,eps,f,tau0, fric_powerlaw,fric_coulomb,ds,s_u,s_h,case_f,c_hat) % change in calc_resid input. Oyvind

%%% Extract velocity
u = optvar(1:Nx);
x_g = optvar(Nx+1);
x_h = s_h*x_g;
x_u = s_u*x_g;
dx = ds*x_g;

%%% Topography
% b = det_b(x_u,case_f,n,m,eps,del);
b = det_b(x_h,case_f,n,m,eps,del);
bx = det_bx(x_u,case_f,n,m,eps,del);

%%% Floatation thickness
hf = det_hf(x_g,case_f,n,m,eps,del);

%%% Calculate layer thickness
h = calc_h(u,x_g,x_h,Nx,dx,n,m,eps,del,case_f);

%%% To store residual
r = 0*optvar;

%%% On the first u-gridpoint we just impose the no-flux boundary condition
if (fric_powerlaw)
  r(1) = u(1);
else
  r(1) = 0;
end

%%% Enforce floatation condition
r(Nx+1) = h(Nx)-hf;

for i=2:Nx
  
  %%% Compute extensional and driving stresses
  if i~=Nx,
    ux_p = (u(i+1)-u(i))/dx;
  else
    ux_p = (del*hf/(8*eps))^n; % Implements right BC

  end
  ux_n = (u(i)-u(i-1))/dx;
  if (ux_p==0)
    hux_p = 0;
  else
    hux_p = h(i)*abs(ux_p)^(1/n-1)*ux_p;
  end
  if(ux_n == 0);
    hux_n = 0;
  else
    hux_n = h(i-1)*abs(ux_n)^(1/n-1)*ux_n;
  end
  extens = 4*eps*(hux_p-hux_n)/dx;
  driving = (h(i)+h(i-1))/2*((h(i)-h(i-1))/dx-bx(i));
  
  %%% Calculate basal stress
  if (fric_powerlaw)
    % basal = abs(u(i))^(m-1)*u(i); Previous basal stress. Oyvind
    basal = c_hat * abs(u(i))^(m-1)*u(i); % basal stress with c_hat. Oyvind
  else
    basal = tau0;
  end
  if (fric_coulomb)
%     coulomb = f*((h(i)+h(i-1))/2-b(i)/(1-del));
    coulomb = f*(h(i)-b(i)/(1-del));
    basal = coulomb
%     basal = min([basal coulomb]);
  end
  
  %%% Calculate residual
  resid = extens - basal - driving;           
  r(i)=resid;  
  
end
%%% Calculates layer thickness from velocity
function h = calc_h (u,x_g,x_h,Nx,dx,n,m,eps,del,case_f)

%%% Accumulation on h-gridpoints
a = det_a(x_h,case_f);

%%% Floatation thickness
hf = det_hf(x_g,case_f,n,m,eps,del);

%%% Velocity gradient at grounding line
ux_xg = (del*hf/(8*eps))^n;

%%% Transport on u-gridpoints
T = [0 cumsum(a(1:Nx-1))]*dx;

%%% Initialize layer thickness
h = 0*x_h;

%%% Calculate h on interior points
for i=Nx-1:-1:1
  
  %%% Centered averaging of h
%   h(i) = 2*T(i+1)/u(i+1) - h(i+1); 

  %%% Upwinding
  h(i) = T(i+1)/u(i+1); 
  
  %%% Centered averaging of T/u
%   h(i) = (T(i+1)+T(i))/(u(i+1)+u(i)); 

end

%%% Sets h equal to floatation thickness at grounding line
% h(Nx) = hf; 

%%% Upwinding
u_xg = u(Nx) + 0.5*dx*ux_xg; 
T_xg = T(Nx) + 0.5*dx*a(Nx);
h(Nx) = T_xg/u_xg;
%%% Computes the terms in the momentum equation and plots them, along with
%%% their residual
function [] = check_consistency (h,u,x_g,Nx,n,m,del,eps,f,tau0,fric_powerlaw,fric_coulomb,dx,x_u,x_h,case_f, c_hat) % change in check_consistency input. Oyvind
  
%%% Topography
% b = det_b(x_u,case_f,n,m,eps,del);
b = det_b(x_h,case_f,n,m,eps,del);
bx = det_bx(x_u,case_f,n,m,eps,del);

%%% Floatation thickness
hf = det_hf(x_g,case_f,n,m,eps,del);

%%% Storage for each term
extens = 0*u;
basal = 0*u;
driving = 0*u;
drag = 0*u;
coulomb = 0*u;

%%% Compute terms
for i=2:Nx
  
  %%% Calculate extensional and driving stresses
  if i~=Nx
    ux_p = (u(i+1)-u(i))/dx;
  else
    ux_p = (del*hf/(8*eps))^n; % Implements right BC
  end
  ux_n = (u(i)-u(i-1))/dx;
  if (ux_p==0)
    hux_p = 0;
  else
    hux_p = h(i)*abs(ux_p)^(1/n-1)*ux_p;
  end
  if(ux_n == 0);
    hux_n = 0;
  else
    hux_n = h(i-1)*abs(ux_n)^(1/n-1)*ux_n;
  end 
  extens(i) = 4*eps*(hux_p-hux_n)/dx; 
  driving(i) = (h(i)+h(i-1))/2*((h(i)-h(i-1))/dx-bx(i));   
  
  %%% Calculate basal shear stress
  if (fric_powerlaw)
    % drag(i) = abs(u(i))^(m-1)*u(i); % previous basal stress. Oyvind
    drag(i) = c_hat * abs(u(i))^(m-1)*u(i); % basal stress with c_hat. Oyvind
  else
    drag(i) = tau0;
  end
%   coulomb(i) = f*((h(i)+h(i-1))/2-b(i)/(1-del));  
  coulomb(i) = f*(h(i)-b(i)/(1-del));  
  basal(i) = drag(i);
  if (fric_coulomb)
%     basal(i) = min([basal(i) coulomb(i)]);
  end  
   
end

%%% Residual
resid = zeros(1,Nx+1);
resid(1:Nx) = extens-basal-driving;
if (fric_powerlaw)
  resid(1) = u(1);
else
  resid(1) = 0;
end
resid(Nx+1) = h(Nx)-hf;
['Sum of squared residuals = ',num2str(sum(resid.^2))]

%%% Make plots
subplot(3,1,3)
plot(x_u*100,extens,'b');
hold on;
plot(x_u*100,basal,'g');
plot(x_u*100,driving,'r');
plot(x_u*100,extens-basal-driving,'k');
% plot(x_u,u,'r--')
% plot(x_h,h,'b--')
% plot(x_u,drag,'g--');
% plot(x_u,coulomb,'k--');
hold off;
% set(gca,'Xlim',[x_g*0.95 x_g]);
% set(gca,'Ylim',[min(driving) max(u)]);
legend('extens','basal','driving','sum','Location','NorthWest');
xlabel('x in km');
ylabel('dimensionless stress')
%%% Floatation thickness
function a = det_a (x,case_f)

if case_f == 1
  a = 1*ones(size(x));
else    
  a = 1*ones(size(x));
end
%%% Floatation thickness
function hf = det_hf (x,case_f,n,m,eps,del)

hf = det_b(x,case_f,n,m,eps)/(1-del);
%%% Topographic depth
function b = det_b (x,case_f,n,m,eps,del)

% if case_f == 1
%   gamma=n/(n+2);    
%   b = (1-del)*eps^gamma*(10-5*x.^2+5*x.^4/4);
% else    
%   gamma=n/(n+m+3);    
%   b = (1-del)*eps^gamma*(10-5*x.^2+5*x.^4/4);
% end

b = 0.05 + 0.25*(x-1).^2;
% b = 0*x;
%%% Topographic slope
function bx = det_bx (x,case_f,n,m,eps,del)

% if case_f == 1
%   gamma=n/(n+2);    
%   bx = (1-del)*eps^gamma*(-10*x+5*x.^3);
% else    
%   gamma=n/(n+m+3);    
%   bx = (1-del)*eps^gamma*(-10*x+5*x.^3);
% end

bx = 0.5*(x-1);
% x = 0*x;