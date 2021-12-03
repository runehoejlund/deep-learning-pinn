close all;

X = linspace(-1,1,301);
T = linspace(0,1,101);

%%
x = nan(301,length(T));
u = nan(301,length(T));

f = waitbar(0, 'Starting');
for i = 1:length(T)
    % WARNING: Currently I rerun the ode-solver (Runge-Kutta ode45), which
    % is stupid and inefficient, but it had to go fast. A simple update
    % will fix the problem.
    [x(:,i), u(:,i)] = solveBurgers(T(i), 60, 0.0005);
    waitbar(i/length(T), f, sprintf('Solving PDE: %d %%', floor(i/length(T)*100)));
end
close(f)

%%
close all
imagesc(T,X,u)
colorbar
xlabel('Time $t$','interpreter','latex')
ylabel('Distance $x$','interpreter','latex')
title('Burgers Equation for $-1 \le x \le 1$ and $0 \le t \le 1$','interpreter','latex')

%%
close all
plot(x, u)
legend(string(T))

%% 
save('burgers.mat','X', 'T','u')