a = 1.0;
rho0 = 0.01;
gammai = 5/3;   % 或 1.6667
tau = 1.0;

q = @(x) 1.05 + 4*x.^2;
T = @(x) 0.2 + 0.8*(1 - x.^2).^2;

OmegaASq = @(x) T(x) ./ (2*q(x).^2);
OmegaGSq = @(x) T(x) .* (1 + 1./(2*q(x).^2));

NumeratorDrd = @(OmegaSq, x) (OmegaSq - OmegaASq(x)) .* (OmegaSq - OmegaGSq(x));
DenominatorDrd = @(OmegaSq, x) 2*(gammai + tau)*T(x).^2.*OmegaSq + 1e-6;
Drd = @(OmegaSq, x) NumeratorDrd(OmegaSq, x) ./ DenominatorDrd(OmegaSq, x);

OmegaSq_val = 0.91;  % 替换为实际需要的参数
odefun = @(x, y) [y(2); (4/(rho0^2)) * Drd(OmegaSq_val, x) * y(1)];

y0 = [0; 1];
xspan = [0, a];
sol = ode45(odefun, xspan, y0);

% ---------- 修改部分：隐藏图形窗口并直接保存图片 ----------
hFig = figure('Visible', 'off');  % 关闭图形显示
x_plot = linspace(0, a, 100);
y_plot = deval(sol, x_plot);

plot(x_plot, y_plot(1,:));
xlabel('x');
ylabel('E0r');
title('ODE Solution');
grid on;

saveas(hFig, 'ode_solution.png');  % 保存为 PNG 文件
close(hFig);  % 关闭图形对象释放内存
% --------------------------------------------------------
display('tt');
exit;  % 确保 MATLAB 正常退出
