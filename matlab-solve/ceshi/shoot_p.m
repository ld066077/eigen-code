function tt = shootev()
    format long;
    tol = 1.5e-4;
    % 静默配置ODE求解器选项
    options = odeset('AbsTol', 1e-8, 'RelTol', 1e-8); %精度

    % 如果不存在，则创建plots目录
    if ~exist('plots', 'dir')
        mkdir('plots');
    end

    omega_initial = 0.708851405598007;
    deri = 0.3e-12;

    omega = omega_initial;
    % 使用匿名函数将k传递给evfun
    evfun_handle = @(x, Er) evfun(x, Er, omega);
    [x, Er] = ode15s(evfun_handle, [0, 1], [0, deri], options); %导数

    % 绘图并保存（使用存储的x和Er）
    fig = figure('Visible', 'off');
    % Er(:, 1) = Er(:, 1) / max(Er(:, 1));  % 归一化
    plot(x, Er(:, 1), 'LineWidth', 1.5);
    title(sprintf('omega=%.4f的特征函数', omega));
    xlabel('x');
    ylabel('Er(x)');
    grid on;
    saveas(fig, fullfile('plots', sprintf('eigenfunc.png')));
    close(fig);

    tt = omega;
    disp(tt);
    

    % 嵌套函数（现在将omg作为参数）
    function yy = evfun(x, Er, omg_val)
        rho0 = 0.01;
        gammai = 5/3;   % 或 1.6667
        tau = 1.0;
        q = @(x) 1.05 + 4*x.^2;
        T = @(x) 0.2 + 0.8*(1 - x.^2).^2;

        OmegaASq = @(x) T(x) ./ (2*q(x).^2);
        OmegaGSq = @(x) T(x) .* (1 + 1./(2*q(x).^2));
        NumeratorDrd = @(omg_val, x) (omg_val^2 - OmegaASq(x)) .* (omg_val^2 - OmegaGSq(x));
        DenominatorDrd = @(omg_val, x) 2*(gammai + tau)*T(x).^2.*omg_val^2 + 1e-6;

        Drd = @(omg_val, x) NumeratorDrd(omg_val, x) ./ DenominatorDrd(omg_val, x);

        yy = [Er(2);
        (4/(rho0^2)) * Drd(omg_val, x) * Er(1)];
    end
end
