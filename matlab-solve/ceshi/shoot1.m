function tt = shootev()
    format long;
    tol = 1.5e-4;
    m = 1;
    omg_log = zeros(1, m);

    % 静默配置ODE求解器选项
    options = odeset('AbsTol', 1e-8, 'RelTol', 1e-8); %精度

    % 如果不存在，则创建plots目录
    if ~exist('plots', 'dir')
        mkdir('plots');
    end

    omega_initial = 1.0; % k的初始值
    deri = 1;

    for n = 1:m  % 计算前m个特征值
        omega = omega_initial;
        domg = omega / 100;
        % domg = 0.01;
        omega = omega + domg;

        % 使用匿名函数将k传递给evfun
        evfun_handle = @(x, Er) evfun(x, Er, omega);
        [x, Er] = ode45(evfun_handle, [0, 1], [0, deri], options); %导数
        oldEr = Er(end, 1);
        dEr = oldEr;

        while abs(dEr) > tol
            omega = omega + domg;
            % 使用新的k更新函数句柄
            evfun_handle = @(x, Er) evfun(x, Er, omega);
            [x, Er] = ode45(evfun_handle, [0, 1], [0, deri], options);
            dEr = Er(end, 1);

            if dEr * oldEr < 0
                omega = omega - domg;
                domg = domg / 2;
            end
            display('------');
            disp(omega);
            display(domg);
            display(dEr);
        end
        omg_log(n) = omega;  % 存储特征值
        omega_initial = omega; % 关键：用于下一次循环迭代

        % 绘图并保存（使用存储的x和Er）
        fig = figure('Visible', 'off');
        % Er(:, 1) = Er(:, 1) / max(Er(:, 1));  % 归一化
        plot(x, Er(:, 1), 'LineWidth', 1.5);
        title(sprintf('omega=%.4f的特征函数', omega));
        xlabel('x');
        ylabel('Er(x)');
        grid on;
        saveas(fig, fullfile('plots', sprintf('eigenfunc_%d.png', n)));
        close(fig);
    end

    tt = omg_log;
    disp(tt);

   % 组合图（对每个k值求解ODE *一次*，使用存储的k值）
    composite_fig = figure('Visible', 'off');
    hold on;
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'w'];
    for n = 1:m
        % 为*存储的*k值创建函数句柄
        evfun_handle = @(x, Er) evfun(x, Er, omg_log(n));
        [x, Er] = ode45(evfun_handle, [0, 1], [0, deri], options);
        Er(:, 1) = Er(:, 1) / max(Er(:, 1));  % 归一化
        plot(x, Er(:, 1), colors(n), 'LineWidth', 1.2, ...
            'DisplayName', sprintf('omega=%.4f', omg_log(n)));
    end
    title('组合特征函数');
    xlabel('x');
    ylabel('Er(x)');
    legend('show');
    grid on;
    saveas(composite_fig, fullfile('plots', 'composite_eigenfuncs.png'));
    close(composite_fig);
    

    % 嵌套函数（现在将k作为参数）
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

    function DD = outputD(x, omg_val)
        rho0 = 0.01;
        gammai = 5/3;   % 或 1.6667
        tau = 1.0;
        q = @(x) 1.05 + 4*x.^2;
        T = @(x) 0.2 + 0.8*(1 - x.^2).^2;

        OmegaASq = @(x) T(x) ./ (2*q(x).^2);
        OmegaGSq = @(x) T(x) .* (1 + 1./(2*q(x).^2));
        NumeratorDrd = @(omg_val, x) (omg_val^2 - OmegaASq(x)) .* (omg_val^2 - OmegaGSq(x));
        % DenominatorDrd = @(omg_val, x) 2*(gammai + tau)*T(x).^2.*omg_val^2 + 1e-6;
        DD = NumeratorDrd(omg_val, x);
        display(DD);
        % Drd = @(omg_val, x) NumeratorDrd(omg_val, x) ./ DenominatorDrd(omg_val, x);

        % DD = [Er(2);
        % (4/(rho0^2)) * Drd(omg_val, x) * Er(1)];
    end
end
