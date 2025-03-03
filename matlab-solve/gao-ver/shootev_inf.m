function tt = shootev_inf()
    format long;
    tol = 1.5e-4;
    m = 12;
    omg_log = zeros(1, m) + 1i * zeros(1, m);

    % fsolve 的选项 (可以根据需要调整)
    options = optimoptions('fsolve', 'Display', 'iter', 'FunctionTolerance', tol, ...
        'StepTolerance', tol, 'Algorithm', 'levenberg-marquardt'); % 使用 trust-region-dogleg 算法

    % 如果不存在，则创建plots目录
    if ~exist('plots', 'dir')
        mkdir('plots');
    end

    omega_initial = 1.55 - 0.001i; % 初始猜测
    deri = 0.000001 + 0.000001i;       % 初始导数

    % ODE 求解器的选项 (可以根据需要调整)
    ode_options = odeset('AbsTol', 1e-8, 'RelTol', 1e-8);

      tspan = [-0.999, 0.999]; %避免奇点

    for n = 1:m  % 计算前m个特征值

        % 使用 fsolve 求解
        objective_fun = @(omega) shooting_objective(omega, deri, tspan, ode_options);
        [omega, fval, exitflag] = fsolve(objective_fun, omega_initial, options);

        if exitflag <= 0
            warning('fsolve did not converge for n = %d.  exitflag = %d', n, exitflag);
        end


        omg_log(n) = omega;  % 存储特征值
        omega_initial = omega + 0.1 + 0.00005i;  % 更新初始猜测.  重要：防止fsolve找到同一个根。


        % 使用找到的特征值求解 ODE，以获取特征函数
        [t, Er] = ode15s(@(t, Er) evfun(t, Er, omega), tspan, [0, deri], ode_options);
        x = atanh(t);

        % 绘图并保存
        fig = figure('Visible', 'off');
        plot(x, real(Er(:, 1)), 'LineWidth', 1.5, 'DisplayName', 'Real');
        hold on;
        plot(x, imag(Er(:, 1)), 'LineWidth', 1.5, 'DisplayName', 'Imag');
        title(sprintf('omega = %.6f + %.6fi', real(omega), imag(omega)));
        xlabel('x');
        ylabel('Er(x)');
        legend('show');
        grid on;
        saveas(fig, fullfile('plots', sprintf('eigenfunc_%d.png', n)));
        close(fig);
    end

    tt = omg_log;
    disp(tt);

    % 组合图 (与之前相同)
    composite_fig = figure('Visible', 'off');
    hold on;
     colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'r', 'g', 'b', 'c', 'm','y','k']; %
    for n = 1:m
        [t, Er] = ode15s(@(t, Er) evfun(t, Er, omg_log(n)), tspan, [0, deri], ode_options);
        x = atanh(t);
        %Er(:, 1) = Er(:, 1) / max(abs(Er(:, 1)));  % Normalization, optional
        plot(x, real(Er(:, 1)), colors(n), 'LineWidth', 1.2, ...
            'DisplayName', sprintf('Re(omega)=%.4f', real(omg_log(n))));
        plot(x, imag(Er(:, 1)), [colors(n), '--'], 'LineWidth', 1.2, ...
            'DisplayName', sprintf('Im(omega)=%.4f', imag(omg_log(n))));
    end
    title('Combined Eigenfunctions');
    xlabel('x');
    ylabel('Er(x)');
    legend('show', 'Location', 'best');
    grid on;
    saveas(composite_fig, fullfile('plots', 'composite_eigenfuncs.png'));
    close(composite_fig);


    % --------------------------------------------------------
    % 嵌套函数：shooting_objective (残差函数)
    % --------------------------------------------------------
    function residual = shooting_objective(omega, deri, tspan, ode_options)
        [~, Er] = ode15s(@(t, Er) evfun(t, Er, omega), tspan, [0, deri], ode_options);
        residual = Er(end, 1);          % 残差是边界处的值
        residual = [real(residual); imag(residual)]; % 分离实部和虚部
    end


    % --------------------------------------------------------
    % 嵌套函数：evfun (ODE 定义) (与之前相同)
    % --------------------------------------------------------
    function yy = evfun(t, Er, omg_val)
        x = atanh(t); % 将 t 转换回 x

        rho0 = 0.01;
        tau = 1.0;
        q = 1.05 + 4*x.^2;
        T = 0.2 + 0.8*(1 - x.^2).^2;

        zeta_val = q*omg_val/sqrt(T);
        Zdf = -2*dawson(zeta_val) + 1i*sqrt(pi)*exp(-zeta_val.^2);
        Q0 = -Zdf;

        zeta_val = q*omg_val/(sqrt(T)*2.0);
        Zdf = -2*dawson(zeta_val) + 1i*sqrt(pi)*exp(-zeta_val.^2);
        R0 = -1.0/2.0*Zdf;
        zeta_val = q*omg_val/sqrt(T);
        rho = sqrt(T)*rho0;
        a1 = rho*rho;

        Q1 = -zeta_val+zeta_val^2*Q0;
        Q2 = -1.0/2.0*zeta_val+zeta_val^2*Q1;
        Q3 = Q2*zeta_val^2-3.0/4.0*zeta_val;
        Q4 = Q3*zeta_val^2-15.0/8.0*zeta_val;
        R1 = 1.0/4.0*(-zeta_val+zeta_val^2*R0);
        R2 = 1.0/4.0*(-zeta_val/2.0+zeta_val^2*R1);
        R3 = 1.0/4.0*(zeta_val^2*R2-3.0/4.0*zeta_val);
        R4 = 1.0/4.0*(zeta_val^2*R3-15.0/8.0*zeta_val);
        lexpr1 = 2.0*(R1+R0/2.0)-(Q1+Q0/2.0);
        lexpr2 = 2.0*(R2+R1+R0/2.0)-(Q2+Q1+Q0/2.0);
        lexpr3 = 4.0*(R3+3.0*R2/2.0+3.0*R1/2.0+3.0*R0/4.0)- ...
            (Q3+3.0*Q2/2.0+3.0*Q1/2.0+3.0*Q0/4.0);
        lexpr4 = 4.0*(R4+2.0*R3+3.0*R2+3.0*R1+3.0*R0/2.0)- ...
            (Q4+2.0*Q3+3.0*Q2+3.0*Q1+3.0*Q0/2.0);
        Gexpr = 1.0/q^2-1.0/zeta_val*(Q2+Q1+Q0/2.0)- ...
            tau*(Q1+Q0/2.0)^2/(1.0+tau*(1.0-zeta_val*Q0));
        Aexpr = 3.0/(4.0*q^2)-1.0/zeta_val*(Q2+2.0*Q1+ ...
            3.0*Q0/2.0)-tau*(Q1+Q0/2.0)*(2.0*(Q1+Q0)+zeta_val*Q0* ...
            (Q1+Q0/2.0)/(1.0+tau*(1.0-zeta_val*Q0)))/(1.0+tau*(1.0-zeta_val*Q0));
        Bexpr = lexpr4/(2.0*zeta_val^3)+tau*lexpr2*lexpr2/(2.0*zeta_val^2* ...
            (1.0+tau*(1.0-zeta_val*R0)))+tau*(Q1+Q0/2.0)*( ...
            lexpr3/zeta_val^2 + tau*lexpr1*lexpr2/(zeta_val*(1.0+tau*(1.0-zeta_val*R0))) + ...
            (1.0-tau^2)*(Q1+Q0/2.0)*lexpr1*lexpr1/(2.0*(1.0+tau*(1.0-zeta_val*R0))* ...
            (1.0+tau*(1.0-zeta_val*Q0))) + (Q1+Q0/2.0)*lexpr2/(zeta_val*(1.0+ ...
            tau*(1.0-zeta_val*Q0))) + tau*(Q1+Q0/2.0)*(Q2+Q1+Q0/2.0)/ ...
            (2.0*zeta_val*(1.0+tau*(1.0-zeta_val*Q0))) )/(1.0+tau*(1.0-zeta_val*Q0));

        a3 = 2.0*Gexpr/(Aexpr+q^2*Bexpr);
        % 应用变换后的导数
        dxdt = 1 - t^2;
        yy = [Er(2); ...
            -a3/a1 * Er(1) / dxdt^2 + 2*t*Er(2)/dxdt];  % 修正的方程
    end

end
