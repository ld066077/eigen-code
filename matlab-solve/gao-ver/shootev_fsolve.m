function tt = shootev_fsolve()
    format long;
    tol = 1.0e-13;
    m = 1;
    omg_log = zeros(1, m) + 1i * zeros(1, m);

    % 如果 'plots' 目录不存在，创建它
    if ~exist('plots', 'dir')
        mkdir('plots');
    end

    omega_initial = 1.8 - 0.001i;
    deri = - 0.0001 - 0.00001i;
    options = odeset('AbsTol', 1e-8, 'RelTol', 1e-8, 'OutputFcn',[]); % 'OutputFcn',[] 抑制输出

    for n = 1:m
        % 使用 fsolve 进行根查找
        objective_fun = @(omega) shooting_objective(omega, deri, options); % 嵌套函数，减少计算量，计算evfun时omega为定值
        objf = shooting_objective(omega_initial, deri, options);
        disp(objf);
        omega = fsolve(objective_fun, omega_initial, optimoptions('fsolve', 'Display', 'iter', 'FunctionTolerance', tol, 'StepTolerance', tol));

        omg_log(n) = omega;
        omega_initial = omega;  % 更新为下一个特征值 +domg

        % 使用找到的特征值*一次*求解ODE
        [x, Er] = ode15s(@(x, Er) evfun(x, Er, omega), [0, 1], [0, deri], options);

        % 绘图和保存
        fig = figure('Visible', 'off');
        plot(x, real(Er(:, 1)), 'LineWidth', 1.5, 'DisplayName', '实部');
        hold on;
        plot(x, imag(Er(:, 1)), 'LineWidth', 1.5, 'DisplayName', '虚部');
        title(sprintf('omega=%.6f + %.6fi', real(omega), imag(omega))); % 改进的格式化
        xlabel('x');
        ylabel('Er(x)');
        legend('show');
        grid on;
        saveas(fig, fullfile('plots', sprintf('eigenfunc_%d.png', n)));
        close(fig);
    end
    objf = shooting_objective(omega, deri, options);
    disp(objf);
    tt = omg_log;
    disp(tt);

   % 组合绘图（为每个存储的 omg 值*一次*求解ODE）
    composite_fig = figure('Visible', 'off');
    hold on;
    colors = lines(m); % 使用 'lines' 颜色图生成不同的颜色
    for n = 1:m
        [x, Er] = ode15s(@(x, Er) evfun(x, Er, omg_log(n)), [0, 1], [0, deri], options);
        Er(:, 1) = Er(:, 1) / max(abs(Er(:, 1)));
        plot(x, real(Er(:, 1)), 'Color', colors(n,:), 'LineWidth', 1.2, 'DisplayName', sprintf('Re(omega)=%.4f', real(omg_log(n))));
        plot(x, imag(Er(:, 1)), 'Color', colors(n,:), 'LineStyle', '--', 'LineWidth', 1.2, 'DisplayName', sprintf('Im(omega)=%.4f', imag(omg_log(n))));
    end
    title('组合特征函数'); % 一致的标题
    xlabel('x');
    ylabel('Er(x)');
    legend('show', 'Location', 'best');
    grid on;
    saveas(composite_fig, fullfile('plots', 'composite_eigenfuncs.png'));
    close(composite_fig);


    function residual = shooting_objective(omega, deri, options)
        % 嵌套函数，用于计算 fsolve 的残差
        [~, Er] = ode15s(@(x, Er) evfun(x, Er, omega), [0, 1], [0, deri], options);
        residual = Er(end, 1);  % 残差是边界处的值
        residual = real(residual)^2 + imag(residual)^2; % 将实部和虚部分开以便 fsolve 使用
    end



    function yy = evfun(x, Er, omg_val)
        % 预先计算依赖于 x 和 omg_val 的量
        rho0 = 0.01;
        tau = 1.0;
        q = 1.5*(1.0+x);
        T = 1.0*(1.005-x);

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

        yy = [Er(2); -a3/a1 * Er(1)];
    end
end
