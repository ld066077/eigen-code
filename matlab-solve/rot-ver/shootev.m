function tt = shootev()
    format long;
    tol = 1.5e-4;
    m = 1;
    omg_log = zeros(1, m) + 1i * zeros(1, m);

    % 如果 'plots' 目录不存在，创建它
    if ~exist('plots', 'dir')
        mkdir('plots');
    end

    omega_initial = 1.8 + 0.001i;
    deri = 0.0001 + 0.00001i;
    options = odeset('AbsTol', 1e-8, 'RelTol', 1e-8, 'OutputFcn',[]); % 'OutputFcn',[] 抑制输出

    for n = 1:m
        % 使用 fsolve 进行根查找
        objective_fun = @(omega) shooting_objective(omega, deri, options); % 嵌套函数，减少计算量，计算evfun时omega为定值
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
        residual = [real(residual); imag(residual)]; % 将实部和虚部分开以便 fsolve 使用
    end



    function yy = evfun(x, Er, omg_val)
        % 预先计算依赖于 x 和 omg_val 的量
        rho0 = 0.01;
        tau = 1.0;
        q = 1.05 + 4*x.^2;
        T = 0.2 + 0.8*(1 - x.^2).^2;

        zeta_val1 = q*omg_val/sqrt(T);
        Zdf1 = -2*dawson(zeta_val1) + 1i*sqrt(pi)*exp(-zeta_val1.^2);
        zeta_val2 = q*omg_val/(2.0*sqrt(T));
        Zdf2 = -2*dawson(zeta_val2) + 1i*sqrt(pi)*exp(-zeta_val2.^2);

        C_0_plus = zeta_val1*Zdf1+1.0;
        C_0_minus = zeta_val1*Zdf1+1.0;
        C_1_plus = -Zdf1*(M_val^2-2.0*M_val*tau*zeta_val1- ...
            2.0*M_val*zeta_val1+tau*zeta_val1^2+tau+zeta_val1^2+ ...
            1.0)/(1.0+tau)+2.0*M_val-zeta_val1;
        C_1_minus = Zdf1*(M_val^2+2.0*M_val*tau*zeta_val1+ ...
            2.0*M_val*zeta_val1+tau*zeta_val1^2+tau+zeta_val1^2+ ...
            1.0)/(1.0+tau)+2.0*M_val+zeta_val1;
        C_2_plus = M_val^4*Zdf1/(zeta_val1*(tau+1.0)^2)- ...
            2.0*M_val^2*(2.0*M_val*Zdf1*zeta_val1- ...
            Zdf1*zeta_val1^2-Zdf1-zeta_val1)/(zeta_val1*(tau+1.0))+ ...
            4.0*Zdf1*M_val^2*zeta_val1+4.0*M_val^2-4.0*Zdf1*M_val*zeta_val1^2- ...
            4.0*Zdf1*M_val-4.0*M_val*zeta_val1+Zdf1*zeta_val1^3+ ...
            2.0*Zdf1*zeta_val1+3.0*Zdf1/(2.0*zeta_val1)+zeta_val1^2+5.0/2.0;
        C_2_minus = M_val^4*Zdf1/(zeta_val1*(tau+1.0)^2) + ...
            2.0*M_val^2*(2.0*M_val*Zdf1*zeta_val1+ ...
            Zdf1*zeta_val1^2+Zdf1+zeta_val1)/(zeta_val1*(tau+1.0))+ ...
            4.0*M_val^2*Zdf1*zeta_val1+4.0*M_val^2+4.0*M_val*Zdf1*zeta_val1^2+ ...
            4.0*M_val*Zdf1+4.0*M_val*zeta_val1+Zdf1*zeta_val1^3+ ...
            2.0*Zdf1*zeta_val1+3.0*Zdf1/(2.0*zeta_val1)+zeta_val1^2+5.0/2.0;

        


        A_1_plus = D_1_plus/(tau*D_0_plus+1.0);
        A_1_minus = D_1_minus/(tau*D_0_minus+1.0);
        A_2_plus = (2.0*tau*A_1_plus*(D_1_plus-L_1_plus)- ...
            2.0*D_2_plus+L_2_plus)/(tau*L_0_plus+1.0);
        A_2_minus = (2.0*tau*A_1_minus*(D_1_minus-L_1_minus)- ...
            2.0*D_2_minus+L_2_minus)/(tau*L_0_minus+1.0);
        A_3_plus = 4.0*(-tau*A_1_plus*(C_0_plus-1.0)+ ...
            C_1_plus)/(tau*D_0_plus+1.0);
        A_3_minus = 4.0*(-tau*A_1_minus*(C_0_minus-1.0)+ ...
            C_1_minus)/(tau*D_0_minus+1.0);
        A_4_plus = -(4.0*tau*A_1_plus*D_2_plus- 2.0*tau*A_1_plus*L_2_plus+ ...
            tau*A_1_minus*D_2_plus+ tau*A_1_minus*D_2_minus+ ...
            A_2_plus*D_1_plus*tau- A_2_plus*L_1_plus*tau- ...
            3.0*D_3_plus-D_3_minus+ L_3_plus)/(tau*D_0_plus+1.0);
        A_4_minus = -(tau*(A_1_plus*D_2_plus+ A_1_plus*D_2_minus+ ...
            4.0*A_1_minus*D_2_minus- 2.0*A_1_minus*L_2_minus+ ...
            A_2_minus*D_1_minus- A_2_minus*L_1_minus)- ...
            D_3_plus-3.0*D_3_minus+ L_3_minus)/(tau*D_0_minus+1.0);

        rho = sqrt(T)*rho0;
        a1 = -rho*rho;

        R1 = tau * (3.0*A_1_plus*D_3_plus + ...
            A_1_plus*D_3_minus - A_1_plus* ...
            L_3_plus + A_2_plus*D_2_plus - ...
            A_2_plus*L_2_plus/2.0 + A_4_plus* ...
            D_1_plus + A_1_minus*D_3_plus + ...
            3.0*A_1_minus*D_3_minus - A_1_minus* ...
            L_3_minus + A_2_minus*D_2_minus - ...
            A_2_minus*L_2_minus/2.0 + A_4_minus* ...
            D_1_minus) - 2.0*(D_4_plus + D_4_minus) + ...
            1.0/2.0 * (L_4_plus + L_4_minus);

        R2 = tau*(A_1_plus*C_1_plus+A_1_minus*C_1_minus)- ...
            C_2_plus-C_2_minus+tau*(D_1_plus*A_3_plus+ ...
            D_1_minus*A_3_minus)/4.0;
        
        R3 = -tau*(A_1_plus*D_1_plus+A_1_minus*D_1_minus)+ ...
            D_2_plus+D_2_minus;

        a3 = (8.0*q*q*R3 + 16.0)/(q^4*R1+ ...
            4.0*q*q*R2-6.0);

        yy = [Er(2); -a3/a1 * Er(1)];
    end
end
