function tt = shootev()
    format long;
    tol = 1.5e-4;
    m = 12;
    omg_log = zeros(1, m) + 1i * zeros(1, m);

    % 静默配置ODE求解器选项
    options = odeset('AbsTol', 1e-8, 'RelTol', 1e-8); %精度

    % 如果不存在，则创建plots目录
    if ~exist('plots', 'dir')
        mkdir('plots');
    end

    omega_initial = 1.35 + 0.00001i; % k的初始值
    deri = 0.0001 + 0.00001i;

    for n = 1:m  % 计算前m个特征值  %加上边界值差值/边界值 作为误差限 large then max iter then iter+1
        omega = omega_initial;
        domg = omega / 30;
        % domg = 0.01;
        omega = omega + domg;

        % 使用匿名函数将omega传递给evfun
        evfun_handle = @(x, Er) evfun(x, Er, omega);
        [x, Er] = ode15s(evfun_handle, [0, 1], [0, deri], options); %导数
        oldEr = Er(end, 1);
        dEr = oldEr;

        while abs(dEr) > tol
            omega = omega + domg;
            % 使用新的omega更新函数句柄
            evfun_handle = @(x, Er) evfun(x, Er, omega);
            [x, Er] = ode15s(evfun_handle, [0, 1], [0, deri], options);
            dEr = Er(end, 1);

            if real(dEr) * real(oldEr) < 0 % 分别检查实部虚部
                omega = omega - real(domg);
                domg = real(domg) / 2 + imag(domg) * 1i;
            end
            if imag(dEr) * imag(oldEr) < 0
                omega = omega - imag(domg);
                domg = imag(domg) / 2 + real(domg);
            end
            fprintf('------\n');
            display(omega);
            display(domg);
            display(dEr);
        end
        omg_log(n) = omega;  % 存储特征值
        omega_initial = omega; % 关键：用于下一次循环迭代

        % 绘图并保存（使用存储的x和Er）
        fig = figure('Visible', 'off');
        % Er(:, 1) = Er(:, 1) / max(abs(Er(:, 1)));  % 归一化
        plot(x, real(Er(:, 1)), 'LineWidth', 1.5, 'DisplayName', 'Real');
        hold on;
        plot(x, imag(Er(:, 1)), 'LineWidth', 1.5, 'DisplayName', 'Imag');
        title(sprintf('omega=%.4f + %.4fi', real(omega), imag(omega)));
        xlabel('x');
        ylabel('Er(x)');
        legend('show');
        grid on;
        saveas(fig, fullfile('plots', sprintf('eigenfunc_%d.png', n)));
        close(fig);
    end

    tt = omg_log;
    disp(tt);

   % 组合图（对每个omg值求解ODE *一次*，使用存储的omg值）
    composite_fig = figure('Visible', 'off');
    hold on;
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'r', 'g', 'b', 'c', 'm','y','k'];
    for n = 1:m
        % 为*存储的*omg值创建函数句柄
        evfun_handle = @(x, Er) evfun(x, Er, omg_log(n));
        [x, Er] = ode15s(evfun_handle, [0, 1], [0, deri], options);
        Er(:, 1) = Er(:, 1) / max(abs(Er(:, 1)));  % 归一化
        plot(x, real(Er(:, 1)), colors(n), 'LineWidth', 1.2, ...
            'DisplayName', sprintf('Re(omega)=%.4f', real(omg_log(n))));
        plot(x, imag(Er(:, 1)), [colors(n), '--'], 'LineWidth', 1.2, ...  % Dashed line
            'DisplayName', sprintf('Im(omega)=%.4f', imag(omg_log(n))));
    end
    title('组合特征函数');
    xlabel('x');
    ylabel('Er(x)');
    legend('show', 'Location', 'best');
    grid on;
    saveas(composite_fig, fullfile('plots', 'composite_eigenfuncs.png'));
    close(composite_fig);
    

    % 嵌套函数（现在将omg作为参数）
    function yy = evfun(x, Er, omg_val)
        Zdf = @(zeta_val) -2*dawson(zeta_val) + 1i*sqrt(pi)*exp(-zeta_val.^2);
        rho0 = 0.01;
        tau = 1.0;
        q = @(x) 1.05 + 4*x.^2;
        T = @(x) 0.2 + 0.8*(1 - x.^2).^2;

        rho = sqrt(T(x))*rho0;
        a1 = rho*rho;
        zeta_val = q(x)*omg_val/sqrt(T(x));
        Q0 = -Zdf(zeta_val);
        R0 = -1.0/2.0*Zdf(zeta_val/2.0);
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
        Gexpr = 1.0/q(x)^2-1.0/zeta_val*(Q2+Q1+Q0/2.0)- ...
            tau*(Q1+Q0/2.0)^2/(1.0+tau*(1.0-zeta_val*Q0));
        Aexpr = 3.0/(4.0*q(x)^2)-1.0/zeta_val*(Q2+2.0*Q1+ ...
            3.0*Q0/2.0)-tau*(Q1+Q0/2.0)*(2.0*(Q1+Q0)+zeta_val*Q0* ...
            (Q1+Q0/2.0)/(1.0+tau*(1.0-zeta_val*Q0)))/(1.0+tau*(1.0-zeta_val*Q0));
        Bexpr = lexpr4/(2.0*zeta_val^3)+tau*lexpr2*lexpr2/(2.0*zeta_val^2* ...
            (1.0+tau*(1.0-zeta_val*R0)))+tau*(Q1+Q0/2.0)*( ...
            lexpr3/zeta_val^2 + tau*lexpr1*lexpr2/(zeta_val*(1.0+tau*(1.0-zeta_val*R0))) + ...
            (1.0-tau^2)*(Q1+Q0/2.0)*lexpr1*lexpr1/(2.0*(1.0+tau*(1.0-zeta_val*R0))* ...
            (1.0+tau*(1.0-zeta_val*Q0))) + (Q1+Q0/2.0)*lexpr2/(zeta_val*(1.0+ ...
            tau*(1.0-zeta_val*Q0))) + tau*(Q1+Q0/2.0)*(Q2+Q1+Q0/2.0)/ ...
            (2.0*zeta_val*(1.0+tau*(1.0-zeta_val*Q0))) )/(1.0+tau*(1.0-zeta_val*Q0));

        a3 = 2.0*Gexpr/(Aexpr+q(x)^2*Bexpr);

        yy = [Er(2); ...
        -a3/a1 * Er(1)];
    end
end
