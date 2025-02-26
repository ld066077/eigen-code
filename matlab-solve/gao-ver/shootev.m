function tt = shootev()
    format long;
    tol = 1.5e-4;
    m = 12;
    omg_log = zeros(1, m);

    % 静默配置ODE求解器选项
    options = odeset('AbsTol', 1e-8, 'RelTol', 1e-8); %精度

    % 如果不存在，则创建plots目录
    if ~exist('plots', 'dir')
        mkdir('plots');
    end

    omega_initial = 0.35; % k的初始值
    deri = 0.0001;

    for n = 1:m  % 计算前m个特征值  %加上边界值差值/边界值 作为误差限 large then max iter then iter+1
        omega = omega_initial;
        domg = omega / 30;
        % domg = 0.01;
        omega = omega + domg;

        % 使用匿名函数将k传递给evfun
        evfun_handle = @(x, Er) evfun(x, Er, omega);
        [x, Er] = ode15s(evfun_handle, [0, 1], [0, deri], options); %导数
        oldEr = Er(end, 1);
        dEr = oldEr;

        while abs(dEr) > tol
            omega = omega + domg;
            % 使用新的k更新函数句柄
            evfun_handle = @(x, Er) evfun(x, Er, omega);
            [x, Er] = ode15s(evfun_handle, [0, 1], [0, deri], options);
            dEr = Er(end, 1);

            if dEr * oldEr < 0
                omega = omega - domg;
                domg = domg / 2;
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
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'w'];
    for n = 1:m
        % 为*存储的*k值创建函数句柄
        evfun_handle = @(x, Er) evfun(x, Er, omg_log(n));
        [x, Er] = ode15s(evfun_handle, [0, 1], [0, deri], options);
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
        Zdf = @(zeta_val) -2*dawson(zeta_val) + 1i*sqrt(pi)*exp(-zeta_val^2);
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

        yy = [Er(2);
        -a3/a1 * Er(1)];
    end
end
