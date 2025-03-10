function shootderi_newton()
    format long;
    
    % 创建plots目录
    if ~exist('plots', 'dir')
        mkdir('plots');
    end

    % 定义参数扫描范围
    realomg = 1.441949519673746;%1.8;    % 实部初始值
    imagomg = - 0.038735637540929;%-0.1; % 虚部初始值

    omega = realomg + 1i*imagomg;
    resitol = 1.0e-10;
    % ODE求解参数
    deri1 = - 0.000105263 - 2.10526e-5i;%0.0001 + 0.00001i;
    options = odeset('AbsTol', 1e-8, 'RelTol', 1e-8);
    y1 = shooting_objective(omega, deri1, options);
    disp(abs(y1));
    if (abs(y1) < resitol)
        deri = deri1;
    else
        deri2 = deri1*resitol/y1;
        y2 = shooting_objective(omega, deri2, options);
        if (abs(y2) < resitol)
            deri = deri2;
        else
            derinm1 = deri1;
            derin = deri2;
            yn = y2;
            ynm1 = y1;
            maxiter = 100;
            for k = 1:maxiter
                derinp1 = derin + (derin-derinm1)*(resitol-yn)/(yn-ynm1);
                ynp1 = shooting_objective(omega, derinp1, options);
                if (abs(ynp1) < resitol)
                    deri = derinp1;
                    break;
                end
                derinm1 = derin;
                derin = derinp1;
                ynm1 = yn;
                yn = ynp1;
                disp(k);
                disp(derin);
                disp(abs(yn));
                disp('-------');
            end
        end
    end

    [~, Er] = ode15s(@(x, Er) evfun(x, Er, omega), [0, 1], [0, deri], options);
    x = linspace(0, 1, length(Er));
    % 绘图并保存（使用存储的x和Er）
    % fig = figure('Visible', 'off');
    % Er(:, 1) = Er(:, 1) / max(abs(Er(:, 1)));  % 归一化
    plot(x, real(Er(:, 1)), 'LineWidth', 1.5, 'DisplayName', 'Real');
    hold on;
    plot(x, imag(Er(:, 1)), 'LineWidth', 1.5, 'DisplayName', 'Imag');
    title(sprintf('omega=%.4f + %.4fi', real(omega), imag(omega)));
    xlabel('x');
    ylabel('Er(x)');
    legend('show');
    grid on;
    % saveas(fig, fullfile('plots', sprintf('eigenfunc.png')));
    % close(fig);

end
    % 嵌套函数保持与原始代码一致
    function residual = shooting_objective(omega, deri, options)
        [~, Er] = ode15s(@(x, Er) evfun(x, Er, omega), [0, 1], [0, deri], options);
        residual = Er(end, 1); % 改为直接返回残差
    end

    function yy = evfun(x, Er, omg_val)
        % 保持原有物理模型计算不变
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