function shootev_newton()
    format long;
    
    % 创建plots目录
    if ~exist('plots', 'dir')
        mkdir('plots');
    end

    % 定义参数扫描范围
    realomg = 1.8;    % 实部初始值
    imagomg = -0.1; % 虚部初始值
    domg = 0.1-0.001i;

    omgint = realomg + 1i*imagomg;

    % ODE求解参数
    deri = 0.0001 + 0.00001i;
    options = odeset('AbsTol', 1e-8, 'RelTol', 1e-8);

    omgnext = omgint;
    maxiter = 100;
    for k = 1:maxiter
        currentomg = omgnext;
        % 将参数显式传递进目标函数
        currentresidual = shooting_objective(currentomg, deri, options);
        tmpomg = currentomg + domg;
        tmpresidual = shooting_objective(tmpomg, deri, options);
        deriomg = -domg*currentresidual/(tmpresidual - currentresidual);
        comomg = omgnext;
        omgnext = omgnext + deriomg;
        
        if (abs(comomg-omgnext)<1.0e-5)
            break;
        end
    end



        % ========== 三维可视化 ==========
        fig = figure('Position', [100 100 1200 800]);
        
        % 曲面图
        subplot(2,2,[1 3]);
        surf(Re, Im, residual_values, 'EdgeColor', 'none');
        title('目标函数值曲面图');
        xlabel('Re(\omega)');
        ylabel('Im(\omega)');
        zlabel('||Residual||');
        view(-30, 30);
        colormap jet;
        colorbar;
        
        % 等高线图
        subplot(2,2,2);
        contourf(Re, Im, log10(residual_values), 20);
        title('对数坐标等高线图');
        xlabel('Re(\omega)');
        ylabel('Im(\omega)');
        colorbar;
        
        % 二维投影图
        subplot(2,2,4);
        imagesc(realomg, imagomg, residual_values);
        set(gca,'YDir','normal');
        title('二维投影热图');
        xlabel('Re(\omega)');
        ylabel('Im(\omega)');
        colorbar;
        
        % 保存图像
        saveas(fig, fullfile('plots', 'residual_landscape.png'));
        close(fig);


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