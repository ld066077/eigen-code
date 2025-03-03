function shootev_search2d()
    format long;
    
    % 创建plots目录
    if ~exist('plots', 'dir')
        mkdir('plots');
    end

    % 定义实部 虚部 参数扫描范围
    realomg = linspace(1.76, 1.94, 25);
    imagomg = linspace(-4.2e-2, -1.5e-2, 12);
    [Re, Im] = meshgrid(realomg, imagomg);    % 生成网格
    
    % 初始化存储矩阵
    residual_values = zeros(size(Re));
    
    % ODE求解参数
    deri = 0.0001 + 0.00001i;
    options = odeset('AbsTol', 1e-8, 'RelTol', 1e-8);

    % 遍历所有网格点
    for i = 1:size(Re, 1)
        for j = 1:size(Re, 2)
            % 构造当前复数频率
            current_omega = Re(i,j) + 1i*Im(i,j);
            
            % 计算目标函数值（使用残差的模）
            residual = shooting_objective(current_omega, deri, options);
            residual_values(i,j) = norm(residual); % 计算残差的2-范数
        end
        fprintf('已完成 %.1f%%\n', 100*i/size(Re,1)); % 进度显示
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

    % 嵌套函数保持与原始代码一致
    function residual = shooting_objective(omega, deri, options)
        [~, Er] = ode15s(@(x, Er) evfun(x, Er, omega), [0, 1], [0, deri], options);
        boundary_value = Er(end, 1);
        residual = [real(boundary_value); imag(boundary_value)]; % 保持二维残差
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
end
