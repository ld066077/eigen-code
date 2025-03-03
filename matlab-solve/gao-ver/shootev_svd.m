function tt = shootev_svd()
    format long;
    tol = 1.5e-4;
    m = 1;
    N = 100; % 离散化点数
    omg_log = zeros(1, m) + 1i * zeros(1, m);
    omega_initial = 1.8 - 0.001i;
    % 创建plots目录（如果不存在）
    if ~exist('plots', 'dir')
        mkdir('plots');
    end
    % 初始化参数
    deri = 0.0001 + 0.00001i;
    options = odeset('AbsTol', 1e-8, 'RelTol', 1e-8, 'OutputFcn',[]); % 与原始代码相同的ODE选项

    % 预存储特征函数数据用于组合绘图
    all_Er = cell(1, m);
    all_x = cell(1, m);

    for n = 1:m
        % 使用 fminsearch 优化最小奇异值
        options_optim = optimset('Display', 'iter', 'TolX', tol, 'TolFun', tol);
        omega = fminsearch(@(omega) svd_objective(omega, N), omega_initial, options_optim);
        
        omg_log(n) = omega;
        omega_initial = omega + 0.1; % 更新初始猜测
        
        % ========== 添加绘图部分 ==========
        % 使用找到的特征值求解ODE
        [x, Er] = ode15s(@(x, Er) evfun(x, Er, omega), [0, 1], [0, deri], options);
        
        % 存储数据用于组合绘图
        all_Er{n} = Er(:,1);
        all_x{n} = x;
        
        % 单个特征函数绘图
        fig = figure('Visible', 'off');
        plot(x, real(Er(:,1)), 'LineWidth', 1.5, 'DisplayName', '实部');
        hold on;
        plot(x, imag(Er(:,1)), 'LineWidth', 1.5, 'DisplayName', '虚部');
        title(sprintf('特征函数 (omega=%.6f + %.6fi)', real(omega), imag(omega)));
        xlabel('x');
        ylabel('Er(x)');
        legend('show');
        grid on;
        saveas(fig, fullfile('plots', sprintf('svd_eigenfunc_%d.png', n)));
        close(fig);
    end
    % ========== 组合绘图 ==========
    composite_fig = figure('Visible', 'off');
    hold on;
    colors = lines(m);
        
    for n = 1:m
        % 获取归一化后的特征函数
        Er_normalized = all_Er{n} / max(abs(all_Er{n}));
        
        % 绘制实部和虚部
        plot(all_x{n}, real(Er_normalized), 'Color', colors(n,:), ...
            'LineWidth', 1.2, 'DisplayName', sprintf('Re(ω)=%.4f', real(omg_log(n))));
        plot(all_x{n}, imag(Er_normalized), 'Color', colors(n,:), ...
            'LineStyle', '--', 'LineWidth', 1.2, 'DisplayName', sprintf('Im(ω)=%.4f', imag(omg_log(n))));
    end
        
    title('组合特征函数 (SVD方法)');
    xlabel('x');
    ylabel('归一化Er(x)');
    legend('show', 'Location', 'best');
    grid on;
    saveas(composite_fig, fullfile('plots', 'svd_composite_eigenfuncs.png'));
    close(composite_fig);
    tt = omg_log;
    disp('找到的特征值:');
    disp(tt);
end

function residual = svd_objective(omega, N)
    K = construct_matrix(omega, N);
    [~, S, ~] = svd(K);
    sigma_min = min(diag(S));
    residual = sigma_min;
end

function K = construct_matrix(omega, N)
    dx = 1/(N-1);
    x = linspace(0, 1, N)';
    K = zeros(N, N);
    
    for i = 2:N-1
        [a1, a3] = compute_coefficients(x(i), omega);
        K(i, i-1) = -a3/(a1*dx^2);
        K(i, i)   = 2*a3/(a1*dx^2);
        K(i, i+1) = -a3/(a1*dx^2);
    end
    
    K(1, 1) = 1;
    K(N, N) = 1;
end

function [a1, a3] = compute_coefficients(x, omg_val)
    % 从原函数 evfun 中提取系数计算逻辑
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
