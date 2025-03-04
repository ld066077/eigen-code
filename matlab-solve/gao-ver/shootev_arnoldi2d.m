function shootev_arnoldi_parallel()
    % 使用 Arnoldi 方法（eigs）在二维扫描平面上求解 GAM 连续谱特征值（并行版本）
    % 目标：提取靠近 sigma = 1.94 - 0.015i 的特征值
    %
    % 注意：本代码依赖于 construct_matrix() 和 compute_coefficients() 函数，
    %       以及外部的 dawson() 函数（请确保已在 MATLAB 路径中）。
    
    format long;
    % --- 参数设置 ---
    N = 1000;   % 离散化节点数（矩阵尺寸）
    
    % 设置扫描的二维网格：实部和虚部的取值范围
    nRe = 50;  % 实部网格点数
    nIm = 50;  % 虚部网格点数
    Re_vals = linspace(1.74, 1.98, nRe);    % 可根据需要调整
    Im_vals = linspace(-0.03, 0, nIm);        % 虚部范围（负值代表衰减）
    [Re_grid, Im_grid] = meshgrid(Re_vals, Im_vals);
    
    % 目标移位 sigma（用于 eigs 求解）
    sigma = 1.94 - 0.015i;
    
    % 将二维网格展平为一维数组，以便并行处理
    omega_list = Re_grid(:) + 1i * Im_grid(:);
    total_points = numel(omega_list);
    resultEig_flat = complex(zeros(total_points, 1));
    
    % 设置 eigs 参数
    opts = struct('Tolerance', 1e-8, 'MaxIterations', 1000);
    % 注意：由于各点独立计算，这里不再传递上一次的初始向量
    
    % --- 初始化运行监控 ---
    progressQueue = parallel.pool.DataQueue;
    afterEach(progressQueue, @updateProgress);
    progressCount = 0;
    totalStart = tic;
    
    fprintf('开始二维并行扫描，共 %d 个点...\n', total_points);
    
    % --- 并行计算二维扫描 ---
    parfor k = 1:total_points
        omega = omega_list(k);
        % 构造矩阵 K(omega)
        K = construct_matrix(omega, N);
        
        % 使用 eigs 求解最接近 sigma 的特征值
        try
            [V, D, flag] = eigs(K, 1, sigma, opts);
            if flag ~= 0
                fprintf('警告：在 omega = %g + %gi 时 eigs 未完全收敛 (flag = %d)\n', ...
                    real(omega), imag(omega), flag);
            end
            computedEig = D(1,1);
        catch ME
            warning('在 omega = %g + %gi 处求解出现错误: %s', real(omega), imag(omega), ME.message);
            computedEig = NaN;
        end
        resultEig_flat(k) = computedEig;
        
        % 进度更新（利用 DataQueue 传递空消息）
        send(progressQueue, []);
    end
    
    % 重构为二维矩阵
    resultEig = reshape(resultEig_flat, size(Re_grid));
    totalElapsed = toc(totalStart);
    fprintf('二维并行扫描完成，总耗时 %.1fs\n', totalElapsed);
    
    % --- 可视化结果 ---
    % 1. 绘制 |computedEig - sigma| 的热图（绝对差值）
    figure;
    imagesc(Re_vals, Im_vals, abs(resultEig - sigma));
    set(gca, 'YDir','normal');
    xlabel('Re(\omega)');
    ylabel('Im(\omega)');
    title('|Eigenvalue - sigma| across \omega plane');
    colorbar;
    
    % 2. 三维曲面图显示 |computedEig - sigma|
    figure;
    surf(Re_grid, Im_grid, abs(resultEig - sigma));
    shading interp;
    xlabel('Re(\omega)');
    ylabel('Im(\omega)');
    zlabel('|Eigenvalue - sigma|');
    title('Surface plot of |Eigenvalue - sigma|');
    colorbar;
    
    % 3. 分别显示计算特征值的实部与虚部
    figure;
    subplot(1,2,1);
    imagesc(Re_vals, Im_vals, real(resultEig));
    set(gca, 'YDir','normal');
    xlabel('Re(\omega)'); ylabel('Im(\omega)');
    title('Real part of computed eigenvalue');
    colorbar;
    subplot(1,2,2);
    imagesc(Re_vals, Im_vals, imag(resultEig));
    set(gca, 'YDir','normal');
    xlabel('Re(\omega)'); ylabel('Im(\omega)');
    title('Imaginary part of computed eigenvalue');
    colorbar;
    
    % --- 保存结果（可选） ---
    % save('arnoldi_scan_results_parallel.mat', 'Re_grid', 'Im_grid', 'resultEig');
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % 内部嵌套函数：进度更新函数
    function updateProgress(~)
        progressCount = progressCount + 1;
        elapsed = toc(totalStart);
        remaining = elapsed/progressCount * (total_points - progressCount);
        fprintf('进度: %.1f%%, 已用时间: %.1fs, 预计剩余: %.1fs\n', ...
            progressCount/total_points*100, elapsed, remaining);
    end
end

%======================================================================
function K = construct_matrix(omega, N)
    % 构造离散化后的矩阵 K(omega)
    % 使用二阶中心差分法离散区间 [0,1] 上的二阶微分算子，
    % 并利用 compute_coefficients 计算系数 a3（依赖于 omega）
    dx = 1/(N-1);
    x_nodes = linspace(0, 1, N)';
    
    % 预分配一个稀疏矩阵
    K = spalloc(N, N, 3*N);
    
    for i = 2:N-1
        [~, a3] = compute_coefficients(x_nodes(i), omega);
        % 二阶中心差分格式
        K(i, i-1) = -a3/(dx^2);
        K(i, i)   =  2*a3/(dx^2);
        K(i, i+1) = -a3/(dx^2);
    end
    
    % 边界条件（Dirichlet）
    K(1,1) = 1;
    K(N,N) = 1;
end

%======================================================================
function [a1, a3] = compute_coefficients(x, omg_val)
    % 从原方程中提取系数计算逻辑
    % 此函数利用物理参数计算 a1 与 a3，后者用于构造离散算子
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
    
    Q1 = -zeta_val + zeta_val^2*Q0;
    Q2 = -1.0/2.0*zeta_val + zeta_val^2*Q1;
    Q3 = Q2*zeta_val^2 - 3.0/4.0*zeta_val;
    Q4 = Q3*zeta_val^2 - 15.0/8.0*zeta_val;
    R1 = 1.0/4.0*(-zeta_val + zeta_val^2*R0);
    R2 = 1.0/4.0*(-zeta_val/2.0 + zeta_val^2*R1);
    R3 = 1.0/4.0*(zeta_val^2*R2 - 3.0/4.0*zeta_val);
    R4 = 1.0/4.0*(zeta_val^2*R3 - 15.0/8.0*zeta_val);
    
    lexpr1 = 2.0*(R1 + R0/2.0) - (Q1 + Q0/2.0);
    lexpr2 = 2.0*(R2 + R1 + R0/2.0) - (Q2 + Q1 + Q0/2.0);
    lexpr3 = 4.0*(R3 + 3.0*R2/2.0 + 3.0*R1/2.0 + 3.0*R0/4.0) - ...
             (Q3 + 3.0*Q2/2.0 + 3.0*Q1/2.0 + 3.0*Q0/4.0);
    lexpr4 = 4.0*(R4 + 2.0*R3 + 3.0*R2 + 3.0*R1 + 3.0*R0/2.0) - ...
             (Q4 + 2.0*Q3 + 3.0*Q2 + 3.0*Q1 + 3.0*Q0/2.0);
    
    Gexpr = 1.0/q^2 - 1.0/zeta_val*(Q2+Q1+Q0/2.0) - ...
            tau*(Q1+Q0/2.0)^2/(1.0+tau*(1.0-zeta_val*Q0));
    Aexpr = 3.0/(4.0*q^2) - 1.0/zeta_val*(Q2+2.0*Q1+3.0*Q0/2.0) - ...
            tau*(Q1+Q0/2.0)*(2.0*(Q1+Q0)+zeta_val*Q0*(Q1+Q0/2.0)/(1.0+tau*(1.0-zeta_val*Q0)))/(1.0+tau*(1.0-zeta_val*Q0));
    Bexpr = lexpr4/(2.0*zeta_val^3) + tau*lexpr2^2/(2.0*zeta_val^2*(1.0+tau*(1.0-zeta_val*R0))) + ...
            tau*(Q1+Q0/2.0)*( lexpr3/zeta_val^2 + tau*lexpr1*lexpr2/(zeta_val*(1.0+tau*(1.0-zeta_val*R0))) + ...
            (1.0-tau^2)*lexpr1^2*(Q1+Q0/2.0)/(2.0*(1.0+tau*(1.0-zeta_val*R0))*(1.0+tau*(1.0-zeta_val*Q0))) + ...
            lexpr2/(zeta_val*(1.0+tau*(1.0-zeta_val*Q0)))*(Q1+Q0/2.0) + ...
            tau*(Q1+Q0/2.0)*(Q2+Q1+Q0/2.0)/(2.0*zeta_val*(1.0+tau*(1.0-zeta_val*Q0))) )/(1.0+tau*(1.0-zeta_val*Q0));
    
    a3 = 2.0*Gexpr/(Aexpr + q^2*Bexpr);
end
