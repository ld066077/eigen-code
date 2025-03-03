function omg_log = shootev_svd_parallel()
    format long;
    
    % 定义参数扫描范围
    realomg = linspace(1.76, 1.94, 25);    % 实部扫描范围
    imagomg = linspace(-4.2e-2, -1.5e-2, 12); % 虚部扫描范围
    [Re, Im] = meshgrid(realomg, imagomg);    % 生成网格
    
    % 将二维网格展平为一维数组以便并行处理
    omega_list = Re(:) + 1i*Im(:);
    total_points = numel(omega_list);
    sigma_min_values = zeros(total_points, 1);
    
    % 离散化参数配置
    N = 1000; % 保持与原始代码一致的空间离散度
    K = construct_matrix(1.8, N); % 初始化验证矩阵结构（任意omega值）

    % 创建plots目录
    if ~exist('plots', 'dir')
        mkdir('plots');
    end

    % 启动并行池
    if isempty(gcp('nocreate'))
        parpool('Processes', 4); % 根据实际CPU核心设置
    end

    % 创建数据队列用于进度跟踪
    progressQueue = parallel.pool.DataQueue;
    afterEach(progressQueue, @updateProgress);
    progressCount = 0;
    totalStart = tic;
    
    % 资源监控初始化
    monitorTimer = timer('ExecutionMode', 'fixedRate', 'Period', 5,...
                        'TimerFcn', @(x,y) monitorResources());
    start(monitorTimer);

    try
        % 并行计算主循环
        parfor k = 1:total_points
            current_omega = omega_list(k);
            sigma_min_values(k) = svd_objective(current_omega, N);
            
            % 进度更新
            send(progressQueue, []);
        end

        % 结果重构为二维矩阵
        sigma_min_matrix = reshape(sigma_min_values, size(Re));

        % 终止资源监控
        stop(monitorTimer);
        delete(monitorTimer);

        % ====== 三维可视化 ======
        fig = figure('Position', [100 100 1200 800]);
        
        % 曲面图
        subplot(2,2,[1 3]);
        surf(Re, Im, log10(sigma_min_matrix), 'EdgeColor', 'none');
        title('SVD最小奇异值曲面图 (对数坐标)');
        xlabel('Re(\omega)');
        ylabel('Im(\omega)');
        zlabel('log_{10}(\sigma_{min})');
        view(-30, 30);
        colormap(jet);
        colorbar;
        
        % 等高线图
        subplot(2,2,2);
        contourf(Re, Im, sigma_min_matrix, 20);
        title('奇异值等高线图');
        xlabel('Re(\omega)');
        ylabel('Im(\omega)');
        colorbar;
        
        % 热图投影
        subplot(2,2,4);
        imagesc(realomg, imagomg, sigma_min_matrix);
        set(gca, 'YDir', 'normal');
        title('低维投影热图');
        xlabel('Re(\omega)');
        ylabel('Im(\omega)');
        colorbar;
        
        % 保存图像
        saveas(fig, fullfile('plots', 'svd_min_singular_values.png'));
        close(fig);
        
        % 提取候选特征值
        threshold = 0.1 * min(sigma_min_values); % 经验阈值
        candidate_indices = find(sigma_min_values < threshold);
        omg_log = omega_list(candidate_indices);
        fprintf('找到的候选特征值:\n');
        disp(omg_log);
        
    catch ME
        % 异常处理
        stop(monitorTimer);
        delete(monitorTimer);
        rethrow(ME);
    end
    
    % 资源监控函数 (Windows专用)
    function monitorResources()
        try
            [~, cmdout] = system('wmic OS get FreePhysicalMemory,TotalVisibleMemorySize /value');
            mem_info = regexp(cmdout, '(\d+)', 'match');
            total_mem = str2double(mem_info{2})/1024; % MB
            free_mem = str2double(mem_info{1})/1024;
            
            [~, cpu_info] = system('wmic cpu get LoadPercentage /value');
            cpu_load = regexp(cpu_info, '(\d+)', 'match');
            
            fprintf('[资源监控] 内存使用: %.1f/%.1f MB | CPU负载: %s%%\n',...
                   total_mem - free_mem, total_mem, cpu_load{1});
        catch
            fprintf('资源监控仅支持Windows系统\n');
        end
    end

    % 进度更新函数
    function updateProgress(~)
        progressCount = progressCount + 1;
        elapsed = toc(totalStart);
        remaining = (elapsed/progressCount)*(total_points - progressCount);
        fprintf('进度: %.1f%%, 已用: %.1fs, 剩余: %.1fs\n',...
               progressCount/total_points*100, elapsed, remaining);
    end
end

function sigma_min = svd_objective(omega, N)
    try
        K = construct_matrix(omega, N);
        [~, S, ~] = svd(K);
        sigma_min = S(end,end); % 最小奇异值
    catch
        sigma_min = Inf; % 奇异值计算失败时返回大值
    end
end

function K = construct_matrix(omega, N)
    dx = 1/(N-1);
    x_nodes = linspace(0,1,N)';
    
    K = spalloc(N, N, 3*N); % 使用稀疏矩阵提高效率
    
    for i = 2:N-1
        [~, a3] = compute_coefficients(x_nodes(i), omega);
        
        % 二阶中心差分格式
        K(i, i-1) = -a3/(dx^2);
        K(i, i)   =  2*a3/(dx^2);
        K(i, i+1) = -a3/(dx^2);
    end
    
    % 边界条件
    K(1,1) = 1;
    K(N,N) = 1;
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