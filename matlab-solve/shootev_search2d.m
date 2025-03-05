function shootev_search2d()
    format long;
    
    % 创建plots目录
    if ~exist('plots', 'dir')
        mkdir('plots');
    end

    % 定义参数扫描范围
    omg = linspace(0.42, 0.46, 1000);

    % 将二维网格展平为一维数组以便并行处理
    omega_list = omg;
    total_points = numel(omega_list);
    residual_flat = zeros(total_points, 1);
    
    % ODE求解参数
    deri = 0.000001;
    options = odeset('AbsTol', 1e-8, 'RelTol', 1e-8); %精度

    % 启动并行池
    if isempty(gcp('nocreate'))
        parpool; % 根据系统核心数自动分配
    end

    % 创建数据队列用于进度跟踪
    progressQueue = parallel.pool.DataQueue;
    afterEach(progressQueue, @updateProgress);
    progressCount = 0;
    totalStart = tic;

    % 资源监测初始化
    monitorTimer = timer('ExecutionMode', 'fixedRate', 'Period', 5, ...
        'TimerFcn', @(x,y) monitorResources());
    start(monitorTimer);
    try
        % 并行计算主循环
        parfor k = 1:total_points
            current_omega = omega_list(k);
            % 将参数显式传递进目标函数
            residual = shooting_objective(current_omega, deri, options);
            residual_flat(k) = norm(residual);
            
            % 进度更新
            send(progressQueue, []); 
        end

        residual_values = residual_flat;

        % 终止资源监控
        stop(monitorTimer);
        delete(monitorTimer);

        % ========== 找到极小值点并加上标记 ==========

        % 查找局部极小值点
        [pks, locs] = findpeaks(-residual_values);  % -residual_values表示局部最小值
        min_values = omega_list(locs);  % 极小值点的 omega

        % ========== 三维可视化 ==========

        fig = figure('Position', [100 100 1200 800]);
        plot(omega_list, residual_values, 'b-', 'LineWidth', 2);
        xlabel('Omega');
        ylabel('Residual Norm');
        title('Residual Landscape');
        grid on;

        % 在图上显示数据点
        hold on;
        scatter(omega_list, residual_values, 40, 'r', 'filled');
        
        % 标记极小值点并显示其值
        for i = 1:length(min_values)
            text(min_values(i), residual_values(locs(i)), ...
                 sprintf('%.4f', residual_values(locs(i))), ...
                 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'right', ...
                 'Color', 'black', 'FontSize', 12, 'FontWeight', 'bold');
        end

        hold off;
        
        % 保存图像
        saveas(fig, fullfile('plots', 'residual_landscape_with_minima.png'));
        close(fig);
    catch ME
        % 异常处理
        stop(monitorTimer);
        delete(monitorTimer);
        rethrow(ME);
    end

    % 资源监控函数
    function monitorResources()
        try
            % Windows系统
            [~, cmdout] = system('wmic OS get FreePhysicalMemory,TotalVisibleMemorySize /value');
            mem_info = regexp(cmdout, '(\d+)', 'match');
            total_mem = str2double(mem_info{2})/1024; % MB
            free_mem = str2double(mem_info{1})/1024;
            
            [~, cpu_info] = system('wmic cpu get LoadPercentage /value');
            cpu_load = regexp(cpu_info, '(\d+)', 'match');
            
            fprintf('[资源监控] 内存使用: %.1f/%.1f MB | CPU负载: %s%%\n', ...
                   total_mem-free_mem, total_mem, cpu_load{1});
        catch
            fprintf('资源监控仅支持Windows系统\n');
        end
    end

    % 进度更新函数
    function updateProgress(~)
        progressCount = progressCount + 1;
        elapsed = toc(totalStart);
        remaining = (elapsed/progressCount)*(total_points - progressCount);
        
        fprintf('进度: %.1f%%, 已用时间: %.1fs, 预计剩余: %.1fs\n', ...
               progressCount/total_points*100, ...
               elapsed, ...
               remaining);
    end

end

% 嵌套函数保持与原始代码一致
function residual = shooting_objective(omega, deri, options)
    [~, Er] = ode45(@(x, Er) evfun(x, Er, omega), [0, 1], [0, deri], options);
    residual = Er(end, 1);
end

function yy = evfun(x, Er, omg_val)
    % 保持原有物理模型计算不变
    rho0 = 0.01;
    gammai = 5/3;
    tau = 1.0;
    q = 1.05 + 4*x.^2;
    T = 0.2 + 0.8*(1 - x.^2).^2;

    OmegaASq = T ./ (2*q.^2);
    OmegaGSq = T .* (1 + 1./(2*q.^2));
    NumeratorDrd = (omg_val^2 - OmegaASq) .* (omg_val^2 - OmegaGSq);
    DenominatorDrd = 2*(gammai + tau)*T.^2.*omg_val^2 + 1e-6;

    Drd = NumeratorDrd ./ DenominatorDrd;

    yy = [Er(2);
    (4/(rho0^2)) * Drd * Er(1)];
end
