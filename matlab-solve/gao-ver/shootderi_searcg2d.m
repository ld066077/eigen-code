function shootev_search2d()
    format long;
    
    % 创建plots目录
    if ~exist('plots', 'dir')
        mkdir('plots');
    end
    
    omega = 1.825833489577557 - 0.090343626229904i;

    % ODE求解参数
    % 定义参数扫描范围
    reald = linspace(0.0004, -0.0004, 20);    % 实部扫描范围
    imagd = linspace(0.0004, -0.0004, 20); % 虚部扫描范围
    [Re, Im] = meshgrid(reald, imagd);    % 生成网格
    deri_list = Re(:) + 1i*Im(:);
    total_points = numel(deri_list);
    residual_flat = zeros(total_points, 1);
    options = odeset('AbsTol', 1e-8, 'RelTol', 1e-8);

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
            current_deri = deri_list(k);
            % 将参数显式传递进目标函数
            residual = shooting_objective(omega, current_deri, options);
            residual_flat(k) = norm(residual);
            
            % 进度更新
            send(progressQueue, []);
        end

        % 结果重构为二维矩阵
        residual_values = reshape(residual_flat, size(Re));

        % 终止资源监控
        stop(monitorTimer);
        delete(monitorTimer);
        % ========== 三维可视化 ==========
        fig = figure('Position', [100 100 1200 800]);
        
        % 曲面图
        subplot(2,2,[1 3]);
        surf(Re, Im, residual_values, 'EdgeColor', 'none');
        title('目标函数值曲面图');
        xlabel('Re(deri)');
        ylabel('Im(deri)');
        zlabel('||Residual||');
        view(-30, 30);
        colormap jet;
        colorbar;
        
        % 等高线图
        subplot(2,2,2);
        contourf(Re, Im, log10(residual_values), 20);
        title('对数坐标等高线图');
        xlabel('Re(deri)');
        ylabel('Im(deri)');
        colorbar;
        
        % 二维投影图
        subplot(2,2,4);
        imagesc(reald, imagd, residual_values);
        set(gca,'YDir','normal');
        title('二维投影热图');
        xlabel('Re(deri)');
        ylabel('Im(deri)');
        colorbar;
        
        % 保存图像
        saveas(fig, fullfile('plots', 'residual_landscape.png'));
        % close(fig);
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
            
            fprintf('[资源监控] 内存使用: %.1f/%.1f MB | CPU负载: %s%%\n',...
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
        
        fprintf('进度: %.1f%%, 已用时间: %.1fs, 预计剩余: %.1fs\n',...
               progressCount/total_points*100,...
               elapsed,...
               remaining);
    end

end
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