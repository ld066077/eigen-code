function tt = shootev()
    format long;
    tol = 1.5e-4;

    % 如果不存在，则创建plots目录
    if ~exist('plots', 'dir')
        mkdir('plots');
    end

    omega_initial = 1.443569123412118 + 0.000010666666667i; % k的初始值 (omega的初始猜测值)

    % 3. 提供初始猜测 (重要：bvp4c需要一个初始猜测)
    solinit = bvpinit(linspace(0, 1, 10), @guess, omega_initial);

    % 4. 使用bvp4c求解BVP, 将omega作为参数传递
    sol = bvp4c(@evfun_wrapper, @bc_wrapper, solinit);

    % 5. 优化本征值 (使用fzero进行细化)
    omega = fzero(@(omega) refine_eigenvalue(omega), sol.parameters);

    % Update sol structure and parameters *before* re-solving
    solinit.parameters = omega;
    sol = bvp4c(@evfun_wrapper, @bc_wrapper, solinit);

    % 6. 提取解和绘图
    x = sol.x;
    Er = sol.y;

    % 绘图并保存
    fig = figure('Visible', 'off');
    plot(x, real(Er(1,:)), 'LineWidth', 1.5, 'DisplayName', '实部');
    hold on;
    plot(x, imag(Er(1,:)), 'LineWidth', 1.5, 'DisplayName', '虚部');
    title(sprintf('omega=%.8f + %.8fi', real(omega), imag(omega)));
    xlabel('x');
    ylabel('Er(x)');
    legend('show');
    grid on;
    saveas(fig, fullfile('plots', sprintf('eigenfunc.png')));
    close(fig);

    tt = omega;
    disp(tt);

    % --- 辅助函数 ---

    % 初始猜测函数
    function y = guess(x)
        y = [sin(pi*x);
             pi*cos(pi*x)];
    end

    % 细化本征值的函数
    function residual = refine_eigenvalue(omega)
        % Update solinit.  Crucially, do this *before* the bvp4c call.
        solinit.parameters = omega;
        sol = bvp4c(@evfun_wrapper, @bc_wrapper, solinit);
        Er_end = sol.y(1,end);
        residual = Er_end;
    end

    % evfun 的包装函数 (nested function)
    function dydx = evfun_wrapper(x, y)
        dydx = evfun(x, y);
    end

    % bc 的包装函数 (nested function)
    function res = bc_wrapper(ya, yb)
        res = bc(ya, yb);
    end


    % 1. 定义边界条件函数 (nested function)
    function res = bc(ya, yb)
        omega = solinit.parameters; % Access omega from solinit
        res = [ya(1);  % Er(0) = 0
               yb(1)]; % Er(1) = 0
    end

% 2. 定义微分方程函数 (nested function with nargin check)
function yy = evfun(x, Er)
    if nargin == 2
        % Check if solinit is available.  If not, it's the initialization
        % phase, so return a dummy output.
        if ~isfield(solinit, 'parameters')
            yy = [0; 0]; % Dummy output of the correct size
            return;  % Exit early
        end

        % If we get here, solinit IS available, and it's the solution phase.
        omega = solinit.parameters;

        Zdf = @(zeta_val) -2*dawson(zeta_val) + 1i*sqrt(pi)*exp(-zeta_val.^2);
        rho0 = 0.01;
        tau = 1.0;
        q = @(x) 1.05 + 4*x.^2;
        T = @(x) 0.2 + 0.8*(1 - x.^2).^2;

        rho = sqrt(T(x))*rho0;
        a1 = rho*rho;
        zeta_val = q(x)*omega/sqrt(T(x));
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
    else
        error('Incorrect number of input arguments to evfun.');

    end
end

end
