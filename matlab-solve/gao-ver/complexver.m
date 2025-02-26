function tt = shootev()
    format long;
    tol = 1.5e-4;
    m = 12;
    omg_log = zeros(1, m) + 1i * zeros(1, m); % Store complex eigenvalues

    % Suppress ODE solver output
    options = odeset('AbsTol', 1e-8, 'RelTol', 1e-8);

    % Create plots directory if it doesn't exist
    if ~exist('plots', 'dir')
        mkdir('plots');
    end

    omega_initial = 0.35 + 0.1i;  % Initial guess for omega (now complex)
    deri = 0.0001;

    for n = 1:m  % Compute the first m eigenvalues
        omega = omega_initial;
        domg = omega / 30;
        omega = omega + domg;

        % Pass omega to evfun using an anonymous function
        evfun_handle = @(x, Er) evfun(x, Er, omega);
        [x, Er] = ode15s(evfun_handle, [0, 1], [0, deri], options);
        oldEr = Er(end, 1);
        dEr = oldEr;

        while abs(dEr) > tol  % Check magnitude of the error
            omega = omega + domg;
            % Update the function handle with the new omega
            evfun_handle = @(x, Er) evfun(x, Er, omega);
            [x, Er] = ode15s(evfun_handle, [0, 1], [0, deri], options);
            dEr = Er(end, 1);

            if real(dEr) * real(oldEr) < 0  % Check real parts
                omega = omega - domg;
                domg = domg / 2;
                omega = omega+domg;
            end
            if imag(dEr) * imag(oldEr) < 0  % Check imaginary parts, too.
                omega = omega-domg;
                domg = domg/2;
                omega = omega+domg;
            end
            oldEr = dEr;  % Crucial: Update oldEr for the next iteration


            fprintf('------\n');
            display(omega);
            display(domg);
            display(dEr);
        end
        omg_log(n) = omega;  % Store the eigenvalue
        omega_initial = omega; % Key: Use for next iteration

        % Plot and save (using the stored x and Er)
        fig = figure('Visible', 'off');
        % Er(:, 1) = Er(:, 1) / max(abs(Er(:, 1)));  % Normalize by magnitude
        plot(x, real(Er(:, 1)), 'LineWidth', 1.5, 'DisplayName', 'Real');
        hold on;
        plot(x, imag(Er(:, 1)), 'LineWidth', 1.5, 'DisplayName', 'Imag');
        title(sprintf('omega=%.4f + %.4fi', real(omega), imag(omega)));
        xlabel('x');
        ylabel('Er(x)');
        legend('show');
        grid on;
        saveas(fig, fullfile('plots', sprintf('eigenfunc_%d.png', n)));
        close(fig);
    end

    tt = omg_log;
    disp(tt);

    % Composite plot (solve ODE *once* for each omega, using the stored omegas)
    composite_fig = figure('Visible', 'off');
    hold on;
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'r', 'g', 'b', 'c', 'm','y','k'];  % Removed 'w'
    for n = 1:m
        % Create function handle for the *stored* omega
        evfun_handle = @(x, Er) evfun(x, Er, omg_log(n));
        [x, Er] = ode15s(evfun_handle, [0, 1], [0, deri], options);
        Er(:, 1) = Er(:, 1) / max(abs(Er(:, 1)));  % Normalize by magnitude
         plot(x, real(Er(:, 1)), colors(n), 'LineWidth', 1.2, ...
            'DisplayName', sprintf('Re(omega)=%.4f', real(omg_log(n))));
        plot(x, imag(Er(:, 1)), [colors(n), '--'], 'LineWidth', 1.2, ...  % Dashed line
            'DisplayName', sprintf('Im(omega)=%.4f', imag(omg_log(n))));

    end
    title('Composite Eigenfunctions');
    xlabel('x');
    ylabel('Er(x)');
    legend('show', 'Location', 'best'); % Added location for better legend placement
    grid on;
    saveas(composite_fig, fullfile('plots', 'composite_eigenfuncs.png'));
    close(composite_fig);


    % Nested function (now takes omega as a parameter)
    function yy = evfun(x, Er, omg_val)
        Zdf = @(zeta_val) -2*dawson(zeta_val) + 1i*sqrt(pi)*exp(-zeta_val.^2);
        rho0 = 0.01;
        tau = 1.0;
        q = @(x) 1.05 + 4*x.^2;
        zeta = @(x) omg_val ./ (sqrt(2) * q(x));  % Use omg_val here
        a1 = -rho0 * sqrt(pi) * (1.0 + tau) / sqrt(2.0);
        a3 = -rho0 * tau * Zdf(zeta(x)) / sqrt(2.0);
        
        yy = [Er(2); ...
              -a3/a1 * Er(1)];
    end
end
