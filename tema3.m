
% Încarc și preprocesez datele
data = load('spambase.data');

% Împart datele în caracteristici și etichete
X = data(:, 1:end-1);
y = data(:, end);

% Normalizez caracteristicile
[X, mu, sigma] = zscore(X);

% Împart datele în seturi de antrenament și testare
train_ratio = 0.8;
[trainInd, ~, testInd] = dividerand(size(X, 1), train_ratio, 0, 1 - train_ratio);

X_train = X(trainInd, :);
y_train = y(trainInd);
X_test = X(testInd, :);
y_test = y(testInd);

% Inițializez parametrii
initial_w = zeros(size(X_train, 2), 1);
alpha = 0.01;
num_iters = 400;
lambda = 1e-4;
constrangere_limita = 1; % Norma L2 limită

% Antrenez modelul folosind gradient descent și măsor timpul
[w_gd, J_history_gd, times_gd] = gradientDescent(X_train, y_train, initial_w, alpha, num_iters, lambda, constrangere_limita);

% Antrenez modelul folosind metoda Newton și măsor timpul
[w_newton, J_history_newton, times_newton] = newtonMethod(X_train, y_train, initial_w, num_iters, lambda, constrangere_limita);

% Salvez modelul antrenat
save('spamModel.mat', 'w_gd', 'w_newton', 'mu', 'sigma');

% Evaluez performanța pe setul de testare pentru gradient descendent
pred_test_gd = sigmoid(X_test * w_gd) >= 0.5;
accuracy_test_gd = mean(pred_test_gd == y_test);
fprintf('Acuratețea pe setul de testare (Gradient Descent): %.2f%%\n', accuracy_test_gd * 100);


y_test = double(y_test);
pred_test_gd = double(pred_test_gd);

% Calculez matricea de confuzie pentru gradient descendent
confusion_matrix_gd = confusionmat(y_test, pred_test_gd);
disp('Matricea de confuzie pentru Gradient Descent:');
disp(confusion_matrix_gd);

% Evaluez performanța pe setul de testare pentru metoda Newton
pred_test_newton = sigmoid(X_test * w_newton) >= 0.5;
pred_test_newton = double(pred_test_newton); % Convertire la double pentru compatibilitate
accuracy_test_newton = mean(pred_test_newton == y_test);
fprintf('Acuratețea pe setul de testare (Newton Method): %.2f%%\n', accuracy_test_newton * 100);

% Calculez matricea de confuzie pentru metoda Newton
confusion_matrix_newton = confusionmat(y_test, pred_test_newton);
disp('Matricea de confuzie pentru Newton Method:');
disp(confusion_matrix_newton);

% Plotez funcția de cost
figure;
subplot(2,2,1);
plot(1:num_iters, J_history_gd, 'LineWidth', 2);
xlabel('Numărul de iterații');
ylabel('Funcția de cost J');
title('Evoluția funcției de cost - Gradient Descent');
grid on;

subplot(2,2,2);
plot(times_gd, J_history_gd, 'LineWidth', 2);
xlabel('Timpul acumulat (secunde)');
ylabel('Funcția de cost J');
title('Evoluția funcției de cost în funcție de timp - Gradient Descent');
grid on;

subplot(2,2,3);
plot(1:num_iters, J_history_newton, 'LineWidth', 2);
xlabel('Numărul de iterații');
ylabel('Funcția de cost J');
title('Evoluția funcției de cost - Newton Method');
grid on;

subplot(2,2,4);
plot(times_newton, J_history_newton, 'LineWidth', 2);
xlabel('Timpul acumulat (secunde)');
ylabel('Funcția de cost J');
title('Evoluția funcției de cost în funcție de timp - Newton Method');
grid on;

% Afisez timpul total de antrenare pentru ambele metode
fprintf('Timpul total de antrenare (Gradient Descent): %.2f secunde\n', times_gd(end));
fprintf('Timpul total de antrenare (Newton Method): %.2f secunde\n', times_newton(end));

% Utilizarea cu CVX 
cvx_begin
    variable w_cvx(size(X_train, 2))
    minimize( sum(log(1 + exp(-y_train .* (X_train * w_cvx)))) + (lambda/2) * sum_square(w_cvx) )
    subject to
        norm(w_cvx, 2) <= constrangere_limita  % Constrângerea pe norma L2 a vectorului de parametri
cvx_end

% Evaluez performanța modelului CVX pe setul de testare
pred_test_cvx = sigmoid(X_test * w_cvx) >= 0.5;
accuracy_test_cvx = mean(pred_test_cvx == y_test);
fprintf('Acuratețea modelului CVX pe setul de testare: %.2f%%\n', accuracy_test_cvx * 100);

% Calculez matricea de confuzie pentru modelul CVX
confusion_matrix_cvx = confusionmat(y_test, double(pred_test_cvx));
disp('Matricea de confuzie pentru modelul CVX:');
disp(confusion_matrix_cvx);
% Definesc funcția de cost pentru fmincon
costFunc = @(w) fct(w, X_train, y_train, lambda);

% Definesc funcția de constrângere pentru fmincon
nonlcon = @(w) deal([], norm(w, 2) - constrangere_limita);

% Utilizez fmincon 
options = optimoptions('fmincon', 'Algorithm', 'interior-point', 'Display', 'iter');
[w_fmincon, J_fmincon] = fmincon(costFunc, initial_w, [], [], [], [], [], [], nonlcon, options);

% Evaluez performanța modelului fmincon
pred_test_fmincon = sigmoid(X_test * w_fmincon) >= 0.5;
accuracy_test_fmincon = mean(pred_test_fmincon == y_test);
fprintf('Acuratețea pe setul de testare (fmincon): %.2f%%\n', accuracy_test_fmincon * 100);

% Calculez matricea de confuzie pentru fmincon
confusion_matrix_fmincon = confusionmat(y_test, double(pred_test_fmincon));
disp('Matricea de confuzie pentru fmincon:');
disp(confusion_matrix_fmincon);

% Definesc funcția sigmoidă
function g = sigmoid(z)
    g = 1 ./ (1 + exp(-z));
end

% Definesc funcția costului cu regularizare
function [J, grad] = fct(w, X, y, lambda)
    m = length(y); % Numărul de exemple de antrenament
    h = sigmoid(X * w);
    J = (1/m) * sum(-y .* log(h) - (1 - y) .* log(1 - h)) + (lambda/(2*m)) * sum(w(2:end).^2);

    grad = (1/m) * (X' * (h - y));
    grad(2:end) = grad(2:end) + (lambda/m) * w(2:end);
end

% Definesc funcția Hessiana pentru metoda Newton
function H = hessiana(w, X, y, lambda)
    m = length(y); % Numărul de exemple de antrenament
    h = sigmoid(X * w);
    R = diag(h .* (1 - h));
    H = (1/m) * (X' * R * X) + (lambda/m) * diag([0; ones(length(w) - 1, 1)]);
end

% Gradient Descent pentru regresia logistică cu măsurarea timpului și constrângeri
function [w, J_history, times] = gradientDescent(X, y, w, alpha, num_iters, lambda, constrangere_limita)
    J_history = zeros(num_iters, 1);
    times = zeros(num_iters, 1);
    tic;
    for iter = 1:num_iters
        [J, grad] = fct(w, X, y, lambda);
        w = w - alpha * grad;

        % Aplicarea constrângerii
        if norm(w, 2) > constrangere_limita
            w = (constrangere_limita / norm(w, 2)) * w;
        end

        J_history(iter) = J;
        times(iter) = toc;
    end
end

% Constrângerea
function w_p = pL2(w, constrangere_limita)
    norm_w = norm(w, 2);
    if norm_w > constrangere_limita
        w_p = (constrangere_limita / norm_w) * w;
    else
        w_p = w;
    end
end

% Metoda Newton pentru regresia logistică cu măsurarea timpului și constrângeri
function [w, J_history, times] = newtonMethod(X, y, w, num_iters, lambda, constrangere_limita)
    J_history = zeros(num_iters, 1);
    times = zeros(num_iters, 1);
    tic;
    for iter = 1:num_iters
        [J, grad] = fct(w, X, y, lambda);
        H = hessiana(w, X, y, lambda);
        % Stabilizarea Hessianei
        [V, D] = eig(H);
        D(D < 1e-4) = 1e-4; 
        H = V * D * V';
        p = - inv(H) * grad;
        alpha = 1;
        while true
            w_new = w + alpha * p;
            w_new = pL2(w_new, constrangere_limita);
            if fct(w_new, X, y, lambda) < J
                break;
            end
            alpha = alpha / 2;
            if alpha < 1e-8 
                break;
            end
        end
        w = w_new;
        J_history(iter) = J;
        times(iter) = toc;
    end
end
