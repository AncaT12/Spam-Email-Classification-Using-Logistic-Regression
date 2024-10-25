clc;
clear;
% Încarc modelul antrenat
load('spamModel.mat', 'w_gd', 'w_newton', 'mu', 'sigma');

% Funcția sigmoid
sigmoid = @(z) 1 ./ (1 + exp(-z));

% Introduc textul email-ului
emailText = input('Introduceți textul email-ului: ', 's');

% Extrag caracteristicile
features_gd = extractFeatures(emailText, w_gd);
features_newton = extractFeatures(emailText, w_newton);

% Normalizez caracteristicile
normalizedFeatures_gd = (features_gd - mu) ./ sigma;
normalizedFeatures_newton = (features_newton - mu) ./ sigma;

% Calculez scorul folosind funcția sigmoidă și modelul antrenat cu Gradient Descent
score_gd = sigmoid(normalizedFeatures_gd * w_gd);
score_newton = sigmoid(normalizedFeatures_newton * w_newton);

% Afișez rezultatul pentru modelul Gradient Descent
if score_gd >= 0.5
    fprintf('Email-ul este Non-Spam (Gradient Descent).\n');
else
    fprintf('Email-ul este Spam (Gradient Descent).\n');
end

% Afișez rezultatul pentru modelul Newton
if score_newton >= 0.5
    fprintf('Email-ul este Non-Spam (Newton).\n');
else
    fprintf('Email-ul este Spam (Newton).\n');
end
% Funcția de extragere a caracteristicilor
function features = extractFeatures(emailText, w)
    % Caracteristici simple extrase din textul email-ului:
    % 1. Lungimea textului
    % 2. Numărul de cuvinte
    % 3. Prezența cuvintelor specifice (free, money, offer)
    % 4. Numărul de semne de exclamare
    % 5. Numărul de cifre
    % Caracteristica 1: Lungimea textului
    textLength = length(emailText);
    % Caracteristica 2: Numărul de cuvinte
    numWords = length(strsplit(emailText));
    % Caracteristica 3: Prezența cuvintelor specifice
    words = lower(strsplit(emailText));
    containsFree = sum(strcmp(words, 'free')) > 0;
    containsMoney = sum(strcmp(words, 'money')) > 0;
    containsOffer = sum(strcmp(words, 'offer')) > 0;
    % Caracteristica 4: Numărul de semne de exclamare
    numExclamations = sum(emailText == '!');
    % Caracteristica 5: Numărul de cifre
    numDigits = sum(isstrprop(emailText, 'digit'));
    % Combin toate caracteristicile într-un vector
    features = [textLength, numWords, containsFree, containsMoney, containsOffer, numExclamations, numDigits];
    % Mă asigur că vectorul de caracteristici('features') are aceeași dimensiune ca vectorul de parametri('w')
    if length(features) < length(w)
        features = [features, zeros(1, length(w) - length(features))];
    elseif length(features) > length(w)
        features = features(1:length(w));
    end
end
