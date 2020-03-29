%% Reconocimiento de la felicidad
%Limpiar el Workspace
clear all
close all
clc

%% n es el número de emociones que tenga dividido en carpetas el database
n = 2;

%% Se carga el database
database = imageDatastore('database_happiness','IncludeSubfolders',true,...
    'LabelSource','foldernames');

%% Montaje en pantalla de la primera cara neutral y feliz
figure;
subplot(1,2,1);montage(database.Files(1));
title('Imágen de una cara feliz');
subplot(1,2,2);montage(database.Files(1+200));
title('Imágen de una cara neutral');

%% Mostrar imagen de consulta y base de datos lado a lado
personToQuery = 15;
figure;
subplot(2,2,1);montage(database.Files(personToQuery));
title('Imágen de una cara feliz');
subplot(2,2,3);montage(database.Files(personToQuery+200));
title('Imágen de una cara neutral');
subplot(2,2,[2 4]);montage(database);
title('Todo el database');

%% Se extrae y muestra en pantalla las características del histograma de
% gradientes orientados(HOG) para una sola cara
personToQuery = 11;
[~, visualization]= ...
    extractHOGFeatures(readimage(database,personToQuery));
figure;
subplot(1,2,1);imshow(readimage(database,personToQuery));
title('Cara de entrada');
subplot(1,2,2);imshow(readimage(database,personToQuery));
hold on;plot(visualization);title('Descriptor de características HOG');

%% Se cambia el tamaño de las imagenes al tamaño de entrada de la red
database.ReadFcn = @(loc)imresize(imread(loc),[227,227]);

%% Se parte la base de datos en sección de entrenamiento y test en una
% relación de 80 a 20
[Training ,Test] = splitEachLabel(database,0.8,'randomized');

%% Extraer características de HOG para el conjunto de entrenamiento
[hogFeature, visualization]= ...
    extractHOGFeatures(readimage(database,personToQuery));
cellSize = visualization.CellSize; 
hogFeatureSize = length(hogFeature);
numImages = numel(Training.Files);
trainingFeatures = zeros(numImages, hogFeatureSize, 'single');
for i = 1:numImages
    img = readimage(Training, i);
    img = rgb2gray(img);
    % Aplicar pasos de pre-procesamiento
    img = imbinarize(img);
    trainingFeatures(i, :) = extractHOGFeatures(img, 'CellSize', cellSize);  
end
% Obtener etiquetas para cada imagen
trainingLabels = Training.Labels;

%% Se copia las capas de Alexnet y se modifican las capas 23 y 25 
% La capa 23 por una "capa totalmente conectada" que recibe la cantidad de
% emociones que tiene el database
% La capa 25 por una "capa de clasificación", esta a su vez sera la ultima
% capa o, capa de salida"
fc = fullyConnectedLayer(n);
net = alexnet;
ly = net.Layers;
ly(23) = fc;
cl = classificationLayer;
ly(25) = cl;

%% Se configuran las opciones para entrenar la red como:
% La tasa de aprendizaje, la cantidad de epocas de entrenamiento y minimo
% tamaño de lote (cantidad de ejemplos de capacitación utilizados en una
% iteración, mayor que 1 pero menor que la cantidad de datos)
% Una configuración importante a tener en cuenta es el primer parametro del
% trainingOptions, estos pueden ser 'sgdm' | 'rmsprop' | 'adam', eh indican
% que tipo optimizador por descenso del gradiente se usara.

learning_rate = 0.00001;
opts = trainingOptions("rmsprop","InitialLearnRate",learning_rate,...
    'MaxEpochs',20,'MiniBatchSize',64,'Plots','training-progress');
[newnet,info] = trainNetwork(Training, ly, opts);

%% Extraer características de HOG para el conjunto de prueba
[hogFeature, visualization]= ...
    extractHOGFeatures(readimage(database,personToQuery));
cellSize = visualization.CellSize; 
hogFeatureSize = length(hogFeature);
numImages = numel(Training.Files);
trainingFeatures = zeros(numImages, hogFeatureSize, 'single');
for i = 1:numImages
    img = readimage(Training, i);
    img = rgb2gray(img);
    % Aplicar pasos de pre-procesamiento
    img = imbinarize(img);
    trainingFeatures(i, :) = extractHOGFeatures(img, 'CellSize', cellSize);  
end
% Obtener etiquetas para cada imagen
trainingLabels = Training.Labels;

%% Se clasifican los datos de prueba con la red entrenada
[predict,scores] = classify(newnet,Test);

%% Se mide la precisión de la red entrenada con los datos de prueba
% Se toma las etiquetas del conjunto de pruebas y se guardan en names
names = Test.Labels;

% Se guardan en pred las coincidencias entre las etiquetas del conjunto de
% prueba y las que arrojo la red neuronal.
pred = (predict==names);

% Se hace el calculo del porcentaje de precisión sumando la cantidad de
% aciertos y dividiendolo entre la cantidad de datos (Imagenes)
s = size(pred);
acc = sum(pred)/s(1);

% Se imprime en pantalla el porcentaje de precisión
fprintf('La precisión del conjunto de prueba es %f %% \n',acc*100);

%% Probar la red neuronal con una imagen
img = imread('15.jpg');%Cambiar por la imagen a probar
[img,face] = cropface(img);

% El valor de face sera 1 si detecta una cara, en caso contrario sera 0
if face == 1
    %% Se extrae las características del histograma de gradientes
    % orientados(HOG) para una sola cara
    [hogFeature, visualization]= ...
    extractHOGFeatures(img);

    %% Se muestra en pantalla la cara detectada y el histograma de
    % gradientes orientados (HOG)
    figure;
    subplot(1,2,1);imshow(img);
    title('Cara de entrada');
    subplot(1,2,2);imshow(img);
    hold on;plot(visualization);title('Descriptor de características HOG');
    
    %% Se cambia el tamaño de las imagenes al tamaño de entrada de la red.
    img = imresize(img,[227 227]);
    
    %% Se introduce la imagen a la red neuronal para ser analizada.
    predictFace = classify(newnet,img);
    
    %% Mensaje que saldra al detectar un rostro
    nameOfEmotion01 = 'Rostro neutral';
    nameOfEmotion02 = 'Rostro feliz';

    %% Se comprueba que tipo de emoción es la encontrada y saca un mensaje
    % en consola del tipo de cara detectada.
    if predictFace=='neutral'
        fprintf('The face detected is %s',nameOfEmotion01);
    elseif  predictFace=='happy'
        fprintf('The face detected is %s',nameOfEmotion02);
    end
    
end

%%
% podemos usar [predict,score] = classify(newnet,img) aquí puntaje dice el
% porcentaje que cuán confianza es