%% Reconocimiento de la felicidad
%Limpiar el Workspace
clear all
close all
clc

%% n es el n�mero de emociones que tenga dividido en carpetas el database
n = 2;

%% Se carga el database
database = imageDatastore('database_happiness','IncludeSubfolders',true,...
    'LabelSource','foldernames');

%% Montaje en pantalla de la primera cara neutral y feliz
figure;
subplot(1,2,1);montage(database.Files(1));
title('Im�gen de una cara feliz');
subplot(1,2,2);montage(database.Files(1+200));
title('Im�gen de una cara neutral');

%% Mostrar imagen de consulta y base de datos lado a lado
personToQuery = 15;
figure;
subplot(2,2,1);montage(database.Files(personToQuery));
title('Im�gen de una cara feliz');
subplot(2,2,3);montage(database.Files(personToQuery+200));
title('Im�gen de una cara neutral');
subplot(2,2,[2 4]);montage(database);
title('Todo el database');

%% Se extrae y muestra las caracter�sticas del histograma de gradientes orientados(HOG) para una sola cara
personToQuery = 11;
[hogFeature, visualization]= ...
    extractHOGFeatures(readimage(database,personToQuery));
figure;
subplot(1,2,1);imshow(readimage(database,personToQuery));
title('Cara de entrada');
subplot(1,2,2);imshow(readimage(database,personToQuery));
hold on;plot(visualization);title('Descriptor de caracter�sticas HOG');

%% Se cambia el tama�o de las imagenes al tama�o de entrada de la red
database.ReadFcn = @(loc)imresize(imread(loc),[227,227]);

%% Se parte la base de datos en secci�n de entrenamiento y test en una rel%aci�n de 80 a 20
[Training ,Test] = splitEachLabel(database,0.8,'randomized');

%% Extraer caracter�sticas de HOG para el conjunto de entrenamiento
[hogFeature, visualization]= ...
    extractHOGFeatures(readimage(database,personToQuery));
cellSize = visualization.CellSize; 
hogFeatureSize = length(hogFeature);
numImages = numel(Training.Files);
trainingFeatures = zeros(numImages, hogFeatureSize, 'single');
for i = 1:numImages
    img = readimage(Training, i);
    img = rgb2gray(img);
    % Apply pre-processing steps
    img = imbinarize(img);
    trainingFeatures(i, :) = extractHOGFeatures(img, 'CellSize', cellSize);  
end
% Get labels for each image.
trainingLabels = Training.Labels;

%% Se copia las capas de Alexnet y se modifican las capas 23 y 25 
% La capa 23 por una "capa totalmente conectada" que recibe la cantidad de
% emociones que tiene el database
% La capa 25 por una "capa de clasificaci�n", esta a su vez sera la ultima
% capa o, capa de salida"
fc = fullyConnectedLayer(n);
net = alexnet;
ly = net.Layers;
ly(23) = fc;
cl = classificationLayer;
ly(25) = cl;

%
%dt = datastore(trainingFeatures,'IncludeSubfolders',true,...
%    'LabelSource',trainingLabels);

%% Se configuran las opciones para entrenar la red como:
% La tasa de aprendizaje, la cantidad de epocas de entrenamiento y minimo
% tama�o de lote (cantidad de ejemplos de capacitaci�n utilizados en una
% iteraci�n, mayor que 1 pero menor que la cantidad de datos)
% Una configuraci�n importante a tener en cuenta es el primer parametro del
% trainingOptions, estos pueden ser 'sgdm' | 'rmsprop' | 'adam', eh indican
% que tipo optimizador por descenso del gradiente se usara.

learning_rate = 0.00001;
opts = trainingOptions("rmsprop","InitialLearnRate",learning_rate,...
    'MaxEpochs',20,'MiniBatchSize',64,'Plots','training-progress');
[newnet,info] = trainNetwork(Training, ly, opts);

%% Se mide la precisi�n de la red entrenada con los datos de prueba
[predict,scores] = classify(newnet,Test);
names = Test.Labels;
pred = (predict==names);
s = size(pred);
acc = sum(pred)/s(1);
fprintf('La precisi�n del conjunto de prueba es %f %% \n',acc*100);

%% Probar la base de datos con una imagen
img = imread('D:\Documentos\David\U.DISTRITAL\Semestres\Semestre10\Teleinformatica\Happiness_Recognition\11.jpg');
[img,face] = cropface(img);
% El valor de face sera 1 si detecta una cara, en caso contrario sera 0
if face == 1
    img = imresize(img,[227 227]);
    predict = classify(newnet,img)
end
nameofs01 = 'Rostro neutral';
nameofs02 = 'Rostro feliz';
if predict=='neutral'
    fprintf('The face detected is %s',nameofs01);
elseif  predict=='happy'
    fprintf('The face detected is %s',nameofs02);
end
%%
% podemos usar [predict,score] = classify(newnet,img) aqu� puntaje dice el
% porcentaje que cu�n confianza es