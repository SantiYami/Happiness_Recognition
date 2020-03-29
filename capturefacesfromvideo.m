%% Captura reconocimiento de caras a traves de la camara web.
%Limpiar el Workspace
clear all
close all
clc

%% Crear el objeto detector de cara.
% Este es obtenido gracias a la libreria vision toolbox de MATLAB
faceDetector = vision.CascadeObjectDetector('FrontalFaceCART','MinSize',...
    [150,150]);

%% Aquí n la cantidad de veces que se ejecuta el ciclo, se puede cambiar
% el umbral (n) en función de la cantidad de datos que se necesite
n = 400;

%% Cambie str a s01, s02, s03, .... para guardar hasta cuántos individuos
% desee guardar en las carpetas respectivas con la función imwrite en 
% la línea 138.
str = 's01';

%% Se crear el objeto rastreador de puntos.
% Estos seran los que apareceran mientras este activa la camara web
% indicando los sectores claves para la detección de la cara.
pointTracker = vision.PointTracker('MaxBidirectionalError', 2);

%% Se crear el objeto de la cámara web.
cam = webcam();

%% Se captura un frame (cuadro/imagen) de video de la camara web para
% obtener el tamaño que debe tener el video.
videoFrame = snapshot(cam);
frameSize = size(videoFrame);

%% Se crear el objeto de reproducción de video.
videoPlayer = vision.VideoPlayer('Position', [100 100 [frameSize(2),...
    frameSize(1)]+30]);
runLoop = true;
numPts = 0;
frameCount = 0;
i=1;

%% Se inicia el ciclo para la reproducción de video por la camara web y
% para obtener las imegenes de caras de los individuos
while runLoop && frameCount < n

    % Se obtiene el siguiente frame (cuadro/imagen) de la camara web y se
    % guarda para ser despues reproducido en video.
    videoFrame = snapshot(cam);
    
    % Se debe convertir el frame (cuadro/imagen) de la camara web a escala
    % de grises para poder ser analizado.
    videoFrameGray = rgb2gray(videoFrame);
    frameCount = frameCount + 1;

    if numPts < 10
        % Sistema de detección.
        % Se toma el frame en escala de grises y se pasa por el objeto
        % previamente de detección de caras, si se detecta una cara, esta
        % se guardara en una variable.
        bbox = faceDetector.step(videoFrameGray);
        
        % Si se detecto la cara entra a esta comprobación.
        if ~isempty(bbox)
            % Encuentra puntos de esquina dentro de la región detectada,
            % haciendo que se delimiten los limites de la cara.
            points = detectMinEigenFeatures(videoFrameGray, 'ROI',...
                bbox(1, :));

            % Se guardan las coordenadas de los limites de la cara, y se
            % inicializa para empezar a mostrarse en pantalla.
            xyPoints = points.Location;
            numPts = size(xyPoints,1);
            release(pointTracker);
            initialize(pointTracker, xyPoints, videoFrameGray);

            % Guardar una copia de los puntos de esquina.
            oldPoints = xyPoints;

            % Se convierte el rectángulo representado como [x, y, w, h] en 
            % una matriz M por 2 de coordenadas [x, y] de las esquinas del
            % recuadro.
            % Esto es necesario para poder transformar el recuadro
            % delimitador para mostrar la orientación de la cara.
            bboxPoints = bbox2points(bbox(1, :));

            % Se convierte las esquinas de la caja en formato
            % [x1 y1 x2 y2 x3 y3 x4 y4] el cual es el formato requerido por
            % insertShape.
            bboxPolygon = reshape(bboxPoints', 1, []);

            % insertShape saca un cuadro delimitador al rededor de la cara 
            % detectada y este se muestra en el video.
            videoFrame = insertShape(videoFrame, 'Polygon',...
                bboxPolygon, 'LineWidth', 3);

            % Se visualizan los puntos de seguimiento.
            videoFrame = insertMarker(videoFrame, xyPoints, '+',...
                'Color', 'white');
        end

    else
        % Modo de seguimiento.
        % Esto seguira (rastreara) la cara analizando los puntos de esquina
        % viejos y los actuales.
        [xyPoints, isFound] = step(pointTracker, videoFrameGray);
        visiblePoints = xyPoints(isFound, :);
        oldInliers = oldPoints(isFound, :);
        
        % Se toma un puntaje de la cantidad de puntos claves para la
        % detección de una cara.
        numPts = size(visiblePoints, 1);
        
        % Si estos puntos son mayores de 10 se entra al sistema para
        % obtener la persona.
        % Esto es porque si son mas de 10 indica que la cara esta en angulo
        % en el que son visibles la mayor cantidad se zonas claves para ser
        % analizadas en el HOG_Happiness_Recognition (El detector de la
        % felicidad).
        % En caso contrario vuelve a analizar el siguiente frame para
        % encontrar un rostro en ese.
        if numPts >= 10
            
            % Se estima la transformación geométrica entre los puntos de
            % esquina antiguos y los puntos nuevos.
            [xform, oldInliers, visiblePoints] = ...
                estimateGeometricTransform(oldInliers, visiblePoints,...
                'similarity', 'MaxDistance', 4);

            % Se aplicar una transformación al cuadro delimitador para
            % obtener sus puntos claves.
            bboxPoints = transformPointsForward(xform, bboxPoints);
            
            % Se crea la carpeta en la que estaran guardadas las imagenes
            % con la cara detectada.
            mkdir('photos',str);
            
            % Se guardan las imagenes.
            imwrite(videoFrame,fullfile('photos',str,[int2str(i),...
                '.jpg']));
            
            % Se convierte las esquinas de la caja en formato
            % [x1 y1 x2 y2 x3 y3 x4 y4] el cual es formato requerido por
            % insertShape.
            bboxPolygon = reshape(bboxPoints', 1, []);
            
            % insertShape saca un cuadro delimitador al rededor de la cara 
            % detectada y este se muestra en el video.
            videoFrame = insertShape(videoFrame, 'Polygon', bboxPolygon,...
                'LineWidth', 3);

            % Se visualizan los puntos de seguimiento.
            videoFrame = insertMarker(videoFrame, visiblePoints, '+',...
                'Color', 'white');
            
            % Se restablece los puntos y vuelve a iniciarse el ciclo
            oldPoints = visiblePoints;
            setPoints(pointTracker, oldPoints);
            i = i+1;
        end

    end

    % Se visualiza el cuadro de video guardado utilizando el objeto
    % reproducción de video.
    step(videoPlayer, videoFrame);

    % Se comprueba si la ventana del reproductor de video se ha cerrado.
    runLoop = isOpen(videoPlayer);
end

%% Se limpian las variables de video.
clear cam;
release(videoPlayer);
release(pointTracker);
release(faceDetector);
