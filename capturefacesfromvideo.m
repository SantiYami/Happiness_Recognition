clear all
close all
clc

% Crear el objeto detector de cara.
faceDetector = vision.CascadeObjectDetector('FrontalFaceCART','MinSize',[150,150]);

% Aquí el ciclo se ejecuta 50 veces, puede cambiar el umbral (n) en función de la cantidad de datos de entrenamiento que necesita
n = 400;

% cambie str a s01, s02, s03, .... para guardar hasta cuántos temas desea guardar para guardar en las carpetas respectivas para
% imwrite en la línea 88

str = 's02';

% Crear el objeto rastreador de puntos.
pointTracker = vision.PointTracker('MaxBidirectionalError', 2);

% Crear el objeto de la cámara web.
cam = webcam();

% Capture un cuadro para obtener su tamaño.
videoFrame = snapshot(cam);
frameSize = size(videoFrame);

% Crear el objeto del reproductor de video.
videoPlayer = vision.VideoPlayer('Position', [100 100 [frameSize(2), frameSize(1)]+30]);
runLoop = true;
numPts = 0;
frameCount = 0;
i=1;

while runLoop && frameCount < n

    % Obtenga el siguiente cuadro.
    videoFrame = snapshot(cam);
    videoFrameGray = rgb2gray(videoFrame);
    frameCount = frameCount + 1;

    if numPts < 10
        % Modo de detección.
        bbox = faceDetector.step(videoFrameGray);

        if ~isempty(bbox)
            % Encuentra puntos de esquina dentro de la región detectada.
            points = detectMinEigenFeatures(videoFrameGray, 'ROI', bbox(1, :));

            % Reinicialice el rastreador de puntos.
            xyPoints = points.Location;
            numPts = size(xyPoints,1);
            release(pointTracker);
            initialize(pointTracker, xyPoints, videoFrameGray);

            % Guardar una copia de los puntos.
            oldPoints = xyPoints;

            % Convierta el rectángulo representado como [x, y, w, h] en una
            % Matriz M por 2 de coordenadas [x, y] de las cuatro esquinas.
            % Esto es necesario para poder transformar el cuadro delimitador para mostrar
            % La orientación de la cara.
            bboxPoints = bbox2points(bbox(1, :));

            % Convertir las esquinas de la caja en [x1 y1 x2 y2 x3 y3 x4 y4]
            % formato requerido por insertShape.
            bboxPolygon = reshape(bboxPoints', 1, []);

            % Muestra un cuadro delimitador alrededor de la cara detectada.
            videoFrame = insertShape(videoFrame, 'Polygon', bboxPolygon, 'LineWidth', 3);

            % Visualizar esquinas detectadas.
            videoFrame = insertMarker(videoFrame, xyPoints, '+', 'Color', 'white');
        end

    else
        % Modo de seguimiento.
        [xyPoints, isFound] = step(pointTracker, videoFrameGray);
        visiblePoints = xyPoints(isFound, :);
        oldInliers = oldPoints(isFound, :);

        numPts = size(visiblePoints, 1);

        if numPts >= 10
            % Estima la transformación geométrica entre los puntos antiguos
            % y los nuevos puntos.
            [xform, oldInliers, visiblePoints] = estimateGeometricTransform(...
                oldInliers, visiblePoints, 'similarity', 'MaxDistance', 4);

            % Aplicar la transformación al cuadro delimitador.
            bboxPoints = transformPointsForward(xform, bboxPoints);

            % Convertir las esquinas de la caja en [x1 y1 x2 y2 x3 y3 x4 y4]
            % formato requerido por insertShape.
            bboxPolygon = reshape(bboxPoints', 1, []);
            imwrite(videoFrame,[ 'D:\Documentos\David\U.DISTRITAL\Semestres\Semestre10\Teleinformatica\Happiness_Recognition\photos\',str,'\',int2str(i), '.jpg']);
            % Muestra un cuadro delimitador alrededor de la cara que se está rastreando.
            videoFrame = insertShape(videoFrame, 'Polygon', bboxPolygon, 'LineWidth', 3);

            % Mostrar puntos de seguimiento.
            videoFrame = insertMarker(videoFrame, visiblePoints, '+', 'Color', 'white');
            
            % Restablecer los puntos.
            oldPoints = visiblePoints;
            setPoints(pointTracker, oldPoints);
            i = i+1;
        end

    end

    % Visualice el cuadro de video anotado utilizando el objeto reproductor de video.
    step(videoPlayer, videoFrame);

    % Compruebe si la ventana del reproductor de video se ha cerrado.
    runLoop = isOpen(videoPlayer);
end

% Limpiar.
clear cam;
release(videoPlayer);
release(pointTracker);
release(faceDetector);
