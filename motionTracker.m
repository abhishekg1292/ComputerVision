clc
clear all
close all

% Create a video reader.
reader = VideoReader('D:\Image Processing\HeadOnPed1.mp4');
videoPlayer = vision.VideoPlayer('Position', [20, 400, 700, 400]);

% Create a foreground detector
fgDetector = vision.ForegroundDetector();

% Create a blob detector
blobDetector = vision.BlobAnalysis('MinimumBlobArea', 1000);

objTrack = struct(...
    'id', {}, ...
    'kalmanFilter', {}, ...
    'bbox', {}, ...
    'age', {}, ...
    'totalVisible', {}, ...
    'totalInvisible', {});

nextID = 1;

writerObj = VideoWriter('D:\Image Processing\HeadOnPed1_motionTrack.mp4');
writerObj.FrameRate = 30;

% open the video writer
open(writerObj);

while hasFrame(reader)
    
    % read the frame
    frame = readFrame(reader);
    
    % detect the foreground
    mask = fgDetector.step(frame);
    
    % filtering operations
    mask = imopen(mask, strel('rectangle', [10 10]));
    mask = imclose(mask, strel('rectangle', [30 30]));
    mask = imfill(mask, 'holes');
    
    % get the blob information
    [area, centroid, bbox] = blobDetector.step(mask);
    
    numTracks = length(objTrack);
    numDetection = size(centroid, 1);
    
    for i=1:numTracks
        bboxTrack = objTrack(i).bbox;
        
        % predict the current location of the bbox
        predictCentroid = predict(objTrack(i).kalmanFilter);
        
        % get the upper left corner of the bounding box
        predictCentroid = int32(predictCentroid) - bboxTrack(3:4)/2;
        objTrack(i).bbox = [predictCentroid,  bboxTrack(3:4)];
    end
    
    % compute the cost matrix for assignment between original and new
    % detected tracks
    costMatrix = zeros(numTracks, numDetection);
    for i = 1:numTracks
        costMatrix(i, :) = distance(objTrack(i).kalmanFilter, centroid);
    end
    
    % solve the assignment problem
    costOfNonAssignment = 20;
    
    [assignments,unassignedTracks,unassignedDetections] = assignDetectionsToTracks(costMatrix,costOfNonAssignment);
    
    % Handle the assigned tracks
    for i = 1:size(assignments, 1)
        trackIndex = assignments(i, 1);
        detectionIndex = assignments(i, 2);
        
        detectCentroid = centroid(detectionIndex, :);
        detectBBOX = bbox(detectionIndex, :);
        
        % use the new detection to correct the kalman filter estimate
        correct(objTrack(trackIndex).kalmanFilter, detectCentroid);
        
        objTrack(trackIndex).bbox = detectBBOX;
        objTrack(trackIndex).age = objTrack(trackIndex).age + 1;
        objTrack(trackIndex).totalVisible = objTrack(trackIndex).totalVisible + 1;
        objTrack(trackIndex).totalInvisible = 0;        
    end
    
    % Handle the unassigned tracks
    for i = 1:length(unassignedTracks)
        trackIndex = unassignedTracks(i);
        objTrack(trackIndex).age = objTrack(trackIndex).age + 1;
        objTrack(trackIndex).totalInvisible = objTrack(trackIndex).totalInvisible + 1;
    end
    
    % Handle the unassigned detections. These could possibly lead to new
    % tracks
    for i = 1:length(unassignedDetections)
        detectionIndex = unassignedDetections(i);
        
        detectCentroid = centroid(detectionIndex, :);
        detectBBOX = bbox(detectionIndex, :);
        
        kFilter = configureKalmanFilter('ConstantVelocity', ...
            detectCentroid, [200, 50], [100, 25], 100);
        
        newObjTrack = struct(...
            'id', nextID, ...
            'kalmanFilter', kFilter, ...
            'bbox', detectBBOX, ...
            'age', 1, ...
            'totalVisible', 1, ...
            'totalInvisible', 0);
        
        % Add the track
        objTrack(end+1) = newObjTrack;
        
        % Increment the ID
        nextID = nextID + 1;
    end
    
    if ~isempty(objTrack)
        % Remove the obsolete tracks
        invisibleThresh = 20;
        visibileThresh = 0.4;
        
        % Get the invisible tracks indices
        invisibleInd = [objTrack(:).totalInvisible] >= invisibleThresh;
        
        % Get the visible indices where the tracks were visible for very
        % low ratio
        visibleInd = [objTrack(:).totalVisible] ./ [objTrack(:).age] < visibileThresh;
        
        obsoleteInd = invisibleInd | visibleInd;
        
        objTrack = objTrack(~obsoleteInd);
        
        % Get the indices of reliable tracks
        minVisibleCount = 10;
        
        goodTrackInd = [objTrack(:).totalVisible] > minVisibleCount;
        goodTracks = objTrack(goodTrackInd);
        
        % Convert the frame and the mask to uint8 RGB.
        frame = im2uint8(frame);
        
        if ~isempty(goodTracks)
            bbox = cat(1, goodTracks.bbox);
            ids = [goodTracks(:).id];
            
            labels = cellstr(int2str(ids'));
            predictedObjInds = [goodTracks(:).totalInvisible] > 0;
            
            predictedInd = cell(size(labels));
            predictedInd(predictedObjInds) = {' predicted'};
            labels = strcat(labels, predictedInd);
            
            % Draw the boxes on the frame
            frame = insertObjectAnnotation(frame, 'rectangle', bbox, labels);
        end
    end
    videoPlayer.step(frame);
    writeVideo(writerObj, frame);
end

% close the writer object
close(writerObj);
