function showPredictions(TEST_DATASET,TEST_SEQ)

addpath('../')

RNN_OUTPUT_PATH = '/home/shawn/gtri/tracking-prediction/rnn-tracking/output/';

switch TEST_DATASET
    
    case {'kitti', 'KITTI', 'Kitti'}
        
        TRACKING_PATH = '/media/shawn/Windows/data/kitti/data_tracking_image_2/training/image_02/';
        IMAGE_PATH = fullfile(TRACKING_PATH, sprintf('%04d/', TEST_SEQ));
        PREDICTION_FILE = fullfile(RNN_OUTPUT_PATH, 'predictions.txt');
        
        % read and display ground truth and prediction on each image
        tracklets = readLabels(RNN_OUTPUT_PATH, TEST_SEQ);
        
        fid = fopen(PREDICTION_FILE);
        frames_ahead = textscan(fid, '%d', 1);
        predictions = textscan(fid, '%f %f %f %f');
        
        for f = 1:length(tracklets)
            if(~isempty(tracklets{f}))
                image = imread(fullfile(IMAGE_PATH, sprintf('%06d.png', tracklets{f}.frame)));
                imshow(image)
                title(sprintf('frame %d', tracklets{f}.frame));
                
                pos_GT = [tracklets{f}.x1, tracklets{f}.y1, ...
                    tracklets{f}.x2 - tracklets{f}.x1 + 1, ...
                    tracklets{f}.y2 - tracklets{f}.y1 + 1];
                pos_PRED = [predictions{1}(f), predictions{2}(f), ...
                    predictions{3}(f), predictions{4}(f)];
                
                rectangle('Position', pos_GT, 'EdgeColor', 'r', 'LineWidth', 3);
                rectangle('Position', pos_PRED, 'EdgeColor', 'b', 'LineWidth', 3);
                pause(2);
                close Figure 1;
            end
        end
        
end