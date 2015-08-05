function getRNNinput(DATASET)

INPUT_FILE = 'input.txt';

switch DATASET
    
    case {'kitti', 'KITTI', 'Kitti'}
        
        addpath('../')  %path to devkit
        write_dir = '/home/shawn/gtri/tracking-prediction/rnn-tracking/data/kitti/';  %input dir for torch rnn-tracking
        label_dir = '/media/shawn/Windows/data/kitti/data_tracking_image_2/training/label_02/';  %contain frame,id,box locations

        % additional parameters to add to output file to track the dataset and
        % sequence number
        DATASET = 'kitti';
        type = input('Enter object type: ');

        labels = dir(strcat(label_dir,'*.txt'));
        N = length(labels);
        C = cell(1,20);
        nimages = 0;

        for seq_idx = 0:N-1

            % Read Labels
            % parse input file
            labelfile = fullfile(label_dir, sprintf('%04d.txt', seq_idx));
            % count columns
            fid = fopen(labelfile);
            l=strtrim(fgetl(fid));
            ncols = numel(strfind(l,' '))+1;
            fclose(fid);

            fid = fopen(labelfile);
            try
              if ncols == 17 % ground truth file
                A = textscan(fid, '%d %d %s %d %d %f %f %f %f %f %f %f %f %f %f %f %f');
              elseif ncols==18
                A = textscan(fid, '%d %d %s %d %d %f %f %f %f %f %f %f %f %f %f %f %f %f');
              else
                error('This file is not in KITTI tracking format.');
              end
            catch
              error('This file is not in KITTI tracking format.');
            end
            fclose(fid);

            num_imgs_A = max(A{1}) + 1;

            % Sort according to tracklet id and store in C
            [~, I] = sort(A{2});
            for col = 1:ncols
                    C{col+2} = [C{col+2}; A{col}(I)];
            end

            % add in sequence number entries
            sequence = zeros(num_imgs_A, 1);
            sequence(:) = seq_idx;
            C{2} = [C{2}; sequence];

            % update total number of images
            nimages = nimages + num_imgs_A;
        end

        % set dataset entries for C
        C{1} = cell(length(C{2}), 1);
        C{1}(:) = {DATASET};

        % define as objects
        % extract dataset name and sequence number
        objects.dataset    = C{1};
        objects.seq        = C{2};

        % extract label, truncation, occlusion
        objects.frame      = C{3}; % tracklet id
        objects.id         = C{4}; % tracklet id
        objects.type       = C{5};  % 'Car', 'Pedestrian', ...
        objects.truncation = C{6}; % truncated pixel ratio ([0..1])
        objects.occlusion  = C{7}; % 0 = visible, 1 = partly occluded, 2 = fully occluded, 3 = unknown
        objects.alpha      = C{8}; % object observation angle ([-pi..pi])

        % extract 2D bounding box in 0-based coordinates
        objects.x1 = C{9}; % left
        objects.y1 = C{10}; % top
        objects.x2 = C{11}; % right
        objects.y2 = C{12}; % bottom

        % extract 3D bounding box information
        objects.h    = C{13}; % box width
        objects.w    = C{14}; % box height
        objects.l    = C{15}; % box length
        objects.t{1} = C{16}; % location (x)
        objects.t{2} = C{17}; % location (y)
        objects.t{3} = C{18}; % location (z)
        objects.ry   = C{19}; % yaw angle
        if(~isempty(C{20}))
            objects.score   = C{20}; % score for tracker hypotheses
        end

        % parse input file
        %fid = fopen(sprintf('%s/input_%s.txt',write_dir,type),'w');
        fid = fopen(sprintf('%s%s',write_dir,INPUT_FILE),'w');

        for o = 1:nimages % also number of objects

            if isfield(objects,'type')
                %objectType = objects.type(o);
                % Do not write don't cares nor any objects of type other than
                % specified
                if strcmp(objects.type(o),type) %& objects.id ~= -1

                    % print dataset name and sequence number
                    if isfield(objects,'dataset'),      fprintf(fid,'%s ',objects.dataset{o});
                    else                                   error('ERROR: dataset not specified!'),end;
                    if isfield(objects,'seq'),          fprintf(fid,'%d ',objects.seq(o));
                    else                                   error('ERROR: seq not specified!'),end;
                    % set frame and tracking-id
                    %if isfield(objects(o),'frame'),        fprintf(fid,'%d ',f-1);
                    if isfield(objects,'frame'),        fprintf(fid,'%d ',objects.frame(o));
                    else                                   error('ERROR: frame not specified!'), end;
                    if isfield(objects,'id'),           fprintf(fid,'%d ',objects.id(o));
                    else                                   error('ERROR: frame not specified!'), end;
                    % set label, truncation, occlusion
                    %if isfield(objects,'type'),         
                    fprintf(fid,'%s ',objects.type{o});
                    %else                                   error('ERROR: type not specified!'), end;
                    if isfield(objects,'truncation'),   fprintf(fid,'%.2f ',objects.truncation(o));
                    else                                   fprintf(fid,'-1 '); end; % default
                    if isfield(objects,'occlusion'),    fprintf(fid,'%d ',objects.occlusion(o));
                    else                                   fprintf(fid,'-1 '); end; % default
                    if isfield(objects,'alpha'),        fprintf(fid,'%.2f ',wraptopi(objects.alpha(o)));
                    else                                   fprintf(fid,'-10 '); end; % default

                    % set 2D bounding box in 0-based C++ coordinates
                    if isfield(objects,'x1'),           fprintf(fid,'%.2f ',objects.x1(o));
                    else                                   error('ERROR: x1 not specified!'); end;
                    if isfield(objects,'y1'),           fprintf(fid,'%.2f ',objects.y1(o));
                    else                                   error('ERROR: y1 not specified!'); end;
                    if isfield(objects,'x2'),           fprintf(fid,'%.2f ',objects.x2(o));
                    else                                   error('ERROR: x2 not specified!'); end;
                    if isfield(objects,'y2'),           fprintf(fid,'%.2f ',objects.y2(o));
                    else                                   error('ERROR: y2 not specified!'); end;

                    % set 3D bounding box
                    if isfield(objects,'h'),            fprintf(fid,'%.2f ',objects.h(o));
                    else                                   fprintf(fid,'-1 '); end; % default
                    if isfield(objects,'w'),            fprintf(fid,'%.2f ',objects.w(o));
                    else                                   fprintf(fid,'-1 '); end; % default
                    if isfield(objects,'l'),            fprintf(fid,'%.2f ',objects.l(o));
                    else                                   fprintf(fid,'-1 '); end; % default
                    if isfield(objects,'t')            
                        fprintf(fid,'%.2f %.2f %.2f ',objects.t{1}(o),objects.t{2}(o),objects.t{3}(o));
                    else                                   fprintf(fid,'-1000 -1000 -1000 '); end; % default
                    if isfield(objects,'ry'),           fprintf(fid,'%.2f ',wraptopi(objects.ry(o)));
                    else                                   fprintf(fid,'-10 '); end; % default

                    % set score
                    % score is only saved for detections, not for ground truth
                    if isfield(objects,'score'),        fprintf(fid,'%.2f ',objects.score(o)); end


                    % next line
                    fprintf(fid,'\n');
                end

            else
                error('ERROR: type not specified!');
            end

        end

        % close file
        fclose(fid);
        
    case {'caltech', 'Caltech', 'CalTech', 'usa', 'USA'}
        
        addpath('/media/shawn/Windows/data/caltech-pedestrian/code3.2.1/');
        % setIds and vidIds indexed from 0
        [path, setIds, vidIds] = dbInfo('usa');
        write_dir = '/home/shawn/gtri/tracking-prediction/rnn-tracking/data/caltech-ped/';  %input dir for torch rnn-tracking
        
        DATASET = 'caltech';
        
        % opend input.txt to write to
        fid = fopen([write_dir, INPUT_FILE], 'w');
        
        % loop through each video sequence loading the vbb annotations and
        % exporting them to object structs.  Then object data is are 
        % written to rnn input txt file (1 line per frame)
        for s = 1:setIds(end)+1
            for v = 1:vidIds{s}(end)+1
                Seq = sprintf('set%02d/V%03d',setIds(s),vidIds{s}(v));
                A = vbb('vbbLoad', [path, '/annotations/', Seq]);
                % skip the sequence if it contains no pedestrians
                if A.objStr == -1
                    break;
                end
                % loop through each object id in the annotation to get each
                % object struct
                for id = 1:A.maxObj
                    % vbb('get', 'Ann', 'id', 'start frame', 'end frame')
                    obj = vbb('get', A, id, A.objStr(id), A.objEnd(id));
                    num_frames = obj.end - obj.str + 1;
                    fields = fieldnames(obj);
                    num_fields = length(fields);
                    
                    if strcmp(obj.lbl, 'person')
                        % loop through all frames obj appears in
                        for frame = 1:num_frames

                            % write to rnn input file
                            fprintf(fid, [DATASET ' ' Seq ' ']);
                            % print the frame number after the dataset and
                            % sequence
                            fprintf(fid, '%d ', frame-1+obj.str);

                            % loop through each field of obj and print to file
                            for f = 1:num_fields

                                switch fields{f}

                                    case {'id', 'str', 'end', 'hide'}
                                        fprintf(fid, '%d ', obj.(fields{f}));

                                    case 'lbl'
                                        fprintf(fid, '%s ', obj.(fields{f}));

                                    case {'pos', 'posv'}
                                        fprintf(fid, '%.2f ', obj.(fields{f})(frame,1));
                                        fprintf(fid, '%.2f ', obj.(fields{f})(frame,2));
                                        fprintf(fid, '%.2f ', obj.(fields{f})(frame,3));
                                        fprintf(fid, '%.2f ', obj.(fields{f})(frame,4));

                                    case {'occl', 'lock'}
                                        fprintf(fid, '%d ', obj.(fields{f})(frame));

                                end

                            end

                            fprintf(fid, '\n');

                        end
                    end
                end
            end
        end
        
        fclose(fid);
end