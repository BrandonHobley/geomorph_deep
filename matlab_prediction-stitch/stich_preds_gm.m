function [res] = stich_preds_gm()

    root = 'dir_to_tiled_predictions';
    out_root = 'output_dir_for_stiched_prediction';
    wf_root = 'dir_to_tiled_world_files';
    preds_list = dir(root);
    wf_list = dir(strcat(wf_root, '*.tfw'));
    L1 = length(preds_list);
    
    % COLOUR TO LABEL ASSIGNMENT
    map = [255, 102, 102; ... %red - Bedform residual
           255, 178, 102; ... %orange - Bedrock corrugated
           255, 255, 102; ... %yellow - Bedrock fractured
           178, 255, 102; ... %light green - Bedrock lineated
           0, 153, 0; ...     %dark green - Corrugated 
           153, 255, 204; ... %light green/blue - Depressions
           0, 153, 153; ...   %dark blue/green - Featureless
           0, 51, 102; ...    %dark blue - Incisions
           245, 66, 215; ...  %purple - Large duneforms
           128, 128, 128;];   %grey - Small duneforms
       
          
    map = cast(map, 'uint8');
    map = im2double(map);

    for img_idx = 3:L1-1 %first 2 files are '.' and '..' and last file is FINAL
        
        % FETCH PREDICTION AND WORLD FILE
        curr_dir = strcat(root, '\',  preds_list(img_idx).name) 
        file_num = regexp(curr_dir,'\d*','Match');        
        WF = strcat(wf_root, 'gm_', file_num{end-1}, '-', file_num{end}, '.tfw');   
        R = worldfileread(WF, 'planar', [3000, 3000]);
        f_list = dir(strcat(curr_dir, '\*.mat'));               
        L = length(f_list);
        
        res = zeros(3000, 3000, 3);    
        res = cast(res, 'uint8');
        
        for k = 1:L    
            
            % LOAD PREDICTIONS
            preds = load(strcat(curr_dir, '\', f_list(k).name));            
            preds = squeeze(preds.preds);           
            preds = preds + 1;
            % CONVER TO RGB
            preds = label2rgb(preds, map); 
            B = regexp(f_list(k).name,'\d*','Match');
            x_mod = mod(str2num(B{1}), 750);
            y_mod = mod(str2num(B{2}), 750);
            x_start = str2num(B{1}) + 1;
            y_start = str2num(B{2}) + 1;            
            %truncating at original image size w.r.t to tfw file
            res(x_start:x_start + 749, y_start:y_start + 749, :) = preds(76:825, 76:825, :);
        end
        
        %SAVE TILED .TIFs
        g_inds = res == 255;
        s_inds = g_inds(:, :, 3) == 1; 
        res(s_inds) = 0;
        coord_sys = 32629;
        out_file = strcat(out_root, file_num{end-1}, '_', ...
                            file_num{end}, '.tif')
        geotiffwrite(out_file, res, R, 'CoordRefSysCode', coord_sys);  
        
    end
        
end

