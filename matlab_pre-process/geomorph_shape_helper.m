function [class_divisions] = geomorph_shape_helper()
    
    lab_root = 'dir_to_label_root\*.mat';
    flist_root = 'dir_to_flist_label_root\';
    img_out_root = 'output_dir_for_images\';
    lab_out_root = 'output_dir_for_labels\';
    lab_list = dir(lab_root);
    
    IMG_DATASET_INX = 1;
    class_divisions = zeros(1, 10, 'double');
    
    for i=1:length(lab_list) %for each shape file
        
        curr_f = lab_list(i).name; 
        
        disp(curr_f)
        
        class_divisions(i) = IMG_DATASET_INX;
        
        %LABEL SEMANTIC VALUES
        if strcmp(curr_f, 'Bedform_residual.mat')            
            D = 0;                        
        elseif strcmp(curr_f, 'Bedrock_corrugated.mat')
            D = 1;                          
        elseif strcmp(curr_f, 'Bedrock_fractured.mat')            
            D = 2;               
        elseif strcmp(curr_f, 'Bedrock_lineated.mat')
            D = 3;    
        elseif strcmp(curr_f, 'Corrugated.mat')
            D = 4;            
        elseif strcmp(curr_f, 'Depressions.mat')
            D = 5;           
        elseif strcmp(curr_f, 'Featureless.mat')
            D = 6;            
        elseif strcmp(curr_f, 'Incisions.mat')
            D = 7;            
        elseif strcmp(curr_f, 'Large_duneform.mat')
            D = 8;            
        elseif strcmp(curr_f, 'Small_duneform.mat')
            D = 9;            
        end

        %LOAD LABEL MASK 
        res = load(strcat(lab_root(1:end-5), curr_f));
        f = load(strcat(flist_root, curr_f(1:end-4), '_files.mat'));
        B = f.file_list;
        C = cellfun(@isempty,B);
        B(C) = [];
        A = unique(B);        
        
        for j=1:length(A)
            disp(A(j))
            inds = find(contains(B,A(j)));
            img = imread(char(A(j)));      
            
            %MERGE LABELS TO A SINGLE IMAGE
            composite_lab = false(3000, 3000);
            for k=1:length(inds)
                composite_lab = composite_lab | squeeze(res.res(inds(k), :, :));
            end  
            
            for v = 92:256:2907
                for l = 92:256:2907 
                    
                    %CHECK IF LABEL OVERLAYS ON CURRENT IMAGE BLOCK
                    if (sum(sum(composite_lab(v:v+255, l:l+255)))) > 0 
                        
                        %OUTPUT FILES
                        img_out_file = strcat(img_out_root, ...
                                    'img_', int2str(IMG_DATASET_INX), ... 
                                    '.mat');
                        label_out_file = strcat(lab_out_root, ...
                                    'lab_', int2str(IMG_DATASET_INX), ...  
                                    '.mat');
                        
                        %IMAGE BLOCK TO BE SAVED     
                        t_img = img(v:v+255, l:l+255, :);                                                                                                
                        l_img = zeros(256, 256);
                        
                        %NON-LABELLED PIXELS FOR SEMI-SUPERVISION
                        l_img(:,:) = -1;
                        
                        % INDEX ON MASK
                        l_img(composite_lab(v:v+255, l:l+255)) = D;                         
                        t_img = im2double(t_img);  
                        
                        save(img_out_file, 't_img');                
                        save(label_out_file, 'l_img')
                        IMG_DATASET_INX = IMG_DATASET_INX + 1;
                        
                    end
                end
            end
        end
    end
    
end