function [] = geomorph_shape_process()

    shape_root = 'dir_to_labels_shape_root';
    img_root = 'dir_to_tiled_tifs';
    shape_list = dir(strcat(shape_root, '*.shp')); 
    img_list = dir(strcat(img_root, '*.tif')); 
    
    % SETUP A FOLDER FOR OUTPUTS - INTERMEDIATE FOLDER OUTPUT
    out_root = 'dir_for_output_files';
    
    for i=1:length(shape_list) %for each shape file
       
       s = shaperead(strcat(shape_root, shape_list(i).name));
       shape_list(i).name
       
       SHAPE_INDEX = 1;
              
       res = false(length(s), 3000, 3000);
       file_list = cell(length(s), 1);
       
       for j=1:length(s) %for each shape                                  
             
            disp(SHAPE_INDEX)
            SHAPE_INDEX = SHAPE_INDEX + 1;

            s_Xs = s(j).X(1:end-1);
            s_Ys = s(j).Y(1:end-1);
            assert(length(s_Xs) == length(s_Ys))
            
            for k=1:length(img_list) 
                
                % READ WORLD FILE
                WF_rgb = getworldfilename(strcat(img_root, img_list(k).name));
                R = worldfileread(WF_rgb, 'planar', [3000, 3000]);
                
                % CONVERT REAL WORLD TO PIXEL COORDINATES                 
                [ix, iy] = worldToIntrinsic(R,s_Xs,s_Ys); 
                % CREATE MASK FROM COORDINATES
                s_mask = poly2mask(ix,iy,R.RasterSize(1),R.RasterSize(2));
                
                % CHECK IF OVERLAY
                maskcheck=sum(sum(s_mask));
                if maskcheck > 0   
                    %B = imoverlay(ms(:,:,1:3),s_mask,'yellow');
                    
                    % ADDED FILES AND MASK TO BE SAVED
                    file_list{j} = strcat(img_root, img_list(k).name);
                    res(j, :, :) = s_mask;
                end
            end
       end
       
       %OUT FILE STRING GENERATION 
       
       %EACH FOLDER CONTAINS A FILE LIST AND .MAT WITH MASKS TO BE
       %PROCESSED BY THE HELPER FUNCTION
       f = shape_list(i).name;       
       out_file_lab = strcat(out_root, 'labs\', f(1:end-4));
       out_file_flist = strcat(out_root, 'file_list\', f(1:end-4));
       save(strcat(out_file_lab, '.mat'), 'res', '-v7.3')
       save(strcat(out_file_flist, '_files.mat'), 'file_list')
    end
    
end
