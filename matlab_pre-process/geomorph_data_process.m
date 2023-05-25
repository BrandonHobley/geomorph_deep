function [] = geomorph_data_process()
    img_root = 'dir_to_raster_root';    
    img_list = dir(strcat(img_root, '*.tif')); 
    
    % FETCH WORLD FILE COORDINATES AND PAD BY REAL WORLD RESOLUTION
    EASTING = 312687.5000000006 + (645 * (-25));
    NORTHING = 6246262.5000002142 + (5992 * 25);    
    
    % PAD IMAGE TO BE CONCATENATED
    cat_img = zeros(42000, 18000, length(img_list), 'uint8');         
    for i=1:length(img_list) %for each image;        
        img_file = strcat(img_root, img_list(i).name)
        img = imread(img_file);       
        
        % DOES NOT ACCOUNT FOR NO_DATA VALUES
        img = im2uint8(rescale(img)); 
        
        % PAD AND CONCATENATE
        img = padarray(img,[5992 645],0,'pre'); 
        cat_img(:, :, k) = img;         
    end          
    geomorph_split(cat_img, EASTING, NORTHING)
end
