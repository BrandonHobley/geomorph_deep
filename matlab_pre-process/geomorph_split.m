function [] = geomorph_split(c_img, EASTING, NORTHING)


    % REAL WORLD PIXEL TRANSLATION AND RESOLUTION
    x_trans = 25.000000000;
    y_trans = -25.000000000;
    pix_res = 25.00;

    % OUTPUT DIRECTORY FOR TILED MOSAIC
    out_tiles = 'dir_to_output_tiles';
    
    [X, Y, ~] = size(c_img);
    
    % CURRENTLY SET FOR 3000x3000 tiles
    x_count = 0;
    for j=1:3000:X
        y_count = 0;
        for k=1:3000:Y
            
            out_file = strcat(out_tiles, 'gm_', int2str(x_count), ...
                '-', int2str(y_count));
            
            s_x = j; s_y = k;            
            e_x = j + 3000 - 1; e_y = k + 3000 - 1;
            temp_img = c_img(s_x:e_x, s_y:e_y, :);
            
            disp('coordinate correction... saving...')
            x_tfw = (k-1) * pix_res + EASTING;
            y_tfw = NORTHING - ((j-1) * pix_res);

            % WORLD FILE STRING
            out_tfw = strcat(out_file, '.tfw');
            out_tif = strcat(out_file, '.tif');

            % WORLD FILE GENERATION
            fileID = fopen(out_tfw,'w');
            fprintf(fileID,'%.9f\n%.9f\n%.9f\n%.9f\n%.9f\n%.9f\n', ...
                [x_trans,0.0,0.0,y_trans,x_tfw,y_tfw]);
            fclose(fileID);
            
            % WORLD FILE TO CREATE .TIF
            R_out = worldfileread(out_tfw, 'planar', [3000, 3000]);
            coord_sys = 32629;
            geotiffwrite(out_tif, temp_img, R_out, 'CoordRefSysCode', coord_sys);

            disp('saved.')

            y_count = y_count + 1;
        end
        x_count = x_count + 1;
    end  
end


  
