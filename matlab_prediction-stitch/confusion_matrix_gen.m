function [C_p] = bb_cmts()
    lab_root = 'dir_to_test_set_mat_files\*.mat';
    lab_list = dir(lab_root);
    p_cat = [];
    y_cat = [];    
    for i=1:floor(length(lab_list) / 2) - 1 %for each shape file
        p_file = strcat(lab_root(1:end-5), 'p_', int2str(i), '.mat');
        y_file = strcat(lab_root(1:end-5), 'y_', int2str(i), '.mat');
        p = load(p_file);
        y = load(y_file);
        p = cast(p.p,'uint8')';
        y = cast(y.y,'uint8')';        
        p_cat = cat(1,p_cat,p) ; %[p_cat, p];
        y_cat = cat(1,y_cat,y);  %[y_cat, y];
    end 
    %p_out = strcat(lab_root(1:end-5), 'p.mat');
    C_p = confusionmat(y_cat, p_cat);
    %save(p_out,'C_p')
end

