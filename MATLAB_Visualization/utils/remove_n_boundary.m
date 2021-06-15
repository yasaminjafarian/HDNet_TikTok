function mask = remove_n_boundary(mask,n)
    for i = 1 : n
        [row,col] = find(mask);
        k = boundary(col,row,1);
        for i = 1:length(k)
            kid = k(i);
            mask(row(kid),col(kid)) =  false;
        end
    end
end
