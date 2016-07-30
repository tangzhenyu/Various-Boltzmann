function matrixdbn = matrixdbntrain(matrixdbn, x, opts)
    n = numel(matrixdbn.matrixrbm);

    matrixdbn.matrixrbm{1} = matrixrbmtrain(matrixdbn.matrixrbm{1}, x, opts);
    for i = 2 : n
        x = matrixrbmup(matrixdbn.matrixrbm{i - 1}, x);
        matrixdbn.matrixrbm{i} = matrixrbmtrain(dbn.matrixrbm{i}, x, opts);
    end
end