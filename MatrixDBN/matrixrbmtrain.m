function matrixrbm = matrixrbmtrain(matrixrbm, x, opts)
    assert(isfloat(x), 'x must be a float');
    assert(all(x(:)>=0) && all(x(:)<=1), 'all data in x must be in [0:1]');
    m = size(x, 1);
    numbatches = m / opts.batchsize;
    
    assert(rem(numbatches, 1) == 0, 'numbatches not integer');
    [nh,nv]=size(rbm.W);
    
    for i = 1 : opts.numepochs
        kk = randperm(m);
        err = 0;
        %ll = 0;
        %scoreTAP=0;
        %recon_error=0;
        for l = 1 : opts.numbatches
            batch = x(kk((l - 1) * opts.batchsize + 1 : l * opts.batchsize), : , :);
            
            for j=1:batchsize
                v_pos = batch(j);
                h_pos=rbm.U * v_pos * rbm.V' + rbm.C;
                
                v_neg=v_pos;
                h_neg=h_pos;
                for k = 1:(opts.iterations-1)
                    v_neg=rbm.U' * h_neg * rbm.V + rbm.B;
                    h_neg=rbm.U * v_neg * rbm.V' + rbm.C;
                end
                rbm.vU=rbm.vU+sigm(rbm.U * v_neg * rbm.V' + rbm.C) * V * v_neg' - sigm(U * v_pos * V' + rbm.C) * V * v_pos;
                rbm.vV=rbm.vV+sigm(rbm.U * v_neg * rbm.V' + rbm.C)' * U * v_neg - sigm(rbm.U * v_pos * rbm.V' + rbm.C)' * U * v_pos;
                rbm.vB=rbm.vB+v_neg-v_pos;
                rbm.vC=rbm.vC+sigm(rbm.U * v_neg * rbm.V' + rbm.C) - sigm(rbm.U * v_pos * rbm.V' + rbm.C);
            end
            rbm.vU=rbm.vU * opts.momentum+ opts.alpha * rbm.vU/opts.batchsize;
            rbm.vV=rbm.vV * opts.momentum+ opts.alpha * rbm.vV/opts.batchsize;
            rbm.vB=rbm.vB * opts.momentum+ opts.alpha * rbm.vB/opts.batchsize;
            rbm.vC=rbm.vC * opts.momentum+ opts.alpha * rbm.vC/opts.batchsize;
            
            rbm.U = rbm.U + rbm.vU;
            rbm.V = rbm.V + rbm.vV;
            rbm.B = rbm.B + rbm.vB;
            rbm.C = rbm.C + rbm.vC;
            
            err = err + sum(sum((v_pos - v_neg) .^ 2)) / opts.batchsize;
        end
        disp(['epoch ' num2str(i) '/' num2str(opts.numepochs)  '. Average reconstruction error is: ' num2str(err / numbatches)]);
        
     end

end