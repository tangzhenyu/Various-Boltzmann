function matrixdbn = matrixdbnsetup(matrixdbn, x, opts)
    n1 = size(x,2);
    n2 = size(x,3);
    matrixdbn.sizes = [n1,n2,matrixdbn.sizes];

    layers = numel(matrixdbn.sizes) / 2
    for u = 1 : layers - 1
        dbn.matrixrbm{u}.alpha    = opts.alpha;
        dbn.matrixrbm{u}.momentum = opts.momentum;
        dbn.matrixrbm{u}.I=matrixdbn.sizes((u - 1)*2+1);
        dbn.matrixrbm{u}.J=matrixdbn.sizes((u - 1)*2+2);
        dbn.matrixrbm{u}.K=matrixdbn.sizes(u*2+1);
        dbn.matrixrbm{u}.L=matrixdbn.sizes(u*2+2);
        %dbn.rbm{u}.W  = randn(dbn.sizes(u + 1), dbn.sizes(u));
        %dbn.rbm{u}.W  = zeros(dbn.sizes(u*2+1), dbn.sizes(u*2+2),dbn.sizes((u - 1)*2+1), dbn.sizes((u - 1)*2+2));
        
        %I * J * K * L
        dbn.matrixrbm{u}.W  = zeros(I,J,K,L);
%         disp('size W')
%         size(dbn.rbm{u}.W)
%        dbn.rbm{u}.W2 = dbn.rbm{u}.W.^2
%         disp('size W2')
%         size(dbn.rbm{u}.W2)
%        dbn.rbm{u}.W3 = dbn.rbm{u}.W.^3 
        
        %I * J * K * L
%         dbn.matrixrbm{u}.vW = zeros(I,J,K,L);
%         dbn.matrixrbm{u}.vW_prev = zeros(I,J,K,L);
    
        %U
        dbn.matrixrbm{u}.U  = rand(K,I);
        dbn.matrixrbm{u}.vU = zeros(K,I);
        
        %V
        dbn.matrixrbm{u}.V  = rand(L,J);
        dbn.matrixrbm{u}.vV = zeros(L,J);
        
        %visible bias
        dbn.matrixrbm{u}.B  = rand(I,J);
        dbn.matrixrbm{u}.vB = zeros(I,J);

        %hidden bias
        dbn.matrixrbm{u}.C  = rand(K,L);
        dbn.matrixrbm{u}.vC = zeros(K,L);
        
%         dbn.rbm{u}.b  = zeros(dbn.sizes((u - 1)*2+1), dbn.sizes((u - 1)*2+2));
%         dbn.rbm{u}.vb = zeros(dbn.sizes((u - 1)*2+1), dbn.sizes((u - 1)*2+2));
% 
%         dbn.rbm{u}.c  = zeros(dbn.sizes(u*2+1), dbn.sizes(u*2+2));
%         dbn.rbm{u}.vc = zeros(dbn.sizes(u*2+1), dbn.sizes(u*2+2));
    end

end
