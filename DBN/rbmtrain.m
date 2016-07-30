function rbm = rbmtrain(rbm, x, opts)
    assert(isfloat(x), 'x must be a float');
    assert(all(x(:)>=0) && all(x(:)<=1), 'all data in x must be in [0:1]');
    m = size(x, 1);
    numbatches = m / opts.batchsize;
    
    assert(rem(numbatches, 1) == 0, 'numbatches not integer');
    [nh,nv]=size(rbm.W);
    
    for i = 1 : opts.numepochs
        kk = randperm(m);
        err = 0;
        ll = 0;
        scoreTAP=0;
        recon_error=0;
        for l = 1 : numbatches
            batch = x(kk((l - 1) * opts.batchsize + 1 : l * opts.batchsize), :);
            
            %if opts.approx == 'CD'
            v_pos = batch;
            h_sample = sigmrnd(repmat(rbm.c', opts.batchsize, 1) + v_pos * rbm.W');%for MCMC Sampling
            
            
            h_pos=sigm(repmat(rbm.c', opts.batchsize, 1) + v_pos * rbm.W');%for EMFSampling
            
            v_neg=v_pos;
            h_neg = h_sample;
            if strcmp(opts.approx,'CD')
                %MCMC
                for k=1:1:opts.iterations
                     v_neg = sigmrnd(repmat(rbm.b', opts.batchsize, 1) + h_neg* rbm.W);
                     h_neg = sigm(repmat(rbm.c', opts.batchsize, 1) + v_neg * rbm.W');
                end          
            elseif strcmp(opts.approx,'tap2') ||  strcmp(opts.approx,'tap3') ||  strcmp(opts.approx,'naive') ||  strcmp(opts.approx,'semi')
                     vis_init = v_pos;
                     hid_init = h_pos;
                     [v_neg, h_neg] = equilibrate(rbm,opts,vis_init,hid_init);
            end
            
%             disp('size v1')
%             size(v1)
%             disp('size h1')
%             size(h1)
%             disp('size v2')
%             size(v2)
%             disp('size W')
%             size(rbm.W)
%             disp('size b')
%             size(rbm.b)
%             disp('size c')
%             size(rbm.c)
            calculate_weight_gradient(rbm,opts,v_pos,h_sample,v_neg,h_neg);
            calculate_weight_gradient_c(rbm,opts,v_pos,h_sample,v_neg,h_neg);
%             c1 = h_sample' * v_pos;%if use h_pos instead of h_sample , the reconstruction error will swill
%             c2 = h_neg' * v_neg;
%             rbm.vW = c1-c2;
            %rbm.vW = rbm.momentum * rbm.vW + rbm.alpha * vW     / opts.batchsize;
            rbm.vb = rbm.momentum * rbm.vb + rbm.alpha * sum(v_pos - v_neg)' / opts.batchsize;
            rbm.vc = rbm.momentum * rbm.vc + rbm.alpha * sum(h_pos - h_neg)' / opts.batchsize;
%             rbm.vb = rbm.momentum * rbm.vb + rbm.alpha * sum(v1 - v2)' / opts.batchsize;
%             rbm.vc = rbm.momentum * rbm.vc + rbm.alpha * sum(h1 - h2)' / opts.batchsize;
%             if strcmp(opts.approx,'tap2') || strcmp(opts.approx,'tap3')
%                     %disp('tap2')
%                     buf2 = (h_neg-abs(h_neg).^2)' * (v_neg-abs(v_neg).^2) .* rbm.W; %100*784
%                     rbm.vW = rbm.vW - rbm.alpha * buf2;
%             end
%             if strcmp(opts.approx,'tap3')
%                     %disp('tap3')
%                     buf3 = (h_neg-abs(h_neg).^2) .* (0.5-h_neg)' * (v_neg - abs(v_neg).^2) .* rbm.W2;%100*784
%                     rbm.vW = rbm.vW - 2.0* rbm.alpha * buf3;
%             end
            
            if strcmp(opts.weight_decay,'l1')
                   rbm.vW = rbm.vW - rbm.alpha * opts.regularize * rbm.W;
            end
            if strcmp(opts.weight_decay,'l2')
                   rbm.vW = rbm.vW - rbm.alpha * opts.regularize * sign(rbm.W);
            end
            
%             rbm.vW = rbm.vW_prev * rbm.momentum + rbm.vW;
%             rbm.vW_prev = rbm.vW;
 
            rbm.W = rbm.W + rbm.vW;
            
            if strcmp(opts.approx,'semi')
                rbm.C = rbm.C + rbm.vC;
                rbm.C2 = rbm.C.^2;
            end
            if strcmp(opts.approx,'tap2') || strcmp(opts.approx,'tap3')
                rbm.W2 = rbm.W.^2;
            end
            if strcmp(opts.approx,'tap3')
                 rbm.W3 = rbm.W.^3;
            end
           
            
            
            rbm.b = rbm.b + rbm.vb;
            rbm.c = rbm.c + rbm.vc;
            err = err + sum(sum((v_pos - v_neg) .^ 2)) / opts.batchsize;
            ll = ll + mean(persudoLL(rbm,v_pos))/(nh+nv);
            scoreTAP=scoreTAP+mean(score_samples_TAP(rbm,opts,v_pos))/(nh+nv);
            recon_error=reconstruction_error(rbm,v_pos)/(nh+nv);
        end
        
        disp(['epoch ' num2str(i) '/' num2str(opts.numepochs)  '. Average reconstruction error is: ' num2str(err / numbatches)]);
        disp(['epoch ' num2str(i) '/' num2str(opts.numepochs)  '. Persudo Likelihoodis: ' num2str(ll / numbatches)]);
        disp(['epoch ' num2str(i) '/' num2str(opts.numepochs)  '. TAP Score: ' num2str(scoreTAP / numbatches)]);
        disp(['epoch ' num2str(i) '/' num2str(opts.numepochs)  '. Recon Error: ' num2str(recon_error / numbatches)]);
    end
end

function rbm=calculate_weight_gradient(rbm,opts,v_pos,h_pos,v_neg,h_neg)
      c1 = h_pos' * v_pos;%if use h_pos instead of h_sample , the reconstruction error will swill
      c2 = h_neg' * v_neg;
      rbm.vW = c1-c2;
      if strcmp(opts.approx,'tap2') || strcmp(opts.approx,'tap3')
           %disp('tap2')
            buf2 = (h_neg-abs(h_neg).^2)' * (v_neg-abs(v_neg).^2) .* rbm.W; %100*784
            rbm.vW = rbm.vW - rbm.alpha * buf2;
      end
      if strcmp(opts.approx,'tap3')
            %disp('tap3')
            buf3 = (h_neg-abs(h_neg).^2) .* (0.5-h_neg)' * (v_neg - abs(v_neg).^2) .* rbm.W2;%100*784
            rbm.vW = rbm.vW - 2.0* rbm.alpha * buf3;
      end
      rbm.vW = rbm.vW_prev * opts.momentum + rbm.vW;
      rbm.vW_prev = rbm.vW;
end

function rbm=calculate_weight_gradient_c(rbm,opts,v_pos,h_pos,v_neg,h_neg)
    rbm.vC = opts.alpha *( (v_pos' * v_pos) - (v_neg' * v_neg) );
    rbm.vC = rbm.vC + rbm.vC_prev * opts.momentum;
    rbm.vC_prev = rbm.vC;
end

function energy = free_energy(rbm,vis)
    %size(vis(1,:))
    %size(rbm.b)%784*1
    for i=1:size(vis,1)
        vis_bias(i,:) = vis(i,:) .* rbm.b';
    end
    
    vb = sum(vis_bias,2);%按行求和
    %size(vb):batchsize * 1
    hid_bias = vis * rbm.W' ;
    for i=1:size(hid_bias,1)
        hid_bias(i,:) = hid_bias(i,:) + rbm.c';
    end
    Wx_b_log = sum(log(1+exp(hid_bias)),2);%按行求和;
    %size(Wx_b_log):batchsize * 1
    energy=-vb-Wx_b_log;
end

function LL=persudoLL(rbm,vis)
    [n_samples,n_feat] = size(vis);
    vis_corrupted=vis;
    %idxs=round(rand(1,n_samples)*n_feat)+1;
    %idxs=randint(1,n_samples,[1 n_feat]);
    idxs=randint_s(1,n_samples,1,n_feat);
    for i=1:n_samples
        %for i =1:idxs(j)
        j=idxs(i);
        vis_corrupted(i,j) = 1 - vis_corrupted(i,j);
    end
    fe=free_energy(rbm,vis);
    fe_corrupted=free_energy(rbm,vis_corrupted);
    LL=n_feat * logsigm(fe_corrupted-fe);
end

function X = logsigm(P)
    X = log(1./(1+exp(-P)));
end
    
function MCMC(rbm, vis_init,StartMode)

end

function mse = reconstruction_error(rbm,vis)
    hid=sigmrnd(repmat(rbm.c', size(vis,1), 1) + vis * rbm.W');
    vis_rec = sigmrnd(repmat(rbm.b', size(vis,1), 1) + hid* rbm.W);
    dif = vis_rec - vis;
    mse = mean(mean(dif .* dif));
end

function score = score_samples_TAP(rbm,opts,vis)
    [v_pos,h_pos,m_vis,m_hid] = iter_mag(rbm,opts,vis);
    eps=1e-6;
    m_vis = max(m_vis, eps);
    m_vis = min(m_vis, 1.0-eps);
    m_hid = max(m_hid, eps);
    m_hid = min(m_hid, 1.0-eps);

    m_vis2 = m_vis.^2;
    m_hid2 = m_hid.^2;
    
    S= sum(m_vis .* log(m_vis) + (1.0 - m_vis) .* log(1.0 - m_vis),2) - sum(m_hid .* log(m_hid) + (1.0 - m_hid) .* log(1.0 - m_hid),2);
    U_naive= - m_vis * rbm.b - m_hid * rbm.c - sum((m_vis * rbm.W') .* m_hid ,2);
    Onsager= - 0.5 * sum(((m_vis-m_vis2) * rbm.W2') .* (m_hid - m_hid2),2);
    fe_tap=U_naive + Onsager - S;
    fe= free_energy(rbm,vis);
    score=fe_tap-fe;
end

function [v_pos, h_pos, m_vis, m_hid]=iter_mag(rbm,opts,vis)
    v_pos = vis;
    h_pos = sigmrnd(repmat(rbm.c', size(vis,1), 1) + v_pos * rbm.W');
    
    if strcmp(opts.approx,'naive')
        mag_vis = @mag_vis_naive;
        mag_hid = @mag_hid_naive;
    elseif strcmp(opts.approx,'tap3')
        mag_vis = @mag_vis_tap3;
        mag_hid = @mag_hid_tap3;
    elseif strcmp(opts.approx,'xzz')
        mag_vis = @mag_vis_xzz;
        mag_hid = @mag_hid_xzz;
    else
        mag_vis = @mag_vis_tap2;
        mag_hid = @mag_hid_tap2;
    end
    m_vis = 0.5 * mag_vis(rbm, opts,vis, h_pos) + 0.5 * vis;
    m_hid = 0.5 * mag_hid(rbm, opts,m_vis, h_pos) + 0.5 * h_pos;
    for i = 1:1:opts.iterations-1
        m_vis = 0.5 * mag_vis(rbm, opts,m_vis, m_hid) + 0.5 * m_vis;
        m_hid = 0.5 * mag_hid(rbm, opts,m_vis, m_hid) + 0.5 * m_hid;
    end
end

function result=generate(rbm,opts,vis_init)
    Nsamples = size(vis_init,2);
    Nhid= size(rbm.c,1);
    h_init  = zeros(Nsamples,Nhid);
    %if approx=="naive" || contains(approx,"tap")
    if strcmp(opts.approx,'tap2') ||  strcmp(opts.approx,'tap3') ||  strcmp(opts.approx,'naive') ||  strcmp(opts.approx,'xzz')
        vis_mage, hid_mag = equilibrate(rbm,vis_init,hid_init);
    end
    SamplingIterations=3
    if strcmp(approx,'CD')
        vis_mag, hid_mag, vis_sample, hid_sample = MCMC(rbm, vis_init,'visible');
    end

    samples,vis_neg = sample_visibles(rbm,hid_mag);
    result=reshape(samples);
end

function semi=mag_vis_semi(rbm,opts,m_vis,m_hid)
    [rows,cols] = size(m_vis);
    m_vis_buf1 = zeros(size(m_vis));
    m_vis_buf2 = zeros(size(m_vis));
    C_tmp = rbm.C;
    C_tmp2 = rbm.C2;
    for i=1:rows
        C_tmp(i,i)=0;
        C_tmp2(i,i)=0;
    end
    
    m_vis_buf1 = m_vis * C_tmp;
    m_vis_buf2 = ( (m_vis - m_vis.^2) * C_tmp2 ) .* (0.5 - m_vis);
    
    buf = repmat(rbm.b', size(m_hid,1), 1) + m_hid * rbm.W ;
    second_order = ( (m_hid - m_hid.^2 ) * rbm.W2 ).* (0.5 - m_vis);
    
    buf = buf + second_order;
    buf = buf + m_vis_buf1;
    buf = buf + m_vis_buf2;
    semi = sigm(buf);
end

function semi=mag_hid_semi(rbm,opts,m_vis,m_hid)
    buf = repmat(rbm.c', size(m_vis,1), 1) + m_vis * rbm.W';
    second_order = ( (m_vis - m_vis.^2) * rbm.W2'  ) .* (0.5 - m_hid);
    buf = buf + second_order;
    semi = sigm(buf);
end


function [m_vis,m_hid] = equilibrate(rbm,opts,m_vis,m_hid)
    if strcmp(opts.approx,'naive')
        mag_vis = @mag_vis_naive;
        mag_hid = @mag_hid_naive;
    elseif strcmp(opts.approx,'tap3')
        mag_vis = @mag_vis_tap3;
        mag_hid = @mag_hid_tap3;
    elseif strcmp(opts.approx,'semi')
        mag_vis = @mag_vis_semi;
        mag_hid = @mag_hid_semi;
    else
        mag_vis = @mag_vis_tap2;
        mag_hid = @mag_hid_tap2;
    end
    for i = 1:1:opts.iterations
        m_vis = 0.5 * mag_vis(rbm, opts,m_vis, m_hid) + 0.5 * m_vis;
        m_hid = 0.5 * mag_hid(rbm, opts,m_vis, m_hid) + 0.5 * m_hid;
    end
end

%function naive = mag_vis_naive(rbm,opts,m_vhid)


function tap3 = mag_vis_tap3(rbm,opts,m_vis,m_hid)
      buf=repmat(rbm.b', size(m_hid,1), 1) + m_hid * rbm.W;
      second_order=((m_hid - abs(m_hid).^2) * rbm.W2) .* (0.5 - m_vis);
      %size(m_hid)
      %size(rbm.W3)
      third_order= (    (  abs(m_hid).^2   .*  (1 - m_hid)   ) * rbm.W3 )   .* (1/3 - 2 * (m_vis - abs(m_vis).^2 ));
      buf = buf + second_order+third_order;
      tap3 = sigm(buf);
end

function tap3 = mag_hid_tap3(rbm,opts,m_vis,m_hid)
      buf=repmat(rbm.c', size(m_vis,1), 1) + m_vis * rbm.W';
      second_order=((m_vis - abs(m_vis).^2) * rbm.W2') .* (0.5 - m_hid);
      third_order=( ( abs(m_vis).^2 .* (1 - m_vis) )  * rbm.W3')  .* (1/3 - 2* (m_hid - abs(m_hid).^2 ));
      buf = buf + second_order + third_order;
       tap3 = sigm(buf);
end

function tap2 = mag_vis_tap2(rbm,opts,m_vis,m_hid)
      %sigmrnd(sigmrnd(h1 * rbm.W);+ h1 * rbm.W);
      buf=repmat(rbm.b', size(m_hid,1), 1) + m_hid * rbm.W;
      second_order=((m_hid - abs(m_hid).^2) * rbm.W2) .* (0.5 - m_vis);
      buf = buf + second_order;
      tap2 = sigm(buf);
      %return sigm(buf)
end

function tap2 = mag_hid_tap2(rbm,opts,m_vis,m_hid)
       buf=repmat(rbm.c', size(m_vis,1), 1) + m_vis * rbm.W';
       second_order=((m_vis - abs(m_vis).^2) * rbm.W2') .* (0.5 - m_hid);
      buf = buf + second_order;
       tap2 = sigm(buf);
      %return sigm(buf)
end