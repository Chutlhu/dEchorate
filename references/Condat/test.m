function test()
N = 158;
L = 79;
P = L-1;
K = 5;

v = randn(N);
% T=toeplitz(v(P+1:N),v(P+1:-1:1));
% 
% 
% W=ones(N-P,P+1)*(P+1);	%matrix of weights for the weighted Frobenius norm
% for indexcol=2:P+1
%     for indexrow=1:P+2-indexcol
%         W(indexrow,indexcol-1+indexrow)=P+2-indexcol;
%         W(N-P-indexrow+1,P+3-indexcol-indexrow)=P+2-indexcol;
%         disp(N-P-indexrow+1)
%     end
% end
% 
% disp(W)
% size(W)
% size(T)


% P=M;				%the matrices have size N-P x P+1. K<=P<=M required.
Nbiter=10000;			%number of iterations.
mu=0.1;				%parameter. Must be in ]0,2[
gamma=0.51*mu;		%parameter. Must be in ]mu/2,1[
%note: setting mu=0 and gamma=0 yields the Douglas-Rachford method
Tnoisy=toeplitz(v(P+1:N),v(P+1:-1:1));
W=ones(N-P,P+1)*(P+1);	%matrix of weights for the weighted Frobenius norm
for indexcol=2:P+1
    for indexrow=1:P+2-indexcol
        W(indexrow,indexcol-1+indexrow)=P+2-indexcol;
        W(N-P-indexrow+1,P+3-indexcol-indexrow)=P+2-indexcol;
    end
end
Tdenoised=Tnoisy;		%the noisy matrix is the initial estimate
mats=Tdenoised;			%auxiliary matrix
for iter=1:Nbiter
    [U S V]=svd(mats+gamma*(Tdenoised-mats)+mu*(Tnoisy-Tdenoised)./W,0);
    Tdenoised=U(:,1:K)*S(1:K,1:K)*(V(:,1:K))';	%SVD truncation -> Tdenoised has rank K
    mats=mats-Tdenoised+Toeplitzation(2*Tdenoised-mats);
    disp(S(K,K)/S(K+1,K+1))
end
%at this point, Tdenoised has rank K but is not exactly Toeplitz
Tdenoised=Toeplitzation(Tdenoised);
%we reshape the Toeplitz matrix Tdenoised into a Toeplitz matrix with K+1 columns
Tdenoised=toeplitz([Tdenoised(1,P-K+1:-1:1).';Tdenoised(2:N-P,1)],Tdenoised(1,P-K+1:P+1));
end

function Matres=Toeplitzation(Mat)
%this function returns a Toeplitz matrix, closest to Mat for the Frobenius norm.
%this is done by simply averaging along the diagonals.
	[height,width]=size(Mat);  %height>=width required
	Matres=Mat;
	for indexcol=2:width
		valdiag=0;
		valdiag2=0;
		for indexrow=1:width-indexcol+1
			valdiag=valdiag+Mat(indexrow,indexcol-1+indexrow);
			valdiag2=valdiag2+Mat(height-indexrow+1,width-indexcol+2-indexrow);
		end
		valdiag=valdiag/(width-indexcol+1);
		valdiag2=valdiag2/(width-indexcol+1);
		for indexrow=1:width-indexcol+1
			Matres(indexrow,indexcol-1+indexrow)=valdiag;
			Matres(height-indexrow+1,width-indexcol+2-indexrow)=valdiag2;
		end
	end
	for indexcol=1:height-width+1
		valdiag=0;
		for indexrow=1:width
			valdiag=valdiag+Mat(indexcol+indexrow-1,indexrow);
		end
		valdiag=valdiag/width;
		for indexrow=1:width
			Matres(indexcol+indexrow-1,indexrow)=valdiag;
		end
	end
end		%of the function

