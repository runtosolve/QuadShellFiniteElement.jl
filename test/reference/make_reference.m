% Generate reference element matrices with S. Adany's Sept 2026 MATLAB code (to_Cris_Sept15_2026)
% for cross-checking QuadShellFiniteElement.jl. Runs unmodified in GNU Octave or MATLAB:
%   SANDOR=/path/to/to_Cris_Sept15_2026 OUTDIR=test/reference/octave_q42 octave-cli make_reference.m
addpath(getenv("SANDOR"));
outdir = getenv('OUTDIR');
E=200000; nu=0.30; G=E/(2*(1+nu)); t=1.5; mat_sym=1;

% distorted quad, tilted in 3D, slightly warped (node 3 lifted)
nodes=[ 0.0  0.0  0.0  0 1;
       12.0  1.0  3.0  0 2;
       11.0 10.0  4.0  0 3;
       -1.0  9.0  1.5  0 4];
% rotate the whole thing about a skew axis so nothing is aligned with global axes
ax=[1 2 3]/norm([1 2 3]); ang=0.7;
Kx=[0 -ax(3) ax(2); ax(3) 0 -ax(1); -ax(2) ax(1) 0];
R=eye(3)+sin(ang)*Kx+(1-cos(ang))*Kx^2;
nodes(:,1:3)=(R*nodes(:,1:3)')' + [5 -3 2];
elems_row=[1 2 3 4];

[T,P1e,P2e,P3e,P4e] = ct_4node_g2e(elems_row,nodes);
Pe=[P1e';P2e';P3e';P4e'];

ke_uv = ke_uv_4n_condens_from_12to8dof_num(P1e,P2e,P3e,P4e,t,E,E,nu,nu,G,mat_sym,4);
ke_wt = ke_wt_4n_condens_from_18to12dof_num(P1e,P2e,P3e,P4e,t,E,E,nu,nu,G,mat_sym,9);
ke_uv_nc = ke_uv_4n_nocondens_8dof_num(P1e,P2e,P3e,P4e,t,E,E,nu,nu,G,mat_sym,4);

% assemble local 20x20, add drilling (24x24), rotate to global (stiffmat_e lines 141-159)
elemnodenr=4; dpn=6;
induv=[]; indwt=[]; n0=0;
for i=1:elemnodenr
    induv=[induv n0+[1 2]];
    indwt=[indwt n0+[3 4 5]];
    n0=n0+5;
end
ke=zeros(elemnodenr*5);
ke(induv,induv)=ke_uv;
ke(indwt,indwt)=ke_wt;
ke24_local=add_drill(ke,elemnodenr,true);
[TT]=rotate3d(T,elemnodenr,dpn);
ke24_global=TT*ke24_local*TT';

save_mat = @(A, name) dlmwrite(fullfile(outdir, name), A, "delimiter", " ", "precision", "%.17g");
save_mat(nodes(:,1:3), 'nodes_global.txt');
save_mat(T, 'T.txt');
save_mat(Pe, 'nodes_local.txt');
save_mat(ke_uv, 'ke_uv_condensed.txt');
save_mat(ke_uv_nc, 'ke_uv_nocondens.txt');
save_mat(ke_wt, 'ke_wt_condensed.txt');
save_mat(ke24_local, 'ke24_local.txt');
save_mat(ke24_global, 'ke24_global.txt');
save_mat([E nu t], 'material.txt');
disp('reference written');
