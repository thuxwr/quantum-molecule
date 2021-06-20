/*
	 Solve hydrogen molecule, and try parallel calculation.

	 Weiran, June 20, 2021.
*/

static char help[] = "Solve hydrogen atom using multiple cores.";

#include <slepceps.h>


/* Pre-calculated data. */
typedef struct {
	Vec *T, *U, *V;
	PetscInt N, gridsize, nu, DimHilbert;
} CTX_USER;

PetscErrorCode UserMatMult(Mat A, Vec x, Vec y);

int main(int argc, char** argv) {
	Mat A; 
	PetscReal Omega;
	PetscReal length = 20;
	PetscReal D = length/2.;
	PetscReal R1z = 0, R2z = 1.4; // position of nuclei
	const PetscReal Pi = 3.14159265359;
	EPS eps; // eigensolver
	PetscErrorCode ierr;
	PetscInt nu=3, gridsize, N, IstartT, IendT, IstartU, IendU, IstartV, IendV, DimHilbert;
	CTX_USER *ctx;
	PetscLogDouble t1,t2,t3,t4,t5;

	ierr = SlepcInitialize(&argc,&argv,(char*)0,help); if (ierr) return ierr;
	ierr = PetscOptionsGetInt(NULL,NULL,"-nu",&nu,NULL);CHKERRQ(ierr);
	ierr = PetscOptionsGetReal(NULL,NULL,"-length",&length,NULL);CHKERRQ(ierr);
//	ierr = PetscPrintf(PETSC_COMM_WORLD, "Nu value: %D \n", nu);CHKERRQ(ierr);
	gridsize = 2*nu+1;
	N = gridsize*gridsize*gridsize;
	DimHilbert = N * (N-1) / 2;
	Omega = length * length * length;

	/* Pre-calculate and save the T, U, V coefficients. */
	Vec T, U, V, Tseq, Useq, Vseq;
	const PetscInt DimCoeff = (4*nu+1)*(4*nu+1)*(4*nu+1);

	ierr = PetscTime(&t1);CHKERRQ(ierr);

	ierr = VecCreateMPI(PETSC_COMM_WORLD, PETSC_DETERMINE, DimCoeff, &T);CHKERRQ(ierr);
	ierr = VecGetOwnershipRange(T, &IstartT, &IendT); CHKERRQ(ierr);
	for(PetscInt t=-2*nu; t<=2*nu; t++) for(PetscInt u=-2*nu; u<=2*nu; u++) for(PetscInt v=-2*nu; v<=2*nu; v++) {
		PetscInt index = ((t+2*nu)*(4*nu+1) + (u+2*nu)) * (4*nu+1) + v+2*nu;
		if(index<IstartT || index>=IendT) continue; // out of local range
		PetscReal Tcomp = 0;
		for(PetscInt x=-nu; x<=nu; x++) for(PetscInt y=-nu; y<=nu; y++) for(PetscInt z=-nu; z<=nu; z++) { // sum over nu
			PetscReal kx=2*Pi*x/length, ky=2*Pi*y/length, kz=2*Pi*z/length;
			PetscReal rx=t*length/gridsize, ry=u*length/gridsize, rz=v*length/gridsize;
			Tcomp += (kx*kx+ky*ky+kz*kz) * PetscCosScalar(kx*rx+ky*ry+kz*rz);
		}
		Tcomp /= 2 * N;
		if(PetscAbsReal(Tcomp)<1e-12) Tcomp = 0; 
		ierr = VecSetValues(T, 1, &index, &Tcomp, INSERT_VALUES); CHKERRQ(ierr);
	}
	ierr = VecAssemblyBegin(T); VecAssemblyEnd(T); CHKERRQ(ierr);

	VecScatter Tscat;
//	ierr = VecCreateSeq(PETSC_COMM_SELF, DimCoeff, &Tseq);CHKERRQ(ierr);
	ierr = VecScatterCreateToAll(T, &Tscat, &Tseq); CHKERRQ(ierr);
//	ierr = VecScatterBegin(Tscat, T, Tseq, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
//	ierr = VecScatterEnd(Tscat, T, Tseq, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
	VecScatterDestroy(&Tscat);
//	ierr = VecAssemblyBegin(Tseq); VecAssemblyEnd(Tseq); CHKERRQ(ierr);
//
//	ierr = PetscTime(&t2);CHKERRQ(ierr);
//	ierr = PetscPrintf(PETSC_COMM_WORLD,"Elapsed Time:\na)T term: %f\n",t2-t1);CHKERRQ(ierr);
//
//	ierr = VecCreateMPI(PETSC_COMM_WORLD, PETSC_DETERMINE, N, &U);CHKERRQ(ierr);
//	ierr = VecGetOwnershipRange(U, &IstartU, &IendU); CHKERRQ(ierr);
//	for(PetscInt t=-nu; t<=nu; t++) for(PetscInt u=-nu; u<=nu; u++) for(PetscInt v=-nu; v<=nu; v++) {
//		PetscInt index;
//		index = ((t+nu)*gridsize + (u+nu))*gridsize + v+nu;
//		if(index<IstartU || index>=IendU) continue;
//		PetscReal Ucomp = 0;
//		for(PetscInt x=-nu; x<=nu; x++) for(PetscInt y=-nu; y<=nu; y++) for(PetscInt z=-nu; z<=nu; z++) { // sum over nu
//			if(x==0 && y==0 && z==0) continue;
//			PetscReal kx=2*Pi*x/length, ky=2*Pi*y/length, kz=2*Pi*z/length;
//			PetscReal rx=t*length/gridsize, ry=u*length/gridsize, rz=v*length/gridsize;
//			Ucomp += PetscCosScalar(kx*rx+ky*ry+kz*(rz-R1z))/(kx*kx+ky*ky+kz*kz);
//			Ucomp += PetscCosScalar(kx*rx+ky*ry+kz*(rz-R2z))/(kx*kx+ky*ky+kz*kz);
//		}
//		Ucomp *= -4 * Pi / Omega;
//		ierr = VecSetValues(U, 1, &index, &Ucomp, INSERT_VALUES); CHKERRQ(ierr);
//	}
//	ierr = VecAssemblyBegin(U); VecAssemblyEnd(U); CHKERRQ(ierr);
//
//	VecScatter Uscat;
//	ierr = VecCreateSeq(PETSC_COMM_SELF, N, &Useq);CHKERRQ(ierr);
//	ierr = VecScatterCreateToAll(U, &Uscat, &Useq); CHKERRQ(ierr);
//	ierr = VecScatterBegin(Uscat, U, Useq, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
//	ierr = VecScatterEnd(Uscat, U, Useq, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
//	VecScatterDestroy(&Uscat);
//
//	ierr = PetscTime(&t3);CHKERRQ(ierr);
//	ierr = PetscPrintf(PETSC_COMM_WORLD,"b)U term: %f\n",t3-t2);CHKERRQ(ierr);
//
//	ierr = VecCreateMPI(PETSC_COMM_WORLD, PETSC_DETERMINE, DimCoeff, &V);CHKERRQ(ierr);
//	ierr = VecGetOwnershipRange(V, &IstartV, &IendV); CHKERRQ(ierr);
//	for(PetscInt t=-2*nu; t<=2*nu; t++) for(PetscInt u=-2*nu; u<=2*nu; u++) for(PetscInt v=-2*nu; v<=2*nu; v++) {
//		PetscInt index = ((t+2*nu)*(4*nu+1) + (u+2*nu)) * (4*nu+1) + v+2*nu;
//		if(index<IstartV || index>=IendV) continue; // out of local range
//		PetscReal Vcomp = 0;
//		for(PetscInt x=-nu; x<=nu; x++) for(PetscInt y=-nu; y<=nu; y++) for(PetscInt z=-nu; z<=nu; z++) { // sum over nu
//			if(x==0 && y==0 && z==0) continue;
//			PetscReal kx=2*Pi*x/length, ky=2*Pi*y/length, kz=2*Pi*z/length;
//			PetscReal rx=t*length/gridsize, ry=u*length/gridsize, rz=v*length/gridsize;
//			if(rx*rx+ry*ry+rz*rz>D*D) continue; // Coulomb cutoff.
//			Vcomp += PetscCosScalar(kx*rx+ky*ry+kz*rz) / (kx*kx+ky*ky+kz*kz);
//		}
//		Vcomp *= 2 * Pi / Omega;
//		ierr = VecSetValues(V, 1, &index, &Vcomp, INSERT_VALUES); CHKERRQ(ierr);
//	}
//	ierr = VecAssemblyBegin(V); VecAssemblyEnd(V); CHKERRQ(ierr);
//
//	VecScatter Vscat;
//	ierr = VecCreateSeq(PETSC_COMM_SELF, DimCoeff, &Vseq);CHKERRQ(ierr);
//	ierr = VecScatterCreateToAll(V, &Vscat, &Vseq); CHKERRQ(ierr);
//	ierr = VecScatterBegin(Vscat, V, Vseq, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
//	ierr = VecScatterEnd(Vscat, V, Vseq, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
//	VecScatterDestroy(&Vscat);
//
//	ierr = PetscTime(&t4);CHKERRQ(ierr);
//	ierr = PetscPrintf(PETSC_COMM_WORLD,"c)V term: %f\n",t4-t3);CHKERRQ(ierr);
//
//	ierr = PetscNew(&ctx);CHKERRQ(ierr);
//	ctx->N = N;
//	ctx->T = &Tseq;
//	ctx->U = &Useq;
//	ctx->V = &Vseq;
//	ctx->gridsize = gridsize;
//	ctx->nu = nu;
//	ctx->DimHilbert = DimHilbert;
//
//	/* Create shell matrix for Hamiltonian. */
//	ierr = MatCreateShell(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, DimHilbert, DimHilbert, ctx, &A);CHKERRQ(ierr);
//	ierr = MatShellSetOperation(A, MATOP_MULT, (void(*)(void))UserMatMult);CHKERRQ(ierr);
//
//	/* Create eigensolver. */
//	ierr = EPSCreate(PETSC_COMM_WORLD, &eps);CHKERRQ(ierr);
//	ierr = EPSSetOperators(eps, A, NULL);CHKERRQ(ierr);
//	ierr = EPSSetType(eps, EPSARNOLDI);CHKERRQ(ierr);
//	ierr = EPSSetProblemType(eps, EPS_HEP);CHKERRQ(ierr);
//	ierr = EPSSetWhichEigenpairs(eps,EPS_SMALLEST_REAL);CHKERRQ(ierr);
//	ierr = EPSSetFromOptions(eps);CHKERRQ(ierr);
//
//	/* Preconditioner? */
//
//	/* Solve the system. */
//	ierr = EPSSolve(eps);CHKERRQ(ierr);
//	ierr = PetscTime(&t5);CHKERRQ(ierr);
//	ierr = PetscPrintf(PETSC_COMM_WORLD,"d)Solving the system: %f\n",t5-t4);CHKERRQ(ierr);
//
//
//	/* Output. */
//	PetscInt nconv;
//	ierr = EPSGetConverged(eps,&nconv);CHKERRQ(ierr);
//	ierr = PetscPrintf(PETSC_COMM_WORLD," Number of converged eigenpairs: %D\n\n",nconv);CHKERRQ(ierr);
//	PetscScalar eigenreal, eigenimag;
//	Vec eigenvr, eigenvi;
//	ierr = VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, DimHilbert, &eigenvr);CHKERRQ(ierr);
//	ierr = VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, DimHilbert, &eigenvi);CHKERRQ(ierr);
//	if(nconv>0) {
//		ierr = EPSGetEigenpair(eps, 0, &eigenreal, &eigenimag, eigenvr, eigenvi);CHKERRQ(ierr);
//	}
//	ierr = PetscPrintf(PETSC_COMM_WORLD, "The eigenvalue is: %9f\n", eigenreal);CHKERRQ(ierr);
//	ierr = PetscViewerPopFormat(PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
	ierr = EPSDestroy(&eps);CHKERRQ(ierr);
//	ierr = MatDestroy(&A);CHKERRQ(ierr);
	ierr = VecDestroy(&T);CHKERRQ(ierr);
//	ierr = VecDestroy(&U);CHKERRQ(ierr);
//	ierr = VecDestroy(&V);CHKERRQ(ierr);
//	ierr = VecDestroy(ctx->T);CHKERRQ(ierr);
//	ierr = VecDestroy(ctx->U);CHKERRQ(ierr);
//	ierr = VecDestroy(ctx->V);CHKERRQ(ierr);
	ierr = VecDestroy(&Tseq);CHKERRQ(ierr);
//	ierr = VecDestroy(&Useq);CHKERRQ(ierr);
//	ierr = VecDestroy(&Vseq);CHKERRQ(ierr);
//	ierr = PetscFree(ctx);CHKERRQ(ierr);
//	ierr = VecDestroy(&eigenvr);CHKERRQ(ierr);
//	ierr = VecDestroy(&eigenvi);CHKERRQ(ierr);
	ierr = SlepcFinalize();
	return ierr;
}

PetscErrorCode UserMatMult(Mat A, Vec x, Vec y) {
	CTX_USER *ctx;
	PetscErrorCode ierr;
	const PetscReal *xcomp;
	PetscReal *ycomp;
	Vec xseq;

	PetscFunctionBeginUser;
	ierr = MatShellGetContext(A,(void**)&ctx);CHKERRQ(ierr);
	ierr = VecCreateSeq(PETSC_COMM_SELF, ctx->DimHilbert, &xseq);CHKERRQ(ierr); // save the scattered vector

	VecScatter scat;
	ierr = VecScatterCreateToAll(x, &scat, &xseq); CHKERRQ(ierr);
	ierr = VecScatterBegin(scat, x, xseq, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
	ierr = VecScatterEnd(scat, x, xseq, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
	VecScatterDestroy(&scat);

	ierr = VecGetArrayRead(xseq, &xcomp);CHKERRQ(ierr);
	ierr = VecGetArray(y, &ycomp);CHKERRQ(ierr);

	/* Get ownership. */
	PetscInt low, high;
	ierr = VecGetOwnershipRange(x, &low, &high); CHKERRQ(ierr);

	/* Mat-vec multiplication for dual basis with second quantized form. */
	PetscInt pnext=0, qnext=0;
	for(PetscInt localIndex=low; localIndex<high; localIndex++) {
		/* Restore the p, q values, assuming p>q. */
		/* Indexing: |0,0,...0,1,1> -- 1st, (smaller index on the right)
			           |0,0,...1,0,1> -- 2nd, ... */
		PetscInt p, q;
		if(localIndex==low) { // for the lowest index, calculate p and q.
			q = (PetscInt)PetscFloorReal(ctx->N - 0.5 - PetscSqrtReal(ctx->N*ctx->N-ctx->N+0.249-2*localIndex)); // 0.25->0.249 to avoid loss of machine accuracy
			p = q + 1 + localIndex - q * (2*ctx->N-q-1) / 2;
		}
		else {
			p = pnext;
			q = qnext;
		}

		PetscInt px = (PetscInt)PetscFloorReal((PetscReal)p/ctx->gridsize/ctx->gridsize) - ctx->nu;
		PetscInt py = (PetscInt)PetscFloorReal(((PetscReal)p-(px+ctx->nu)*ctx->gridsize*ctx->gridsize)/ctx->gridsize) - ctx->nu;
		PetscInt pz = p - (px+ctx->nu)*ctx->gridsize*ctx->gridsize - (py+ctx->nu)*ctx->gridsize - ctx->nu;
		PetscInt qx = (PetscInt)PetscFloorReal((PetscReal)q/ctx->gridsize/ctx->gridsize) - ctx->nu;
		PetscInt qy = (PetscInt)PetscFloorReal(((PetscReal)q-(qx+ctx->nu)*ctx->gridsize*ctx->gridsize)/ctx->gridsize) - ctx->nu;
		PetscInt qz = q - (qx+ctx->nu)*ctx->gridsize*ctx->gridsize - (qy+ctx->nu)*ctx->gridsize - ctx->nu;

		ycomp[localIndex-low] = 0;

		/* First try no phase factor. */
		// third orbital r != p or q.
		for(PetscInt rx=-ctx->nu; rx<=ctx->nu; rx++) for(PetscInt ry=-ctx->nu; ry<=ctx->nu; ry++) for(PetscInt rz=-ctx->nu; rz<=ctx->nu; rz++) {

			PetscInt r = ((rx+ctx->nu)*ctx->gridsize + ry+ctx->nu)*ctx->gridsize + rz+ctx->nu;
			PetscInt Tindex, globalIndex;
			PetscReal amp, coefficient;

			// |rp> -> |pq>
			for(PetscInt i=0; i<=0; i++) {
				if(r==q) continue; // in 1-electron subspace
				Tindex = ((rx-px+2*ctx->nu)*(4*ctx->nu+1) + ry-py+2*ctx->nu)*(4*ctx->nu+1) + rz-pz+2*ctx->nu;
				ierr = VecGetValues(*ctx->T, 1, &Tindex, &coefficient); CHKERRQ(ierr);
				if(PetscAbsReal(coefficient)<1e-10) continue; //sparsity
				if(r==p) { // self-kinetic term
					amp = xcomp[localIndex];
					ycomp[localIndex-low] += amp * coefficient;
					continue;
				}
				PetscInt phase = 1;
				if(r<p) phase *= -1;
				if(q<p) phase *= -1;
				globalIndex = r < p ? (2*ctx->N-r-1)*r/2 + p-r-1 : (2*ctx->N-p-1)*p/2 + r-p-1;
				amp = xcomp[globalIndex];
				ycomp[localIndex-low] += phase * amp * coefficient;
			}

			// |rq> -> |pq>
			for(PetscInt i=0; i<=0; i++) {
				if(r==p) continue;
				Tindex = ((rx-qx+2*ctx->nu)*(4*ctx->nu+1) + ry-qy+2*ctx->nu)*(4*ctx->nu+1) + rz-qz+2*ctx->nu;
				ierr = VecGetValues(*ctx->T, 1, &Tindex, &coefficient); CHKERRQ(ierr);
				if(PetscAbsReal(coefficient)<1e-10) continue; //sparsity
				if(r==q) {
					amp = xcomp[localIndex];
					ycomp[localIndex-low] += amp * coefficient;
					continue;
				}
				PetscInt phase = 1;
				if(r<q) phase *= -1;
				if(p<q) phase *= -1;
				globalIndex = r < q ? (2*ctx->N-r-1)*r/2 + q-r-1 : (2*ctx->N-q-1)*q/2 + r-q-1;
				amp = xcomp[globalIndex];
				ycomp[localIndex-low] += phase * amp * coefficient;
			}
		}

		// U and V terms. 
		PetscReal amp = xcomp[localIndex];
		PetscReal coefficient;

		ierr = VecGetValues(*ctx->U, 1, &p, &coefficient); CHKERRQ(ierr);
		ycomp[localIndex-low] += amp * coefficient;
		ierr = VecGetValues(*ctx->U, 1, &q, &coefficient); CHKERRQ(ierr);
		ycomp[localIndex-low] += amp * coefficient;

		PetscInt Vindex = ((px-qx+2*ctx->nu)*(4*ctx->nu+1) + py-qy+2*ctx->nu)*(4*ctx->nu+1) + pz-qz+2*ctx->nu;
		ierr = VecGetValues(*ctx->V, 1, &Vindex, &coefficient); CHKERRQ(ierr);
		ycomp[localIndex-low] += 2. * amp * coefficient; // *2?

		/* Set next index. */
		if(p==ctx->N-1) {
			qnext = q + 1;
			pnext = qnext + 1;
		}
		else {
			qnext = q;
			pnext = p + 1;
		}
	}

//	ierr = VecView(*ctx->V, PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
	ierr = VecRestoreArrayRead(xseq, &xcomp);CHKERRQ(ierr);
	ierr = VecRestoreArray(y,&ycomp);CHKERRQ(ierr);
	ierr = VecDestroy(&xseq);CHKERRQ(ierr);
	ierr = PetscFree(xcomp); CHKERRQ(ierr);
	ierr = PetscFree(ycomp); CHKERRQ(ierr);
	PetscFunctionReturn(0);
}


