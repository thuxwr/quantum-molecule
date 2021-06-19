/*
	 Solve hydrogen atom, and try parallel calculation.

	 Weiran, May 30, 2021.
*/

static char help[] = "Solve hydrogen atom using multiple cores.";

#include <slepceps.h>


/* Pre-calculated data. */
typedef struct {
	Vec *T, *U, *V;
	PetscInt N, gridsize, nu;
} CTX_USER;

PetscErrorCode UserMatMult(Mat A, Vec x, Vec y);

int main(int argc, char** argv) {
	Mat A; 
	PetscReal Omega;
	PetscReal length = 20;
	const PetscReal Pi = 3.14159265359;
	EPS eps; // eigensolver
	PetscErrorCode ierr;
	PetscInt nu=3, gridsize, N;
	CTX_USER *ctx;
	PetscLogDouble t1,t2,t3,t4;

	ierr = SlepcInitialize(&argc,&argv,(char*)0,help); if (ierr) return ierr;
	ierr = PetscOptionsGetInt(NULL,NULL,"-nu",&nu,NULL);CHKERRQ(ierr);
	ierr = PetscOptionsGetReal(NULL,NULL,"-length",&length,NULL);CHKERRQ(ierr);
	ierr = PetscPrintf(PETSC_COMM_WORLD, "Nu value: %D \n", nu);CHKERRQ(ierr);
	gridsize = 2*nu+1;
	N = gridsize*gridsize*gridsize;
	Omega = length * length * length;

	/* Pre-calculate and save the T, U, V coefficients. */
	/* First try Seq. */
	Vec T, U;
	const PetscInt CoeffDim = (4*nu+1)*(4*nu+1)*(4*nu+1);

	ierr = PetscTime(&t1);CHKERRQ(ierr);
	ierr = VecCreateSeq(PETSC_COMM_SELF, CoeffDim, &T);CHKERRQ(ierr);
	//TODO: only need 0~nu and use the even property.
	for(PetscInt t=0; t<=2*nu; t++) for(PetscInt u=-2*nu; u<=2*nu; u++) for(PetscInt v=-2*nu; v<=2*nu; v++) {
		PetscInt index = ((t+2*nu)*(4*nu+1) + (u+2*nu)) * (4*nu+1) + v+2*nu;
		PetscReal Tcomp = 0;
		for(PetscInt x=-nu; x<=nu; x++) for(PetscInt y=-nu; y<=nu; y++) for(PetscInt z=-nu; z<=nu; z++) { // sum over nu
			PetscReal kx=2*Pi*x/length, ky=2*Pi*y/length, kz=2*Pi*z/length;
			PetscReal rx=t*length/gridsize, ry=u*length/gridsize, rz=v*length/gridsize;
			Tcomp += (kx*kx+ky*ky+kz*kz) * PetscCosScalar(kx*rx+ky*ry+kz*rz);
		}
		Tcomp /= 2 * N;
		ierr = VecSetValues(T, 1, &index, &Tcomp, INSERT_VALUES); CHKERRQ(ierr);
		if(t!=0) { // set symmetric counterparts
			index = ((-t+2*nu)*(4*nu+1) + (-u+2*nu)) * (4*nu+1) - v+2*nu;
			ierr = VecSetValues(T, 1, &index, &Tcomp, INSERT_VALUES); CHKERRQ(ierr);
		}
	}
	ierr = VecAssemblyBegin(T); VecAssemblyEnd(T); CHKERRQ(ierr);
	ierr = PetscTime(&t2);CHKERRQ(ierr);

	ierr = VecCreateSeq(PETSC_COMM_SELF, N, &U);CHKERRQ(ierr);
	for(PetscInt t=-nu; t<=nu; t++) for(PetscInt u=-nu; u<=nu; u++) for(PetscInt v=-nu; v<=nu; v++) {
		PetscInt index;
		index = ((t+nu)*gridsize + (u+nu))*gridsize + v+nu;
		PetscReal Ucomp = 0;
		for(PetscInt x=-nu; x<=nu; x++) for(PetscInt y=-nu; y<=nu; y++) for(PetscInt z=-nu; z<=nu; z++) { // sum over nu
			if(x==0 && y==0 && z==0) continue;
			PetscReal kx=2*Pi*x/length, ky=2*Pi*y/length, kz=2*Pi*z/length;
			PetscReal rx=t*length/gridsize, ry=u*length/gridsize, rz=v*length/gridsize;
			Ucomp += PetscCosScalar(kx*rx+ky*ry+kz*rz)/(kx*kx+ky*ky+kz*kz);
		}
		Ucomp *= -4 * Pi / Omega;
		ierr = VecSetValues(U, 1, &index, &Ucomp, INSERT_VALUES); CHKERRQ(ierr);
	}
	ierr = VecAssemblyBegin(U); VecAssemblyEnd(U); CHKERRQ(ierr);

	ierr = PetscTime(&t3);CHKERRQ(ierr);


	ierr = PetscNew(&ctx);CHKERRQ(ierr);
	ctx->N = N;
	ctx->T = &T;
	ctx->U = &U;
	ctx->gridsize = gridsize;
	ctx->nu = nu;

	/* Create shell matrix for Hamiltonian. */
	ierr = MatCreateShell(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, N, N, ctx, &A);CHKERRQ(ierr);
	ierr = MatShellSetOperation(A, MATOP_MULT, (void(*)(void))UserMatMult);CHKERRQ(ierr);

	/* Create eigensolver. */
	ierr = EPSCreate(PETSC_COMM_WORLD, &eps);CHKERRQ(ierr);
	ierr = EPSSetOperators(eps, A, NULL);CHKERRQ(ierr);
	ierr = EPSSetType(eps, EPSARNOLDI);CHKERRQ(ierr);
	ierr = EPSSetProblemType(eps, EPS_HEP);CHKERRQ(ierr);
	ierr = EPSSetWhichEigenpairs(eps,EPS_SMALLEST_REAL);CHKERRQ(ierr);
	ierr = EPSSetFromOptions(eps);CHKERRQ(ierr);

	/* Preconditioner? */

	/* Solve the system. */
	ierr = EPSSolve(eps);CHKERRQ(ierr);
	ierr = PetscTime(&t4);CHKERRQ(ierr);
	ierr = PetscPrintf(PETSC_COMM_WORLD,"Elapsed Time:\na)T term: %f\nb)U term: %f\nc)Solving the system: %f\n",t2-t1,t3-t2,t4-t3);CHKERRQ(ierr);


	/* Output. */
	PetscInt nconv;
	ierr = EPSGetConverged(eps,&nconv);CHKERRQ(ierr);
	ierr = PetscPrintf(PETSC_COMM_WORLD," Number of converged eigenpairs: %D\n\n",nconv);CHKERRQ(ierr);
	PetscScalar eigenreal, eigenimag;
	Vec eigenvr, eigenvi;
	ierr = VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, N, &eigenvr);CHKERRQ(ierr);
	ierr = VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, N, &eigenvi);CHKERRQ(ierr);
	if(nconv>0) {
		ierr = EPSGetEigenpair(eps, 0, &eigenreal, &eigenimag, eigenvr, eigenvi);CHKERRQ(ierr);
	}
	ierr = PetscPrintf(PETSC_COMM_WORLD, "The eigenvalue is: %9f\n", eigenreal);CHKERRQ(ierr);
	ierr = PetscViewerPopFormat(PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
	ierr = EPSDestroy(&eps);CHKERRQ(ierr);
	ierr = MatDestroy(&A);CHKERRQ(ierr);
	ierr = VecDestroy(&eigenvr);CHKERRQ(ierr);
	ierr = VecDestroy(&eigenvi);CHKERRQ(ierr);
	ierr = SlepcFinalize();
	return ierr;
}

PetscErrorCode UserMatMult(Mat A, Vec x, Vec y) {
	CTX_USER *ctx;
	PetscErrorCode ierr;
	const PetscReal *px;
	PetscReal *py;
	Vec rtemp;
	Vec xseq;

	/* test */
	/*
	PetscInt size;
	ierr = VecGetLocalSize(x, &size);
	ierr = VecGetSize(x, &size);
	*/


	PetscFunctionBeginUser;
	ierr = MatShellGetContext(A,(void**)&ctx);CHKERRQ(ierr);
	ierr = VecCreateSeq(PETSC_COMM_SELF, 3, &rtemp);CHKERRQ(ierr);
	ierr = VecCreateSeq(PETSC_COMM_SELF, ctx->N, &xseq);CHKERRQ(ierr); // save the scattered vector

	VecScatter scat;
	ierr = VecScatterCreateToAll(x, &scat, &xseq); CHKERRQ(ierr);
	ierr = VecScatterBegin(scat, x, xseq, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
	ierr = VecScatterEnd(scat, x, xseq, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);

	ierr = VecGetArrayRead(xseq, &px);CHKERRQ(ierr);
	ierr = VecGetArray(y, &py);CHKERRQ(ierr);

	/* Get ownership. */
	PetscInt low, high;
	ierr = VecGetOwnershipRange(x, &low, &high); CHKERRQ(ierr);

	/* Mat-vec multiplication for dual basis with second quantized form. */
	for(PetscInt t=-ctx->nu; t<=ctx->nu; t++) for(PetscInt u=-ctx->nu; u<=ctx->nu; u++) for(PetscInt v=-ctx->nu; v<=ctx->nu; v++) {
		PetscInt p = ((t+ctx->nu)*ctx->gridsize + (u+ctx->nu))*ctx->gridsize + v+ctx->nu;
		if(p<low || p>=high) continue; // only work on local values
		py[p-low] = 0;
		for(PetscInt i=-ctx->nu; i<=ctx->nu; i++) for(PetscInt j=-ctx->nu; j<=ctx->nu; j++) for(PetscInt k=-ctx->nu; k<=ctx->nu; k++) {
			PetscInt q = ((i+ctx->nu)*ctx->gridsize + (j+ctx->nu))*ctx->gridsize + k+ctx->nu;
			PetscReal amp = px[q];
			PetscInt Tindex = ((i-t+2*ctx->nu)*(4*ctx->nu+1) + (j-u+2*ctx->nu))*(4*ctx->nu+1) + k-v+2*ctx->nu;
			PetscReal coefficient;
			ierr = VecGetValues(*ctx->T, 1, &Tindex, &coefficient); CHKERRQ(ierr);
//			ierr = PetscPrintf(PETSC_COMM_WORLD, "coefficient for p: %D, q: %D, is: %9f \n", p, q, coefficient);
			py[p-low] += coefficient * amp;
			if(p==q) { // U terms
				ierr = VecGetValues(*ctx->U, 1, &q, &coefficient); CHKERRQ(ierr);
				py[p-low] += coefficient * amp;
			}
		}
	}

	ierr = VecRestoreArrayRead(xseq, &px);CHKERRQ(ierr);
	ierr = VecRestoreArray(y,&py);CHKERRQ(ierr);
	PetscFunctionReturn(0);
}


