include ${SLEPC_DIR}/lib/slepc/conf/slepc_common

hydrogen: hydrogen.o
	-${CLINKER} -o hydrogen hydrogen.o ${SLEPC_EPS_LIB}
	${RM} hydrogen.o
