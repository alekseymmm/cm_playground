/*========================== begin_copyright_notice ============================

Copyright (C) 2020-2021 Intel Corporation

SPDX-License-Identifier: MIT

============================= end_copyright_notice ===========================*/

#include <cm/cm.h>

#define SZ 16

// C := alpha*A*B + beta*C,
// A(m x k) , B(k x n) , C(m x n)
// kernel calulate 1x16 block of C
extern "C"
#ifndef __INTELLISENSE__
	_GENX_MAIN_
#endif
	void
	sgemm_kernel_am(int m, int n, int k, int ic, int jc,
			SurfaceIndex indxA [[type("image2d_t float")]],
			SurfaceIndex indxB [[type("image2d_t float")]],
			SurfaceIndex indxC [[type("image2d_t float")]])

{
	// vector<int, SZ> ivector1;
	// vector<int, SZ> ivector2;
	// vector<int, SZ> ovector;

	printf("group_count(0)=%d, group_count(1)=%d local_size(0)=%d local_size(1)=%d gid(0)=%d, gid(1)=%d, lid(0)=%d, lid(1)=%d, cm_linear_global_id=%d\n",
	       cm_group_count(0), cm_group_count(1), cm_local_size(0),
	       cm_local_size(1), cm_group_id(0), cm_group_id(1), cm_local_id(0),
	       cm_local_id(1), cm_linear_global_id());
	unsigned offset = sizeof(unsigned) * SZ * cm_group_id(0);
	//
	// read-in the arguments
	// read(isurface1, offset, ivector1);
	// read(isurface2, offset, ivector2);
	// // perform addition
	// ovector = ivector1 + ivector2;
	// // write-out the results
	// write(osurface, offset, ovector);
}
