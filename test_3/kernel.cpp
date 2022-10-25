/*========================== begin_copyright_notice ============================

Copyright (C) 2020-2021 Intel Corporation

SPDX-License-Identifier: MIT

============================= end_copyright_notice ===========================*/

#include <cm/cm.h>

#define SZ 16
const char zero[SZ] = { 0 };

extern "C"
#ifndef __INTELLISENSE__
	_GENX_
#endif
	void
	calc_c(uint32_t dst_col, uint32_t dst_row, int k,
	       SurfaceIndex indxA [[type("image2d_t float")]],
	       SurfaceIndex indxB [[type("image2d_t float")]], matrix_ref<float, SZ, SZ> C)
{
	for (int kk = 0; kk < k; kk += SZ) {
#pragma unroll
		for (int i = 0; i < SZ; i++) {
			vector<float, SZ> a;
			read(indxA, kk * sizeof(float), dst_row + i, a.select<8, 1>(0));
			read(indxA, (kk + 8) * sizeof(float), dst_row + i, a.select<8, 1>(8));
#pragma unroll
			for (int j = 0; j < SZ; j++) {
				vector<float, SZ> res(zero);

				matrix<float, SZ, 1> b;

				// for (int ii = 0; ii < 16; ii++)
				// 	printf(" a(%d)=%.3f", ii, a(ii));
				// printf("\n");

				// read(indxA, (kk + 16) * sizeof(float), dst_row + i,
				//      a.select<8, 1>(16));
				// read(indxA, (kk + 24) * sizeof(float), dst_row + i,
				//      a.select<8, 1>(24));

				read(indxB, dst_col + j * sizeof(float), kk,
				     b.select<8, 1, 1, 1>(0, 0));
				read(indxB, dst_col + j * sizeof(float), kk + 8,
				     b.select<8, 1, 1, 1>(8, 0));
				// for (int ii = 0; ii < 16; ii++)
				// 	printf(" b(%d, 0)=%.3f ", ii, b(ii, 0));
				// printf("\n");
				// read(indxB, dst_col + j * sizeof(float), kk + 16,
				//      b.select<8, 1, 1, 1>(16, 0));
				// read(indxB, dst_col + j * sizeof(float), kk + 24,
				//      b.select<8, 1, 1, 1>(24, 0));

				b = b * a;
				C(i, j) += cm_sum<float>(b);
			}
			// float c_old = C(i, j);
			// C(i, j) += cm_sum<float>(res);
			// printf("i=%d j=%d c_old=%.3f, c_new=%.3f\n", i, j, c_old, C(i, j));
		}
	}
}

// C := alpha*A*B + beta*C,
// A(m x k) , B(k x n) , C(m x n)
// kernel calulate 1x16 block of C
extern "C"
#ifndef __INTELLISENSE__
	_GENX_MAIN_
#endif
	void
	sgemm_kernel_am(int m, int n, int k, SurfaceIndex indxA [[type("image2d_t float")]],
			SurfaceIndex indxB [[type("image2d_t float")]],
			SurfaceIndex indxC [[type("image2d_t float")]])

{
	// printf("group_count(0)=%d, group_count(1)=%d local_size(0)=%d local_size(1)=%d gid(0)=%d, gid(1)=%d, lid(0)=%d, lid(1)=%d, cm_linear_global_id=%d\n",
	//        cm_group_count(0), cm_group_count(1), cm_local_size(0), cm_local_size(1),
	//        cm_group_id(0), cm_group_id(1), cm_local_id(0), cm_local_id(1),
	//        cm_linear_global_id());

	uint32_t dst_col = SZ * cm_group_id(0) * sizeof(float);
	uint32_t dst_row = SZ * cm_group_id(1);
	// vector<float, 32> res(zero);

	matrix<float, SZ, SZ> c;

	// Read the earlier value of C matrix (Read 32x32 block of C)
	read(indxC, dst_col, dst_row, c.select<8, 1, 8, 1>(0, 0));
	read(indxC, dst_col + 8 * sizeof(float), dst_row, c.select<8, 1, 8, 1>(0, 8));
	// read(indxC, dst_col + 16 * sizeof(float), dst_row, c.select<8, 1, 8, 1>(0, 16));
	// read(indxC, dst_col + 24 * sizeof(float), dst_row, c.select<8, 1, 8, 1>(0, 24));

	read(indxC, dst_col, dst_row + 8, c.select<8, 1, 8, 1>(8, 0));
	read(indxC, dst_col + 8 * sizeof(float), dst_row + 8, c.select<8, 1, 8, 1>(8, 8));
	// read(indxC, dst_col + 16 * sizeof(float), dst_row + 8, c.select<8, 1, 8, 1>(8, 16));
	// read(indxC, dst_col + 24 * sizeof(float), dst_row + 8, c.select<8, 1, 8, 1>(8, 24));

	// read(indxC, dst_col, dst_row + 16, c.select<8, 1, 8, 1>(16, 0));
	// read(indxC, dst_col + 8 * sizeof(float), dst_row + 16, c.select<8, 1, 8, 1>(16, 8));
	// read(indxC, dst_col + 16 * sizeof(float), dst_row + 16, c.select<8, 1, 8, 1>(16, 16));
	// read(indxC, dst_col + 24 * sizeof(float), dst_row + 16, c.select<8, 1, 8, 1>(16, 24));

	// read(indxC, dst_col, dst_row + 24, c.select<8, 1, 8, 1>(24, 0));
	// read(indxC, dst_col + 8 * sizeof(float), dst_row + 24, c.select<8, 1, 8, 1>(24, 8));
	// read(indxC, dst_col + 16 * sizeof(float), dst_row + 24, c.select<8, 1, 8, 1>(24, 16));
	// read(indxC, dst_col + 24 * sizeof(float), dst_row + 24, c.select<8, 1, 8, 1>(24, 24));

	calc_c(dst_col, dst_row, k, indxA, indxB, c.select_all());

	write(indxC, dst_col, dst_row, c.select<8, 1, 8, 1>(0, 0));
	write(indxC, dst_col + 8 * sizeof(float), dst_row, c.select<8, 1, 8, 1>(0, 8));
	// write(indxC, dst_col + 16 * sizeof(float), dst_row, c.select<8, 1, 8, 1>(0, 16));
	// write(indxC, dst_col + 24 * sizeof(float), dst_row, c.select<8, 1, 8, 1>(0, 24));

	write(indxC, dst_col, dst_row + 8, c.select<8, 1, 8, 1>(8, 0));
	write(indxC, dst_col + 8 * sizeof(float), dst_row + 8, c.select<8, 1, 8, 1>(8, 8));
	// write(indxC, dst_col + 16 * sizeof(float), dst_row + 8, c.select<8, 1, 8, 1>(8, 16));
	// write(indxC, dst_col + 24 * sizeof(float), dst_row + 8, c.select<8, 1, 8, 1>(8, 24));

	// write(indxC, dst_col, dst_row + 16, c.select<8, 1, 8, 1>(16, 0));
	// write(indxC, dst_col + 8 * sizeof(float), dst_row + 16, c.select<8, 1, 8, 1>(16, 8));
	// write(indxC, dst_col + 16 * sizeof(float), dst_row + 16, c.select<8, 1, 8, 1>(16, 16));
	// write(indxC, dst_col + 24 * sizeof(float), dst_row + 16, c.select<8, 1, 8, 1>(16, 24));

	// write(indxC, dst_col, dst_row + 24, c.select<8, 1, 8, 1>(24, 0));
	// write(indxC, dst_col + 8 * sizeof(float), dst_row + 24, c.select<8, 1, 8, 1>(24, 8));
	// write(indxC, dst_col + 16 * sizeof(float), dst_row + 24, c.select<8, 1, 8, 1>(24, 16));
	// write(indxC, dst_col + 24 * sizeof(float), dst_row + 24, c.select<8, 1, 8, 1>(24, 24));

	// for (int kk = 0; kk < k; kk += SZ) {
	// 	vector<float, 32> a;
	// 	matrix<float, 32, 1> b;

	// 	vector<float, 32> t[32];

	// 	read(indxA, kk * sizeof(float), dst_row, a.select<8, 1>(0));
	// 	read(indxA, (kk + 8) * sizeof(float), dst_row, a.select<8, 1>(8));
	// 	read(indxA, (kk + 16) * sizeof(float), dst_row, a.select<8, 1>(16));
	// 	read(indxA, (kk + 24) * sizeof(float), dst_row, a.select<8, 1>(24));

	// 	// for (int i = 0; i < 16; i++)
	// 	// 	printf(" a(%d)=%.3f", i, a(i));
	// 	// printf("\n");

	// 	read(indxB, dst_col, kk, b.select<8, 1, 1, 1>(0, 0));
	// 	read(indxB, dst_col, kk + 8, b.select<8, 1, 1, 1>(8, 0));
	// 	read(indxB, dst_col, kk + 16, b.select<8, 1, 1, 1>(16, 0));
	// 	read(indxB, dst_col, kk + 24, b.select<8, 1, 1, 1>(24, 0));
	// 	// for (int i = 0; i < 16; i++)
	// 	// 	printf(" b(%d, 0)=%.3f ", i, b(i, 0));
	// 	// printf("\n");

	// 	a = b * a;
	// 	// for (int i = 0; i < 16; i++)
	// 	// 	printf(" a(%d)=%.3f", i, a(i));
	// 	// printf("\n");

	// 	res += a;
	// }
	// vector<float, 1> c_old;
	// read(indxC, dst_col, dst_row, c_old.select<1, 1>(0));

	// // float val = c_old(0);
	// // printf("ic=%d, jc=%d old_c=%.3f\n", ic, jc, val);
	// vector<float, 1> res_scal = c_old(0) + cm_sum<float>(res);

	// write(indxC, dst_col, dst_row, res_scal);
}
