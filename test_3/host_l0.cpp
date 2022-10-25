/*========================== begin_copyright_notice ============================

Copyright (C) 2020-2021 Intel Corporation

SPDX-License-Identifier: MIT

============================= end_copyright_notice ===========================*/

#include <iostream>
#include <cassert>
#include <math.h>
#include <vector>

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>

#include <level_zero/ze_api.h>

#include <gsl/gsl_cblas.h>
#include <algorithm>

using namespace std;

#define SZ 160
#define KERNEL_SZ 16
#define CHECK(a)                                                              \
	do {                                                                  \
		auto err = (a);                                               \
		if (err != 0) {                                               \
			fprintf(stderr, "FAIL: err=%d @ line=%d (%s)\n", err, \
				__LINE__, (#a));                              \
			exit(err);                                            \
		}                                                             \
	} while (0)
#define CHECK2(a, msg)                                                      \
	do {                                                                \
		if ((a)) {                                                  \
			fprintf(stderr, "FAIL: @ line=%d (%s)\n", __LINE__, \
				(msg));                                     \
			exit(-1);                                           \
		}                                                           \
	} while (0)
#ifndef KERNEL
#error "Error: KERNEL must be defined with location of kernel binary"
#endif

#define KERNEL_ALIGN 16LLU
/* @a is a power of 2 value */
#define __ALIGN_KERNEL_MASK(x, mask) (((x) + (mask)) & ~(mask))
#define __ALIGN_KERNEL(x, a) __ALIGN_KERNEL_MASK(x, (typeof(x))(a)-1)
#define ALIGN(x, a) __ALIGN_KERNEL((x), (a))

static float randData(float low, float high)
{
	float t = (float)rand() / (float)RAND_MAX;
	return (1.0f - t) * low + t * high;
}

class Matrix {
	float *M;
	uint32_t nrows;
	uint32_t nrows_aligned;
	uint32_t ncols;
	uint32_t ncols_aligned;

    public:
	float &operator()(int r, int c)
	{
		return M[r * ncols_aligned + c];
	}

	Matrix(uint32_t rows, uint32_t cols, bool init)
	{
		this->nrows = rows;
		this->nrows_aligned = ALIGN(this->nrows, KERNEL_ALIGN);
		this->ncols = cols;
		this->ncols_aligned = ALIGN(this->ncols, KERNEL_ALIGN);

		size_t size = sizeof(float) * this->nrows_aligned * this->ncols_aligned;

		M = (float *)aligned_alloc(4096, size);

		if (init)
			for (int i = 0; i < rows; i++)
				for (int j = 0; j < cols; j++)
					(*this)(i, j) = randData(0.0f, 1.0f);
	}

#define CORRECTNESS_THRESHOLD 0.00002
	bool operator==(Matrix &m)
	{
		double max_relerror = 0.0;
		double max_abserror = 0.0;
		for (int r = 0; r < nrows; r++)
			for (int c = 0; c < ncols; c++) {
				// printf("I=%3d N=%3d  %08x  %08x\n", r, c, *(unsigned int*)&(*this)(r,c), *(unsigned int *)&m(r,c));

				double relerror = fabs((*this)(r, c) - m(r, c)) /
						  max(fabs((*this)(r, c)), fabs(m(r, c)));
				double abserror = fabs((*this)(r, c) - m(r, c));

				max_relerror = max(max_relerror, relerror);
				max_abserror = max(max_abserror, abserror);

				if (relerror > CORRECTNESS_THRESHOLD) {
					printf("Failure %f %f relerror: %lf at [%d, %d]\n",
					       (*this)(r, c), m(r, c), relerror, r, c);
					return false;
				}
			}
		printf("max_relerror = %e  absolute error = %e\n", max_relerror, max_abserror);
		return (max_relerror > CORRECTNESS_THRESHOLD) ? false : true;
		return true;
	}

	bool operator!=(Matrix &m)
	{
		return !operator==(m);
	}

	Matrix(Matrix &m)
		: nrows(m.nrows)
		, nrows_aligned(m.nrows_aligned)
		, ncols(m.ncols)
		, ncols_aligned(m.ncols_aligned)
	{
		size_t size = sizeof(float) * this->nrows_aligned * this->ncols_aligned;

		M = (float *)aligned_alloc(4096, size);
		for (int i = 0; i < nrows; i++) {
			for (int j = 0; j < ncols; j++)
				(*this)(i, j) = m(i, j);
		}
	}

	uint32_t rows()
	{
		return nrows;
	};
	uint32_t cols()
	{
		return nrows;
	};
	uint32_t ld()
	{
		return ncols_aligned;
	}
	float *data()
	{
		return M;
	}

	~Matrix()
	{
		free(M);
	}
};

// C := alpha*A*B + beta*C,
// A(m x k) , B(k x n) , C(m x n)
static int sgemmNxN(int m, int n, int k, float alpha, float *A, int lda, float *B, int ldb,
		    float beta, float *C, int ldc)
{
	for (int r = 0; r < m; r++)
		for (int c = 0; c < n; c++) {
			float tmp = 0.0f;
			for (int t = 0; t < k; t++)
				tmp += A[r * lda + t] * B[t * ldb + c];
			C[r * ldc + c] = alpha * tmp + beta * C[r * ldc + c];
		}

	return 0;
}

int main(int argc, char *argv[])
{
	// uint32_t a_rows = 15, a_cols = 15;
	// uint32_t b_rows = 15, b_cols = 15;
	// uint32_t c_rows = 15, c_cols = 15;
	//
	uint32_t a_rows = 1024, a_cols = 1024;
	uint32_t b_rows = 1024, b_cols = 1024;
	uint32_t c_rows = 1024, c_cols = 1024;

	Matrix A_in(a_rows, a_cols, true);
	Matrix B_in(b_rows, b_cols, true);
	Matrix C_out(c_rows, c_cols, true);
	Matrix C_out_gpu(C_out);
	Matrix C_old(C_out);
	Matrix C_test(C_out);

	float alpha = +1.0, beta = +1.0;

	sgemmNxN(a_rows, b_cols, b_rows, alpha, A_in.data(), A_in.ld(), B_in.data(), B_in.ld(),
		 beta, C_out.data(), C_out.ld());
	printf("sgemmNxN multiplication is done\n");

	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, a_rows, b_cols, b_rows, alpha,
		    A_in.data(), A_in.ld(), B_in.data(), B_in.ld(), beta, C_test.data(),
		    C_test.ld());
	printf("cblas_sgemm multiplication is done\n");

	if (C_out != C_test) {
		printf("Multiplication error\n");
	} else
		printf("Multiplication test PASSED\n");

	// initialize GPU
	ze_driver_handle_t driver = nullptr;
	ze_device_handle_t device = nullptr;
	ze_context_handle_t context = nullptr;
	ze_command_queue_handle_t queue;
	ze_command_list_handle_t commands;
	ze_module_handle_t module;
	ze_kernel_handle_t kernel;

	CHECK(zeInit(ZE_INIT_FLAG_GPU_ONLY));

	// Discover all the driver instances
	uint32_t driverCount = 0;
	CHECK(zeDriverGet(&driverCount, nullptr));
	CHECK2((driverCount == 0), "unable to locate driver(s)");

	ze_driver_handle_t *allDrivers =
		(ze_driver_handle_t *)malloc(driverCount * sizeof(*allDrivers));
	CHECK(zeDriverGet(&driverCount, allDrivers));

	// Find a driver instance with a GPU device
	for (uint32_t i = 0; i < driverCount; ++i) {
		uint32_t deviceCount = 0;
		CHECK(zeDeviceGet(allDrivers[i], &deviceCount, nullptr));
		if (deviceCount == 0)
			continue;
		ze_device_handle_t *allDevices = (ze_device_handle_t *)malloc(
			deviceCount * sizeof(ze_device_handle_t));
		CHECK(zeDeviceGet(allDrivers[i], &deviceCount, allDevices));
		for (uint32_t d = 0; d < deviceCount; ++d) {
			ze_device_properties_t device_properties;
			CHECK(zeDeviceGetProperties(allDevices[d],
						    &device_properties));
			if (ZE_DEVICE_TYPE_GPU == device_properties.type) {
				fprintf(stderr,
					"INFO: GPU device located driver=%d, device=%d\n",
					i, d);
				driver = allDrivers[i];
				device = allDevices[d];
				break;
			}
		}
		if (nullptr != device)
			break;
	}
	CHECK2((driver == nullptr), "unable to locate driver with GPU device");
	CHECK2((device == nullptr), "unable to locate GPU device");

	ze_context_desc_t contextDesc = { ZE_STRUCTURE_TYPE_CONTEXT_DESC,
					  nullptr, 0 };
	CHECK(zeContextCreate(driver, &contextDesc, &context));

	// create a command queue and list
	ze_command_queue_desc_t commandQueueDesc = {
		ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC,
		nullptr,
		0,
		0,
		0,
		ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS,
		ZE_COMMAND_QUEUE_PRIORITY_NORMAL
	};
	CHECK(zeCommandQueueCreate(context, device, &commandQueueDesc, &queue));
	ze_command_list_desc_t commandListDesc = {
		ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC, nullptr, 0, 0
	};
	CHECK(zeCommandListCreate(context, device, &commandListDesc,
				  &commands));

	ze_image_format_t img_fmt = { ZE_IMAGE_FORMAT_LAYOUT_32,
				      ZE_IMAGE_FORMAT_TYPE_FLOAT };
	ze_image_handle_t hAImage;
	ze_image_desc_t desc_A = { ZE_STRUCTURE_TYPE_IMAGE_DESC,
				   nullptr,
				   ZE_IMAGE_FLAG_KERNEL_WRITE,
				   ZE_IMAGE_TYPE_2D,
				   img_fmt,
				   A_in.ld(),
				   A_in.rows(),
				   0,
				   0,
				   0 };
	CHECK(zeImageCreate(context, device, &desc_A, &hAImage));

	ze_image_handle_t hBImage;
	ze_image_desc_t desc_B = { ZE_STRUCTURE_TYPE_IMAGE_DESC,
				   nullptr,
				   ZE_IMAGE_FLAG_KERNEL_WRITE,
				   ZE_IMAGE_TYPE_2D,
				   img_fmt,
				   B_in.ld(),
				   B_in.rows(),
				   0,
				   0,
				   0 };
	CHECK(zeImageCreate(context, device, &desc_B, &hBImage));

	ze_image_handle_t hCImage;
	ze_image_desc_t desc_C = { ZE_STRUCTURE_TYPE_IMAGE_DESC,
				   nullptr,
				   ZE_IMAGE_FLAG_KERNEL_WRITE,
				   ZE_IMAGE_TYPE_2D,
				   img_fmt,
				   C_out_gpu.ld(),
				   C_out_gpu.rows(),
				   0,
				   0,
				   0 };
	CHECK(zeImageCreate(context, device, &desc_C, &hCImage));

	CHECK(zeCommandListAppendImageCopyFromMemory(commands, hAImage, A_in.data(), nullptr,
						     nullptr, 0, nullptr));
	CHECK(zeCommandListAppendImageCopyFromMemory(commands, hBImage, B_in.data(), nullptr,
						     nullptr, 0, nullptr));
	CHECK(zeCommandListAppendImageCopyFromMemory(commands, hCImage, C_out_gpu.data(), nullptr,
						     nullptr, 0, nullptr));

	CHECK(zeCommandListAppendBarrier(commands, nullptr, 0, nullptr));

	// read in and initialize kernel
	FILE *fp = fopen(KERNEL, "rb");
	if (fp == nullptr) {
		fprintf(stderr, "FAIL: unable to open %s\n", KERNEL);
		exit(-1);
	}
	fseek(fp, 0, SEEK_END);
	size_t sz = ftell(fp);
	rewind(fp);

	unsigned char *code = (unsigned char *)malloc(sz);
	size_t ret = fread(code, 1, sz, fp);
	if (ret != sz) {
		if (feof(fp))
			printf("Error reading kernel: unexpected end of file\n");
		else if (ferror(fp)) {
			perror("Error reading kernel\n");
		}
		fclose(fp);
		exit(-1);
	}
	fclose(fp);

	ze_module_desc_t moduleDesc = { ZE_STRUCTURE_TYPE_MODULE_DESC,
					nullptr,
					ZE_MODULE_FORMAT_IL_SPIRV,
					sz,
					code,
					"-vc-codegen",
					nullptr };
	CHECK(zeModuleCreate(context, device, &moduleDesc, &module, nullptr));

	const char *kernel_name = "sgemm_kernel_am";
	ze_kernel_desc_t kernelDesc = { ZE_STRUCTURE_TYPE_KERNEL_DESC, nullptr,
					0, kernel_name };
	CHECK(zeKernelCreate(module, &kernelDesc, &kernel));

	// set group size - single KERNEL_SZ size entry per group
	CHECK(zeKernelSetGroupSize(kernel, /*x*/ 1, /*y*/ 1, /*z*/ 1));

	// unsigned int nThreadsX = ALIGN(c_cols, KERNEL_ALIGN) / 1;
	unsigned int nThreadsX = 1;
	unsigned int nThreadsY = 1;
	// ze_group_count_t groupCount = { nThreadsX, nThreadsY, 1 };
	ze_group_count_t groupCount = { c_cols, c_rows, 1 };

	for (int ic = 0; ic < 1 /* c_rows*/; ic++) {
		for (int jc = 0; jc < 1 /*c_cols*/; jc += 1) {
			/*kernel declarartion
			* sgemm_kernel_am(int m, int n, int k,
			* SurfaceIndex indxA [[type("image2d_t float")]],
			* SurfaceIndex indxB [[type("image2d_t float")]],
			* SurfaceIndex indxC [[type("image2d_t float")]])
			*/
			CHECK(zeKernelSetArgumentValue(kernel, 0, sizeof(a_rows), &a_rows));
			CHECK(zeKernelSetArgumentValue(kernel, 1, sizeof(b_cols), &b_cols));
			CHECK(zeKernelSetArgumentValue(kernel, 2, sizeof(a_cols), &a_cols));

			CHECK(zeKernelSetArgumentValue(kernel, 3, sizeof(hAImage), &hAImage));
			CHECK(zeKernelSetArgumentValue(kernel, 4, sizeof(hBImage), &hBImage));
			CHECK(zeKernelSetArgumentValue(kernel, 5, sizeof(hCImage), &hCImage));

			CHECK(zeCommandListAppendLaunchKernel(
				commands, kernel, &groupCount, nullptr, 0,
				nullptr));
		}
	}

	CHECK(zeCommandListAppendBarrier(commands, nullptr, 0, nullptr));
	// copy result to host
	CHECK(zeCommandListAppendImageCopyToMemory(commands, C_out_gpu.data(), hCImage, nullptr,
						   nullptr, 0, nullptr));
	// CHECK(zeCommandListAppendBarrier(commands, nullptr, 0, nullptr));

	// send to GPU
	CHECK(zeCommandListClose(commands));
	CHECK(zeCommandQueueExecuteCommandLists(queue, 1, &commands, nullptr));

	// think about sync
	CHECK(zeCommandQueueSynchronize(queue, std::numeric_limits<uint32_t>::max()));
	// send to GPU
	//

	// // process output and cleanup
	// CHECK(zeMemFree(context, d_a));
	// CHECK(zeMemFree(context, d_b));
	// CHECK(zeMemFree(context, d_c));

	// // verify results
	// for (unsigned i = 0; i < SZ; i++)
	// 	if ((src1[i] + src2[i]) != dst[i]) {
	// 		fprintf(stderr,
	// 			"FAIL: comparison at index[%d]: %d + %d => %d(host), but %d(gpu)\n",
	// 			i, src1[i], src2[i], (src1[i] + src2[i]),
	// 			dst[i]);
	// 		exit(-1);
	// 	}
	//
	if (C_out_gpu != C_test) {
		printf("GPU Multiplication error\n");
	} else
		printf("GPU Multiplication test PASSED\n");

	zeImageDestroy(hAImage);
	zeImageDestroy(hBImage);
	zeImageDestroy(hCImage);

	zeCommandListDestroy(commands);
	zeContextDestroy(context);

	printf("done\n");

	return 0;
}
