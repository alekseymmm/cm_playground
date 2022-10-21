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

static float *allocMatrix(int rows, int cols)
{
	float *a = NULL;
	uint32_t rows_aligned = ALIGN(rows, KERNEL_ALIGN);
	uint32_t cols_aligned = ALIGN(cols, KERNEL_ALIGN);

	size_t size = sizeof(float) * rows_aligned * cols_aligned;

	a = (float *)aligned_alloc(4096, size);
	if (!a)
		printf("failed to allocate matrix of size %lu\n", size);

	return a;
}

static float randData(float low, float high)
{
	float t = (float)rand() / (float)RAND_MAX;
	return (1.0f - t) * low + t * high;
}

static void initMatrix(float *a, int rows, int cols)
{
	for (int i = 0; i < rows; i++)
		for (int j = 0; j < cols; j++)
			a[i * cols + j] = randData(0.0f, 1.0f);
}

// C := alpha*A*B + beta*C,
// A(m x k) , B(k x n) , C(m x n)
static int sgemmNxN(int m, int n, int k, float alpha, float *A, int lda,
		    float *B, int ldb, float beta, float *C, int ldc)
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

#define CORRECTNESS_THRESHOLD 0.00002
bool matrixCmp(float *A, float *B, int rows, int cols)
{
	double max_relerror = 0.0;
	double max_abserror = 0.0;
	for (int r = 0; r < rows; r++)
		for (int c = 0; c < cols; c++) {
			// printf("I=%3d N=%3d  %08x  %08x\n", r, c, *(unsigned int*)&(*this)(r,c), *(unsigned int *)&m(r,c));

			double relerror =
				fabs(A[r * rows + c] - B[r * rows + c]) /
				max(fabs(A[r * rows + c]),
				    fabs(B[r * rows + c]));
			double abserror =
				fabs(A[r * rows + c] - B[r * rows + c]);

			max_relerror = max(max_relerror, relerror);
			max_abserror = max(max_abserror, abserror);

			if (relerror > CORRECTNESS_THRESHOLD) {
				printf("Failure %f %f relerror: %lf at [%d, %d]\n",
				       A[r * rows + c], B[r * rows + c],
				       relerror, r, c);
				return false;
			}
		}
	printf("max_relerror = %e  absolute error = %e\n", max_relerror,
	       max_abserror);
	return (max_relerror > CORRECTNESS_THRESHOLD) ? false : true;
	return true;
}

int main(int argc, char *argv[])
{
	uint32_t a_rows = 15, a_cols = 15;
	uint32_t b_rows = 15, b_cols = 15;
	uint32_t c_rows = 15, c_cols = 15;

	float *A_in = NULL, *B_in = NULL, *C_out = NULL, *C_gold = NULL;

	A_in = allocMatrix(a_rows, a_cols);
	B_in = allocMatrix(b_rows, b_cols);
	C_out = allocMatrix(c_rows, c_cols);
	C_gold = allocMatrix(c_rows, c_cols);

	if (!A_in || !B_in || !C_out || !C_gold) {
		free(A_in);
		free(B_in);
		free(C_out);
		free(C_gold);
		return -ENOMEM;
	}

	initMatrix(A_in, a_rows, a_cols);
	initMatrix(B_in, b_rows, b_cols);
	initMatrix(C_out, c_rows, c_cols);
	memcpy(C_gold, C_out, c_rows * c_cols * sizeof(float));

	float alpha = +1.0, beta = +1.0;

	sgemmNxN(a_rows, b_cols, b_rows, alpha, A_in, a_cols, B_in, b_cols,
		 beta, C_out, c_cols);
	printf("sgemmNxN multiplication is done\n");

	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, a_rows, b_cols,
		    b_rows, alpha, A_in, a_cols, B_in, b_cols, beta, C_gold,
		    c_cols);
	printf("cblas_sgemm multiplication is done\n");

	if (!matrixCmp(C_out, C_gold, c_rows, c_cols)) {
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
				   ALIGN(a_cols, KERNEL_ALIGN),
				   ALIGN(a_rows, KERNEL_ALIGN),
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
				   ALIGN(b_cols, KERNEL_ALIGN),
				   ALIGN(b_rows, KERNEL_ALIGN),
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
				   ALIGN(c_cols, KERNEL_ALIGN),
				   ALIGN(c_rows, KERNEL_ALIGN),
				   0,
				   0,
				   0 };
	CHECK(zeImageCreate(context, device, &desc_C, &hCImage));

	CHECK(zeCommandListAppendImageCopyFromMemory(
		commands, hAImage, &A_in, nullptr, nullptr, 0, nullptr));
	CHECK(zeCommandListAppendImageCopyFromMemory(
		commands, hBImage, &B_in, nullptr, nullptr, 0, nullptr));
	CHECK(zeCommandListAppendImageCopyFromMemory(
		commands, hCImage, &C_out, nullptr, nullptr, 0, nullptr));

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
	fread(code, 1, sz, fp);
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

	unsigned int nThreadsX = ALIGN(c_cols, KERNEL_ALIGN) / 16;
	unsigned int nThreadsY = c_rows;
	ze_group_count_t groupCount = { nThreadsX, nThreadsY, 1 };

	for (int ic = 0; ic < c_rows; ic++) {
		for (int jc = 0; jc < c_rows; jc += 16) {
			/*kernel declarartion
			* sgemm_kernel_am(int m, int n, int k, int ic, int jc,
			* SurfaceIndex indxA [[type("image2d_t float")]],
			* SurfaceIndex indxB [[type("image2d_t float")]],
			* SurfaceIndex indxC [[type("image2d_t float")]])
			*/
			CHECK(zeKernelSetArgumentValue(
				kernel, 0, sizeof(a_rows), &a_rows));
			CHECK(zeKernelSetArgumentValue(
				kernel, 1, sizeof(b_cols), &b_cols));
			CHECK(zeKernelSetArgumentValue(
				kernel, 2, sizeof(b_rows), &b_rows));
			CHECK(zeKernelSetArgumentValue(kernel, 3, sizeof(ic),
						       &ic));
			CHECK(zeKernelSetArgumentValue(kernel, 4, sizeof(jc),
						       &jc));
			CHECK(zeKernelSetArgumentValue(
				kernel, 5, sizeof(hAImage), &hAImage));
			CHECK(zeKernelSetArgumentValue(
				kernel, 6, sizeof(hBImage), &hBImage));
			CHECK(zeKernelSetArgumentValue(
				kernel, 7, sizeof(hCImage), &hCImage));

			CHECK(zeCommandListAppendLaunchKernel(
				commands, kernel, &groupCount, nullptr, 0,
				nullptr));
		}
	}

	CHECK(zeCommandListAppendBarrier(commands, nullptr, 0, nullptr));
	// copy result to host
	CHECK(zeCommandListAppendImageCopyToMemory(
		commands, C_out, hCImage, nullptr, nullptr, 0, nullptr));

	// send to GPU
	CHECK(zeCommandListClose(commands));
	CHECK(zeCommandQueueExecuteCommandLists(queue, 1, &commands, nullptr));

	//think about sync
	//    CHECK(zeCommandQueueSynchronize(hCommandQueue,
	//	std::numeric_limits<uint32_t>::max()));
	// // send to GPU
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
	fprintf(stderr, "PASSED\n");

	zeImageDestroy(hAImage);
	zeImageDestroy(hBImage);
	zeImageDestroy(hCImage);

	zeCommandListDestroy(commands);
	zeContextDestroy(context);

	free(A_in);
	free(B_in);
	free(C_out);
	free(C_gold);
	return 0;
}
