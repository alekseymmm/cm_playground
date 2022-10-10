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

#define SZ 160
#define KERNEL_SZ 16
#define CHECK(a) do { \
    auto err = (a); \
    if (err != 0) { \
        fprintf(stderr, "FAIL: err=%d @ line=%d (%s)\n", err, __LINE__, (#a)); \
        exit(err); \
    } \
}while (0)
#define CHECK2(a, msg) do { \
    if ((a)) { \
        fprintf(stderr, "FAIL: @ line=%d (%s)\n", __LINE__, (msg)); \
        exit(-1); \
    } \
}while (0)
#ifndef KERNEL
#error "Error: KERNEL must be defined with location of kernel binary"
#endif

int main(int argc, char* argv[])
{
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

	ze_kernel_desc_t kernelDesc = { ZE_STRUCTURE_TYPE_KERNEL_DESC, nullptr,
					0, "hello_world" };
	CHECK(zeKernelCreate(module, &kernelDesc, &kernel));

	uint32_t suggested_group_size_x, suggested_group_size_y,
		suggested_group_size_z;
	zeKernelSuggestGroupSize(kernel, 0, 0, 0, &suggested_group_size_x,
				 &suggested_group_size_y,
				 &suggested_group_size_z);
	printf("suggested_group_size_x=%u, suggested_group_size_y=%u, suggested_group_size_z=%u\n",
	       suggested_group_size_x, suggested_group_size_y,
	       suggested_group_size_z);
	int threadwidth = 8;
	CHECK(zeKernelSetArgumentValue(kernel, 0, sizeof(int), &threadwidth));

	// set group size - single KERNEL_SZ size entry per group
	// CHECK(zeKernelSetGroupSize(kernel, /*x*/ 1, /*y*/ 1, /*z*/ 1));
	// CHECK(zeKernelSetGroupSize(kernel,
	// 			   /*x*/ suggested_group_size_x,
	// 			   /*y*/ suggested_group_size_y,
	// 			   /*z*/ suggested_group_size_z));

	// launch - data split across multiple groups
	// ze_group_count_t groupCount = { 8, 1, 1 };

	ze_group_count_t groupCount = { suggested_group_size_x,
					suggested_group_size_y,
					suggested_group_size_z };
	CHECK(zeCommandListAppendLaunchKernel(commands, kernel, &groupCount,
					      nullptr, 0, nullptr));

	CHECK(zeCommandListAppendBarrier(commands, nullptr, 0, nullptr));

	// send to GPU
	CHECK(zeCommandListClose(commands));
	CHECK(zeCommandQueueExecuteCommandLists(queue, 1, &commands, nullptr));

	fprintf(stderr, "PASSED\n");
	return 0;
}
