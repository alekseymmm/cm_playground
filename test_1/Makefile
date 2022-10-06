PLATFORM=SKL
PLATFORM_EXTENSION=skl

CSDK_DIR ?= /home/amarov/devel/intel/cm_sdk_20211028/
CMC ?= $(CSDK_DIR)/usr/bin/cmc

KERN_CPP := kernel.cpp
KERN_BASENAME := $(basename ${KERN_CPP})
KERN_NAME := ${KERN_BASENAME}.spv.${PLATFORM_EXTENSION}

HOST_CPP := host_l0.cpp

APP := main.l0.${PLATFORM_EXTENSION}

all: ${APP}

${KERN_NAME}: ${KERN_CPP}
	${CMC} -fcmocl -march=${PLATFORM} -emit-spirv -m64 \
	-o ${KERN_NAME} -- ${KERN_CPP}

	echo "kenel ${KERN_NAME} is generated"

kernel: ${KERNEL_NAME}

${APP}: ${HOST_CPP} ${KERN_NAME}
	g++ -m64 -DKERNEL=\"${KERN_NAME}\" \
		-I${CSDK_DIR}/usr/include \
		-L${CSDK_DIR}/usr/lib/x86_64-linux-gnu -Wl,-rpath \
		-Wl,${CSDK_DIR}/usr/lib/x86_64-linux-gnu  \
		 ${HOST_CPP} -lze_loader -o ${APP}

clean:
	rm -f *.o *.skl ${APP}

.PHONY: clean, all, kernel
