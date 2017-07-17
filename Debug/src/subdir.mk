################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../src/BeamFormer.cu 

CU_DEPS += \
./src/BeamFormer.d 

OBJS += \
./src/BeamFormer.o 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-8.0/bin/nvcc -I"/usr/local/cuda-8.0/samples/0_Simple" -I"/usr/local/cuda-8.0/samples/common/inc" -I"/home/amelis/cuda-workspace/test_pharos" -G -g -O0 -gencode arch=compute_52,code=sm_52  -odir "src" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-8.0/bin/nvcc -I"/usr/local/cuda-8.0/samples/0_Simple" -I"/usr/local/cuda-8.0/samples/common/inc" -I"/home/amelis/cuda-workspace/test_pharos" -G -g -O0 --compile --relocatable-device-code=false -gencode arch=compute_52,code=compute_52 -gencode arch=compute_52,code=sm_52  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


