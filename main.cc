#include <utility>
#include <CL/cl.h>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <iterator>
#include <OpenImageIO/imageio.h>
#include <memory>
#include <cstdint>

OIIO_NAMESPACE_USING

const std::string kernels_file{"kernels.cl"};

inline void checkErr(cl_int err, const char *name) {
  if (err != CL_SUCCESS) {
    std::cerr << "ERROR: " << name << " (" << err << ")" << std::endl;
    exit(EXIT_FAILURE);
  }
}

cl_program compile_program(cl_context context, cl_uint devices_count,
                           cl_device_id *devices,
                           const std::string &source_filename) {
  cl_int err;
  std::ifstream file(source_filename);
  checkErr(file.is_open() ? CL_SUCCESS : -1, source_filename.c_str());
  std::string prog(std::istreambuf_iterator<char>(file),
                   (std::istreambuf_iterator<char>()));
  const char *source_str = prog.c_str();
  cl_program program =
      clCreateProgramWithSource(context, 1, &source_str, NULL, &err);
  err = clBuildProgram(program, devices_count, devices, NULL, NULL, NULL);
  if (err != CL_SUCCESS) {
    size_t buf_size;
    clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, 0, NULL,
                          &buf_size);
    char *buf = (char *)malloc(buf_size);
    clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, buf_size,
                          buf, NULL);
    std::cout << "Build log: " << std::endl << buf << std::endl;
  }
  checkErr(err, "clBuildProgram");

  return program;
}

std::tuple<std::vector<float>, int, int, int>
read_image(const std::string &filename) {
  std::unique_ptr<ImageInput> in{ImageInput::open(filename)};
  const ImageSpec &spec = in->spec();
  int x = spec.width;
  int y = spec.height;
  int c = spec.nchannels;
  std::vector<float> pixels(x * y * c);
  in->read_image(TypeDesc::FLOAT, pixels.data());
  in->close();
  return std::make_tuple(std::move(pixels), x, y, c);
}

void write_image(const std::string &filename, int x, int y, int c, void *data) {
  std::unique_ptr<ImageOutput> out{ImageOutput::create(filename)};
  if (!out)
    return;
  ImageSpec spec(x, y, c, TypeDesc::FLOAT);
  out->open(filename, spec);
  out->write_image(TypeDesc::FLOAT, data);
  out->close();
}

int main(int argc, char **argv) {
  cl_int err;
  cl_uint platform_count;
  clGetPlatformIDs(0, NULL, &platform_count);
  cl_platform_id *platforms =
      (cl_platform_id *)malloc(sizeof(cl_platform_id) * platform_count);
  clGetPlatformIDs(platform_count, platforms, NULL);
  checkErr(platform_count != 0 ? CL_SUCCESS : -1, "clGetPlatformIDs");

  if (argc < 6) {
    std::cerr << "Usage: <platform_id> <mode> <imageA> <imageB> <imageOut>"
              << std::endl;
    return EXIT_FAILURE;
  }

  int platformId = std::stoi(argv[1]);
  char *mode = argv[2];
  char *image_a_filename = argv[3];
  char *image_b_filename = argv[4];
  char *image_out_filename = argv[5];

  if (platformId < 0 || platformId >= platform_count) {
    std::cerr << "Platform index out of bounds" << std::endl;
    return EXIT_FAILURE;
  }

  if (strcmp(mode, "multiply") && strcmp(mode, "screen") &&
      strcmp(mode, "normal") && strcmp(mode, "overlay") &&
      strcmp(mode, "darken") && strcmp(mode, "lighten") &&
      strcmp(mode, "color_dodge") && strcmp(mode, "hard_light") &&
      strcmp(mode, "soft_light") && strcmp(mode, "difference") &&
      strcmp(mode, "exclusion")) {
    std::cerr << "Mode unsupported" << std::endl;
    return EXIT_FAILURE;
  }

  for (int p = 0; p < platform_count; p++) {
    cl_device_id *devices;
    cl_uint device_count;
    clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_ALL, 0, NULL, &device_count);
    devices = (cl_device_id *)malloc(sizeof(cl_device_id) * device_count);
    clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_ALL, device_count, devices,
                   NULL);

    for (int i = 0; i < device_count; i++) {
      size_t name_size;
      clGetDeviceInfo(devices[i], CL_DEVICE_NAME, 0, NULL, &name_size);
      char *name = (char *)malloc(name_size);
      clGetDeviceInfo(devices[i], CL_DEVICE_NAME, name_size, name, NULL);
      printf("platform=%d. device%d=%s\n", p, i, name);
    }
  }

  cl_uint device_count;
  err = clGetDeviceIDs(platforms[platformId], CL_DEVICE_TYPE_ALL, 0, NULL,
                       &device_count);
  checkErr(err, "clGetDeviceIDs");
  cl_device_id *devices =
      (cl_device_id *)malloc(sizeof(cl_device_id) * device_count);
  err = clGetDeviceIDs(platforms[platformId], CL_DEVICE_TYPE_ALL, device_count,
                       devices, NULL);
  checkErr(err, "clGetDeviceIDs");

  {
    bool sup;
    clGetDeviceInfo(devices[0], CL_DEVICE_IMAGE_SUPPORT, 1, &sup, NULL);
    std::cout << "Are images supported by the device:  " << sup << std::endl;
  }

  cl_context_properties cprops[3] = {
      CL_CONTEXT_PLATFORM, (cl_context_properties)(platforms[platformId]), 0};
  cl_context context =
      clCreateContext(cprops, device_count, devices, NULL, NULL, &err);
  checkErr(err, "clCreateContext");

  std::vector<float> imageData;
  int x, y, c;
  std::tie(imageData, x, y, c) = read_image(image_a_filename);
  size_t bufferLength = sizeof(float) * imageData.size();

  cl_mem imageA =
      clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                     bufferLength, imageData.data(), &err);
  checkErr(err, "clCreateBuffer");

  std::vector<float> imageBData;
  std::tie(imageBData, std::ignore, std::ignore, std::ignore) =
      read_image(image_b_filename);

  cl_mem imageB =
      clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                     bufferLength, imageBData.data(), &err);
  checkErr(err, "clCreateBuffer");

  float *imageOut = new float[bufferLength];
  cl_mem outCL =
      clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR,
                     bufferLength, imageOut, &err);
  checkErr(err, "clCreateBuffer");

  auto program = compile_program(context, device_count, devices, kernels_file);

  std::string kernel_name{"blend_"};
  kernel_name += mode;
  cl_kernel kernel = clCreateKernel(program, kernel_name.c_str(), &err);
  checkErr(err, "clCreateKernel");
  err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &imageA);
  checkErr(err, "clSetKernelArg");
  err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &imageB);
  checkErr(err, "clSetKernelArg");
  err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &outCL);
  checkErr(err, "clSetKernelArg");

  cl_command_queue queue = clCreateCommandQueue(context, devices[0], 0, &err);
  checkErr(err, "clCreateCommandQueue");

  cl_event event;
  size_t global_item_size = x * y * c;
  size_t local_item_size = 1;
  err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_item_size,
                               &local_item_size, 0, NULL, &event);

  checkErr(err, "clEnqueueNDRangeKernel");

  clWaitForEvents(0, &event);
  err = clEnqueueReadBuffer(queue, outCL, CL_TRUE, 0, bufferLength, imageOut, 0,
                            NULL, 0);
  checkErr(err, "clEnqueueReadBuffer");

  write_image(image_out_filename, x, y, c, imageOut);

  return EXIT_SUCCESS;
}
