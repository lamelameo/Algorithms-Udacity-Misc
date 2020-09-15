""" Using CUDA to try speed up a simple image scan """

import numpy as np
import time
import cv2 as cv

import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
from pycuda.compiler import DynamicSourceModule
import pycuda.gpuarray as gpuarray
from pycuda import tools

# print(tools.DeviceData().warp_size, tools.DeviceData().thread_blocks_per_mp)

# Threads within blocks are done parallel with thread "warps", and each block is done in parallel based on #SMMs
# thus we must make blocks of rows, not columns, as we want to go through row in order, with multiple rows in parallel
# nvidia940m has 3 smm i think, so i need to split up blocks so rows are scanned sequentially...
# I have 1918x1 blocks so only every 3rd pixel column is going to the same core...but if i make block more than 1 in x
# direction, then threads will become parallel. Have to arrange blocks 0,1,2 down y dir =284 then all block will be
# same row in same SMM, but how to arrange like this
# TODO: SMMs are not queued sequentially based on #blocks, can go out of order, ruining the algorithm
# Try split image into chunks to be handed to GPU, then process again in CPU
# EG: cut rows into chunks, and determine longest contiguous sequence of pixels within bgr threshold for the chunk
# saving seqs starting at start, ending at end, then we have to go through all chunks, using GPU and determining whether
# we can splice together contiguous sequences to get at least a 50 pixel sequence eg. chunk i ends in 30 seq, chunk i+1
# starts in 30 seq, then when we check either of these chunks, we get a total sequence of length 60.
# we could do it like this with each kernel doing a small loop of x pixels OR
# TODO: could do it like the sum scan, transform image into 1 or 0 if pixel meets threshold, then we add val to next
# val, if it isnt 0, then we do 2 vals away, etc till 6 vals away which tells us if we have same value as 2^6 vals away
# ie 64 pix away, if so then we both have


def parrallel_scan():
    # Array stored in memory in linear fashion, row by row, col by col, pix/bgr ie r0c0pb, r0c0pg, r0c0pr, r0c1pb
    # We should parallelize blocks of ~284y 1x 1z, keep array[853] to keep counter for each row, then we will be
    # scanning scanning across the image column by column, keeping counter for each row
    image = cv.imread("test images/crop2.png")
    modtest = SourceModule("""
    #include <stdio.h>
    __global__ void line_scan(const unsigned char *img, int *counter)
    {
        //int x = blockIdx.x;
        //int y = threadIdx.y + blockIdx.y * blockDim.y;
        // transposed image height and width against x and y to get threads to run down then across columns of image
        int x = blockIdx.y;
        int y = 852 - (threadIdx.x + (blockIdx.x * 288));
        if(y < 0) { return; }
        // attempt at removing thread divergence, doesnt seem to make any difference in speed
        //int bgr_pass = (img[x*3 + y*1918*3] <= 4)*(153 <= img[1 + x*3 + y*1918*3])*(img[1 + x*3 + y*1918*3] <= 180)*
        //(196 <= img[2 + x*3 + y*1918*3])*(img[2 + x*3 + y*1918*3] <= 210);
        //counter[y] = counter[y]*bgr_pass + bgr_pass;
        if((img[x*3 + y*1918*3] <= 4) && (153 <= img[1 + x*3 + y*1918*3]) && (img[1 + x*3 + y*1918*3] <= 180) 
        && (196 <= img[2 + x*3 + y*1918*3]) && (img[2 + x*3 + y*1918*3] <= 210)) {
            counter[y] += 1;
            if(counter[y] == 50) {
                counter[853] = x;
                counter[854] = y;
            }
        } else { counter[y] = 0; }
        __syncthreads();
    }
    """)
    func = modtest.get_function("line_scan")

    # TODO: added syncing of threads before they finish to try keep columns in sequence...not guaranteed to work?
    # blocks can still run out of order due to SMMs eg 50 line: last+1 line resets counter before last line is scanned
    test = np.array([[[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]],
                     [[12, 13, 14], [15, 16, 17], [18, 19, 20], [21, 22, 23]]])
    counter = np.array([0 for _ in range(855)])
    image_gpu = gpuarray.to_gpu_async(image)
    # counter_gpu = gpuarray.to_gpu_async(counter)
    # image_gpu = cuda.mem_alloc(image.nbytes)
    counter_gpu = cuda.mem_alloc(counter.nbytes)
    # cuda.memcpy_htod(image_gpu, image)
    cuda.memcpy_htod(counter_gpu, counter)
    timer = time.clock()
    func(image_gpu, counter_gpu, block=(288, 1, 1), grid=(3, 1918))
    print(time.clock() - timer)
    cuda.memcpy_dtoh(counter, counter_gpu)
    print(counter[-2:])


# parrallel_scan()


def row_scan():
    # scan full rows each thread, multiple rows in parallel
    # TODO: this is bad for memory coalescing?

    # int bgr_pass = (img[x * 3 + y * 1918 * 3] <= 4) * (153 <= img[1 + x * 3 + y * 1918 * 3]) * (
    #           img[1 + x * 3 + y * 1918 * 3] <= 180) *
    # (196 <= img[2 + x * 3 + y * 1918 * 3]) * (img[2 + x * 3 + y * 1918 * 3] <= 210);
    # counter = counter * bgr_pass + bgr_pass;
    startscan = SourceModule("""
    #include <stdio.h>
    __global__ void line_scan(unsigned char *img, int *flag)
    {
        int y = threadIdx.y + blockIdx.y * blockDim.y;
        int counter = 0;
        if(y > 852) { return; }
        for(int x=0; x<1918; x++) {
            int pix_g = img[1 + x*3 + y*1918*3];
            int pix_r = img[2 + x*3 + y*1918*3];
            if((img[x*3 + y*1918*3] < 5) & (152 < pix_g) & (pix_g < 181)
            & (195 < pix_r) & (pix_r < 211)) {
                counter++;
                if(counter == 50) {
                    flag[0] = x;
                    flag[1] = y;
                }
            } else { counter = 0; }
        }
    }
    """)

    image = cv.imread("test images/crop2.png")
    scan = startscan.get_function("line_scan")
    timer = time.clock()
    flag = np.array([0, 0])
    image_gpu = gpuarray.to_gpu_async(image)
    flag_gpu = gpuarray.to_gpu_async(flag)

    scan(image_gpu, flag_gpu, block=(1, 32, 1), grid=(1, 27))
    print(flag_gpu.get())
    print(time.clock() - timer)
    # print(flag)


# row_scan()


def full_scan():
    # TODO: testing how slow a single full scan is with no parallelism
    sequential = SourceModule("""
        #include <stdio.h>
        __global__ void full_scan(unsigned char *img, int line[2])
        {
            int counter = 0;
            for(int y=0; y<853; y++) {
                for(int x=0; x<1918; x++) {
                    if((img[x*3 + y*1918*3] <= 4) && (153 <= img[1 + x*3 + y*1918*3]) && (img[1 + x*3 + y*1918*3] <= 180)
                    && (196 <= img[2 + x*3 + y*1918*3]) && (img[2 + x*3 + y*1918*3] <= 210)) {
                        counter++;
                        if(counter == 50) {
                            line[0] = x;
                            line[1] = y;
                            return;
                        }
                    } else { counter = 0; }
                }
            }

        }
        """)

    image = cv.imread("test images/crop2.png")
    seq = sequential.get_function("full_scan")
    image_gpu = gpuarray.to_gpu_async(image)
    line = np.array([0, 0])
    timer = time.clock()
    seq(image_gpu, cuda.InOut(line), block=(1, 1, 1))
    print(time.clock() - timer)
    print(line)


def sequential_col_scan():
    # TODO: can add a main function to cuda code to scan manually across cols, instead of doing in python?

    cuda_string = """
    #include <stdio.h>
    __global__ void line_scan(unsigned char *img, int *counter, const int x, int flag[1])
    {
        int y = threadIdx.y + blockIdx.y * blockDim.y;
        //if(y > 852) { return; }
        //int bgr_pass = (img[x*3 + y*1918*3] <= 4)*(153 <= img[1 + x*3 + y*1918*3])*(img[1 + x*3 + y*1918*3] <= 180)*
        //(196 <= img[2 + x*3 + y*1918*3])*(img[2 + x*3 + y*1918*3] <= 210);
        //counter[y] = counter[y]*bgr_pass + bgr_pass;
        if(y < 853) {
            if((img[x*3 + y*1918*3] <= 4) && (153 <= img[1 + x*3 + y*1918*3]) && (img[1 + x*3 + y*1918*3] <= 180)
            && (196 <= img[2 + x*3 + y*1918*3]) && (img[2 + x*3 + y*1918*3] <= 210)) {
                counter[y] += 1;
                if(counter[y] == 50) {
                    flag[0] = y;
                }
            } else { counter[y] = 0; }
        }
    }
    
    __device__ void start_scan(unsigned char *img, int *counter, int flag[1])
    {   
        dim3 blocks = (1, 1, 1);
        dim3 threads = (1, 853, 1);
        //for(int i=0; i<853; i++) {counter[i] = 0;}
        for(int col=0; col<1918; col++) {
            line_scan<<<blocks, threads>>>(img, counter, col, flag);
        }
    }
    """

    scantest = SourceModule("""
    #include <stdio.h>
    __global__ void line_scan(unsigned char *img, int *counter, const int x, int flag[1])
    {
        int y = threadIdx.y + blockIdx.y * blockDim.y;
        //if(y > 852) { return; }
        //int bgr_pass = (img[x*3 + y*1918*3] <= 4)*(153 <= img[1 + x*3 + y*1918*3])*(img[1 + x*3 + y*1918*3] <= 180)*
        //(196 <= img[2 + x*3 + y*1918*3])*(img[2 + x*3 + y*1918*3] <= 210);
        //counter[y] = counter[y]*bgr_pass + bgr_pass;
        if(y < 853) {
            if((img[x*3 + y*1918*3] <= 4) && (153 <= img[1 + x*3 + y*1918*3]) && (img[1 + x*3 + y*1918*3] <= 180)
            && (196 <= img[2 + x*3 + y*1918*3]) && (img[2 + x*3 + y*1918*3] <= 210)) {
                counter[y] += 1;
                if(counter[y] == 50) {
                    flag[0] = y;
                }
            } else { counter[y] = 0; }
        }
    }
    """)

    image = cv.imread("test images/crop2.png")
    counter = np.zeros(853, np.int32)
    flag = np.array([0])
    # scantest.get_function("line_scan")  # , options=["-rdc=true", "-lcudadevrt", "-lcublas_device"
    start = DynamicSourceModule(cuda_string)
    scancol = start.get_function("start_scan")
    image_gpu = gpuarray.to_gpu_async(image)
    # start(image_gpu)
    # test = gpuarray.take()
    # counter_gpu = gpuarray.to_gpu_async(counter)
    # flag_gpu = cuda.mem_alloc(flag.nbytes)
    counter_gpu = cuda.mem_alloc(counter.nbytes)
    cuda.memcpy_htod(counter_gpu, counter)
    # cuda.memcpy_htod(flag_gpu, flag)
    timer = time.clock()
    # stream = cuda.Stream()
    # TODO: as we are scanning columns, is it faster to transpose array to col major order in memory first?
    scancol(image_gpu, counter_gpu, flag, block=(1, 1, 1))
    print(time.clock() - timer)
    quit()
    for i in range(1918):
        scancol(image_gpu, counter_gpu, np.uintc(i), cuda.InOut(flag), block=(1, 32, 1), grid=(1, 28))
        # context.synchronize()
        if flag[0] > 0:
            print(time.clock() - timer)
            print(i, flag[0])
            return
    print(time.clock() - timer)


# sequential_col_scan()


# parallel thresholding function, generates 0 or 1 for every pixel pass or fail bgr test
# now we have a 1918x853 binary matrix, to calc subsequences of x passes, we just take original matrix and take matrix
# that has been shifted left, then increment counter for that pixel if they are both 1, or make -1 otherwise
# if any counter reaches 50, then we have a 50 length line starting at that pixel
def matrix_scan():

    binary_matrix = SourceModule("""
    #include <stdio.h>
    __global__ void convert_binary(const unsigned char *img, int *matrix)
    {
        //__shared__ int pix[384];
        int thx = threadIdx.x;
        int b_len = blockIdx.x * blockDim.x;
        int xI = b_len + thx;
        int yI = threadIdx.y + blockIdx.y * blockDim.y;
        if((xI > 1918) || (yI > 852)) { return; }
        
        // first thread in block loads bgr thresholds and matrix items to be accessed
        //if(thx == 0) {
            //for(int i=0; i<384; i++) {
                //pix[i] = img[i + b_len*3 + yI*1919*3];
            //}
        //}
        //__syncthreads();
        
        // Check if bgr all passed, set 1 if so, else 0
        int index = xI*3 + yI*1919*3;
        int b = img[index];
        int g = img[1 + index];
        int r = img[2 + index];
        if ((b < 5) && (152 < g) && (g < 181) && (195 < r) && (r < 211)) {
            matrix[xI + yI*1919] = 1;
        }

        //int bgr_pass = 1;
        //#pragma unroll
        //for(int i=0; i<3; i++) {
            //const int pixel = img[i + xI*3 + yI*1919*3];
            //bgr_pass *= (rgb[i*2] < pixel) * (pixel < rgb[i*2 + 1]);
            //bgr_pass *= (rgb[i*2] < pix[thx*3 + i]) * (rgb[i*2 + 1] > pix[thx*3 + i]);
        //}
        //matrix[xI + yI*1919] = bgr_pass;
    }
    """)

    binary_matrix2 = SourceModule("""
    #include <stdio.h>
    __global__ void convert_binary2(const unsigned char *img, int *matrix) 
    {
        int x = threadIdx.x + blockIdx.x*blockDim.x;
        int y = threadIdx.y + blockIdx.y*blockDim.y;
        //int z = threadIdx.z;
        if((x > 5756) || (y > 852)) { return; }
        int rgb[6] = {-1, 5, 152, 181, 195, 211};
        int z = threadIdx.x % 3;
        int pixel = img[x + y*5757];
        // process bgr values in parallel using blocks with dim.z = 3
        //int pixel = img[z + x*3 + y*1919*3];
        int bgr_pass = (rgb[z*2] < pixel) & (pixel < rgb[z*2 + 1]);
        atomicAnd(&matrix[int(x/3) + y*1919], bgr_pass);
    }
    """)

    scan = SourceModule("""
    #include <stdio.h>
    __global__ void scan(int *matrix, int *pix)
    {
        int counter = 0;
        const int x = threadIdx.x + blockIdx.x*blockDim.x;
        const int y = blockIdx.y;
        if((x > 1868) || (y > 852)) { return; }
        // check for 50 contiguous passed values
        #pragma unroll
        for(int i=1; i<51; i++) {
            counter += matrix[i + x + y*1919];
            //counter = counter * matrix[i + x + y*1919] + matrix[i + x + y*1919];
        }
        if(counter == 50){
            pix[0] = x;
            pix[1] = y;
        }
    }
    """)

    # need to detect line of length ~50-100, can scan fast by powers of 2, 2^6 = 64
    # loop 6 times: check index 2^i away, val = val * val[index]
    # binary_scan = SourceModule("""
    #     #include <stdio.h>
    #     __global__ void scan(int *matrix, int *counter, int *pix)
    #     {
    #         int x = threadIdx.x + blockIdx.x*blockDim.x;
    #         int y = threadIdx.y + blockIdx.y*blockDim.y;
    #         if((x > 1867) || (y > 852)) {return;}
    #
    #         for(int i=0; i<6; i++) {
    #             counter[x + y*1919] += matrix[i + x + y*1919];
    #         }
    #         if(counter[x + y*1919] >= 50){
    #             pix[0] = x;
    #             pix[1] = y;
    #         }
    #     }
    # """)

    image = cv.imread("test images/crop3.png")
    binary = binary_matrix.get_function("convert_binary")
    detect = scan.get_function("scan")
    binary2 = binary_matrix2.get_function("convert_binary2")
    timer = time.clock()
    b_matrix = np.ones((853, 1919), np.uintc)
    # b_matrix = np.zeros((853, 1919), np.int32)
    rgb = np.array([-1, 5, 152, 181, 195, 211], np.int32)
    # mtx = cuda.managed_empty(shape=10, dtype=np.float32, mem_flags=cuda.mem_attach_flags.GLOBAL)
    # TODO: pinned, mapped memory
    image_gpu = gpuarray.to_gpu_async(image)
    matrix_gpu = gpuarray.to_gpu(b_matrix)
    # rgb_gpu = gpuarray.to_gpu(rgb)

    # binary.prepare([np.ubyte, np.int32, np.int32])
    # binary.prepared_call((120, 54), (16, 16, 3), image_gpu, matrix_gpu, rgb_gpu)
    # TODO: blocks should be chunks of rows, as memory is layout out in row major order
    # binary(image_gpu, matrix_gpu, rgb_gpu, block=(480, 1, 1), grid=(4, 853))
    binary2(image_gpu, matrix_gpu, block=(120, 1, 1), grid=(48, 853))
    print("\nconvert to binary took:", time.clock() - timer, "secs")

    pix = np.array([0, 0], np.int32)
    pix_gpu = gpuarray.to_gpu(pix)
    timer = time.clock()
    detect(matrix_gpu, pix_gpu, block=(480, 1, 1), grid=(4, 853))
    print("\nscan took:", time.clock() - timer, "secs")
    pix_gpu.get(pix)
    print(pix)
    # print(matrix_gpu[pix[1]][pix[0] - 30:pix[0] + 80])
    # print(matrix_gpu[549][1000:1200])
    print(matrix_gpu[321][700:900])


# matrix_scan()


def chunk_scan():
    # TODO: splitting rows into chunks of 64...since our line is ~140 pixels length, worst case positioning we are
    # still guaranteed a contiguous chunk of 64, so we can just have shared block counter and return detection only
    # if all 64 pass, and we can do in parallel. 64 is multiple of the warp size, 32
    # memory coalesced into chunks by thread blocks, so blocks should be spread out along row
    # warp size is not multiple of 3, must sync threads after thresholding as all 3 colours must pass?
    # TODO: Memory coalescing is done by default by Maxwell GPU I think...
    # To gain any more speed, we would have to asynchronously load chunks of the image and call kernels on chunks
    # so we can start processing as soon as our chunks are in GPU, whilst loading the next chunks
    # Also if we could load screenshots directly into the GPU rather than CPU-GPU
    # TODO: we do not have full parallel potential, as each thread in the chunk is accessing same counter, and must
    # therefore use blocking, accessing pixel from global memory and doing threshold is parallel
    # TODO: can use shared array #pixels wide, only have 3 atomic operations per pixel, more parallel, and can
    # use a serial or parallel scan on the binary matrix output, basically matrix_scan()...

    kernel1 = SourceModule("""
    #include <stdio.h>
    __global__ void sync_scan(unsigned char *img, int *flag)
    {
        //extern __shared__ int counter;
        //counter = 0;
        extern __shared__ int counter[64];
        int rgb[6] = {-1, 5, 152, 181, 195, 211};
        int x = threadIdx.x + blockDim.x * blockIdx.x;
        int y = blockIdx.y;
        if(x > 5753) { return; }
        counter[int(threadIdx.x /3)] = 1;
        int z = threadIdx.x % 3;
        // check pixel colour against threshold, update counter and then must sync threads
        int pixel = img[x + y*5754];
        int bgr_pass = (rgb[z*2] < pixel) && (pixel < rgb[z*2 + 1]);
        atomicAnd(&counter[int(threadIdx.x /3)], bgr_pass);
        //atomicAdd(&counter, bgr_pass);
        __syncthreads();
        // first thread in block sums the threshold results for the chunk
        if(threadIdx.x == 0) {
            int count = 0;
            #pragma unroll
            for(int i=0; i<64; i++) {
                count += counter[i];
            }
            if(count == 64) {
                flag[0] = int(x/3);
                flag[1] = y;
            }
        }
        //counter = __syncthreads_and(bgr_pass);
        //if(counter == 192) {
        //    flag[0] = int(x/3);
        //    flag[1] = y;
        //}
    }
    """)

    image = cv.imread("test images/crop2.png")
    func = kernel1.get_function("sync_scan")
    # TODO: 1.7 - 2.5 ms to get thread x,y values and access pixels from global memory, rest of the work is very fast?
    timer = time.clock()
    pix = np.array([0, 0], np.uintc)
    # rgb = np.array([-1, 5, 152, 181, 195, 211], np.int32)
    # rgb_gpu = gpuarray.to_gpu_async(rgb)
    pix_gpu = gpuarray.to_gpu_async(pix)
    image_gpu = gpuarray.to_gpu_async(image)
    # block of 48 = 64 x 3, 64 pixels each block

    func(image_gpu, pix_gpu, block=(192, 1, 1), grid=(30, 853))
    print(time.clock() - timer)
    pix = pix_gpu.get()
    print(pix)


chunk_scan()
