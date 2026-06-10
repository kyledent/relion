/***************************************************************************
 *
 * GPU real-space weighted back-projection for relion_tomo_reconstruct_tomogram.
 *
 * Drop-in replacement for RealSpaceBackprojection::backproject (the Linear,
 * no-taper path) used to build the visualisation/picking tomogram. One thread
 * per output voxel; reproduces real_backprojection.h:135-197 and the bilinear
 * Interpolation::linearXY_clip exactly (double math, edge clamping), so the GPU
 * volume matches the CPU volume to within the GPU<->CPU equivalence tolerance.
 *
 ***************************************************************************/
#ifndef RELION_REAL_BACKPROJECTION_GPU_H
#define RELION_REAL_BACKPROJECTION_GPU_H

// Back-project an (already resampled / CTF-premultiplied) tilt stack into a
// tomogram on the GPU.
//
//   stack     : fc * H * W floats, frame-major (frame f slice = stack + f*H*W),
//               x fastest then y (RawImage layout)
//   projRows  : fc * 8 doubles; for each frame the first two rows of the 4x4
//               projection matrix: [m00 m01 m02 m03  m10 m11 m12 m13]
//               (i.e. projAct[f][0..7], which already folds in `spacing`)
//   W, H      : tilt-image dimensions
//   outX/Y/Z  : output tomogram dimensions
//   ox/oy/oz  : reconstruction origin (orig = (x0,y0,z0))
//   spacing   : output-voxel spacing (binning)
//   out       : outX*outY*outZ floats (host), filled on return (x fastest)
//   tileZ     : output z-slab depth (0 or >=outZ = whole volume in one pass). Tiling
//               keeps the tilt stack resident and only the output slab on the GPU, so
//               volumes larger than VRAM still reconstruct.
//   fast      : false = exact double-precision global-memory gather (bit-equivalent to
//               the CPU); true = layered-texture hardware-bilinear single-precision
//               kernel (faster, equivalent only to ~routeB FSC tolerance).
//   device    : CUDA device id (-1 = leave current)
//
// Voxels onto which no frame projects are left at 0 (matches the CPU wgh==0 case
// with a zero-initialised destination).
void wbpBackprojectGPU(
        const float*  stack,
        const double* projRows,
        int fc, int W, int H,
        int outX, int outY, int outZ,
        double ox, double oy, double oz, double spacing,
        float* out,
        int tileZ,
        bool fast,
        int device);

#endif
