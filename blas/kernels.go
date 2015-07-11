package blas

const ()

const (
	copyKernel = iota
	copyIxKernel
	setKernel
	scaleKernel
	addKernel
	cmpKernel
	sumKernel
	sumRowsKernel
	maxColKernel
	normKernel
	histKernel
	mulElemKernel
	transKernel
	mulKernel
	mulATKernel
	mulBTKernel
	mulABTKernel
	loadImageKernel
	loadImageKernel2
	approxKernel
	scaleImageKernel
	rotateKernel
	randomKernel
	numKernels
)

var name = []string{"copy", "copyIx", "set", "scale", "add", "cmp", "sum", "sumrows", "maxcol", "norm",
	"histogram", "mulelem", "transpose", "mul", "mulAT", "mulBT", "mulABT",
	"loadImage", "loadImage2", "approx", "scaleImage", "rotateImage", "random"}

var srcHead = `
// Matrix header structure
typedef struct {
	int rows, cols, base, stride;
} Dims;

#define ARG const int col=get_global_id(0); const int row=get_global_id(1);

#define P(d, r, c) (d.base+d.stride*(r)+(c))
`

var mulHead = `
#define MHEAD \
	__local float asub[TS][TS+2]; __local float bsub[TS][TS+2];\
	const int tx = get_local_id(0);	const int ty = get_local_id(1);\
	const int gx = TS*get_group_id(0); const int gy = TS*get_group_id(1);\
	float sum = 0.f;

#define MULBLK(s, a, b, x, y) do { \
	s += a[y][0] * b[x][0];		\
	s += a[y][1] * b[x][1];		\
	s += a[y][2] * b[x][2];		\
	s += a[y][3] * b[x][3];		\
	s += a[y][4] * b[x][4];		\
	s += a[y][5] * b[x][5];		\
	s += a[y][6] * b[x][6];		\
	s += a[y][7] * b[x][7];		\
	s += a[y][8] * b[x][8];		\
	s += a[y][9] * b[x][9];		\
	s += a[y][10] * b[x][10];	\
	s += a[y][11] * b[x][11];	\
	s += a[y][12] * b[x][12];	\
	s += a[y][13] * b[x][13];	\
	s += a[y][14] * b[x][14];	\
	s += a[y][15] * b[x][15];	\
} while (0)

#define OPOS int2 pos = oTrans ? (int2)(gx+tx, gy+ty) : (int2)(gy+ty, gx+tx);
`

var unarySrc = `
__kernel void unary(const Dims ad, const __global float* a, const Dims md, __global float* m) {
	ARG float x = a[P(ad,row,col)];
`

var binarySrc = `
__kernel void binary(const Dims ad, const __global float* a, const Dims bd, const __global float* b, 
		const Dims md, __global float* m) {
	ARG float x = a[P(ad,row,col)]; float y = b[P(bd,row,col)];
`

var filterHead = `
	__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

	__kernel void filter(__read_only image2d_array_t img, const Dims kd, const __global float* kern,
			const Dims xd, __global float* outx, const Dims yd, __global float* outy) {
	ARG
	const float xx = col-FILTER_CENTER+.5f;
	const float zz = get_global_id(2);
	float4 conv = (float4)(0.f);
	for (int y = 0; y < FILTER_SIZE; y++) {
		float yy = row+y-FILTER_CENTER+.5f;
`
var filterLoop = "		conv += read_imagef(img,sampler,(float4)(xx+%d.f,yy,zz,0.f)) * kern[FILTER_STRIDE*y+%d];\n"

var filterTail = `	}
	outx[P(xd, get_global_id(2), row*get_global_size(0)+col)] = conv.x;
	outy[P(yd, get_global_id(2), row*get_global_size(0)+col)] = conv.y;	
}`

var source = []string{
	`__kernel void copy(const Dims ad, const __global float* a, const Dims md, __global float* m) {
	ARG m[P(md,row,col)] = a[P(ad,row,col)];
}`,
	`__kernel void copyIx(const Dims ad, const __global float* a, const Dims id, __global float* ix, 
		const Dims md, __global float* m) {
	ARG int irow = ix[P(id,row,0)];
	m[P(md,row,col)] = a[P(ad,irow,col)];
}`,
	`__kernel void set(const float val, const Dims md, __global float* m) {
	ARG	m[P(md,row,col)] = val;
}`,
	`__kernel void scale(const float sc, const Dims md, __global float* m) {
	ARG	m[P(md,row,col)] *= sc;
}`,
	`__kernel void add(const float sc, const Dims ad, const __global float* a, const Dims bd, const __global float* b, 
		const Dims md, __global float* m) {
	ARG	m[P(md,row,col)] = a[P(ad,row,col)] + sc*b[P(bd,row,col)]; 
}`,
	`__kernel void cmp(const float eps, const Dims ad, const __global float* a, const Dims bd, const __global float* b, 
		const Dims md, __global float* m) {
	ARG	m[P(md,row,col)] = fabs(a[P(ad,row,col)] - b[P(bd,row,col)]) > eps; 
}`,
	`__kernel void sum(const Dims md, const __global float* m, __global float* res) {
	ARG
	const int ty = get_local_id(1); 
	const int tx = get_local_id(0);
	__local float buffer[TRBLK][TRBLK];
	buffer[ty][tx] = (row < md.rows && col < md.cols) ? m[P(md, row, col)] : 0.f;
	barrier(CLK_LOCAL_MEM_FENCE);
	if (tx == 0 && ty == 0) {
		float sum = 0.f;
		for (int y = 0; y < get_local_size(1); y++) {
			for (int x = 0; x < get_local_size(0); x++) {
				sum += buffer[y][x];
			}
		}
		int ix = get_group_id(1)*get_num_groups(0) + get_group_id(0);
		res[ix] = sum;
	}
}`,
	`__kernel void sumrows(const Dims ad, const __global float* a, const Dims md, __global float* m) {
	const int row = get_global_id(0);
	float sum = 0.f;
	for (int c = 0; c < ad.cols; c++) {	
		sum += a[P(ad,row,c)];
	}
	m[P(md,row,0)] = sum;
}`,
	`__kernel void maxcol(const Dims ad, const __global float* a, const Dims md, __global float* m) {
	const int row = get_global_id(0);
	int maxcol = 0; 
	float maxval = -1e38f; float v;
	for (int c = 0; c < ad.cols; c++) {	
		if ((v = a[P(ad,row,c)]) > maxval) {
			maxval = v; maxcol = c;
		}
	}
	m[P(md,row,0)] = (float)maxcol;
}`,
	`__kernel void norm(const Dims ad, const __global float* a, const Dims md, __global float* m) {
	const int row = get_global_id(0);
	float sum = 0.f;
	for (int c = 0; c < ad.cols; c++) {	
		sum += a[P(ad,row,c)];
	}
	for (int c = 0; c < ad.cols; c++) {	
		m[P(ad,row,c)] = a[P(ad,row,c)] / sum;
	}
}`,
	`__kernel void histogram(int bins, float xmin, float scale, const Dims ad, const __global float* a, 
		const Dims md, __global float* m, __local int* buffer) {
	const int row = get_global_id(0);
	const int gid = get_group_id(0);
	if (row == 0 && gid == 0) {
		for (int i = 0; i < bins; i++) buffer[i] = m[P(md,i,0)];
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	int bin = floor(scale * (a[P(ad,row,0)] - xmin));
	bin = clamp(bin, 0, bins-1);
	atomic_inc(buffer+bin);
	barrier(CLK_LOCAL_MEM_FENCE);
	if (row == 0 && gid == get_num_groups(0)-1) {
		for (int i = 0; i < bins; i++) m[P(md,i,0)] = buffer[i];
	}
}`,
	`__kernel void mulelem(const Dims ad, const __global float* a, const Dims bd, const __global float* b, 
		const Dims md, __global float* m) {
	ARG	m[P(md,row,col)] = a[P(ad,row,col)] * b[P(bd,row,col)];
}`,
	`__kernel void transpose(const Dims ad, const __global float* a, const Dims md, __global float* m) {
  	__local float buffer[TRBLK][TRBLK];
	const int tx = get_local_id(0);
	const int ty = get_local_id(1);
 	const int col = get_group_id(0)*TRBLK + tx;
	const int row = get_group_id(1)*TRBLK + ty;
	if (row < ad.rows && col < ad.cols) {
		buffer[ty][tx] = a[P(ad,row,col)];
	}
	barrier(CLK_LOCAL_MEM_FENCE);
   	const int col2 = get_group_id(1)*TRBLK + ty;
	const int row2 = get_group_id(0)*TRBLK + tx;
	if (row2 < md.rows && col2 < md.cols) {
		m[P(md,row2,col2)] = buffer[ty][tx];
	}
}`,
	`__kernel void mul(const Dims ad, const __global float* a, const Dims bd, const __global float* b,
		const Dims md, __global float* m, int oTrans) {
	MHEAD
	const int maxt = ad.stride;
	const int abase = ad.base + ad.stride*(gy+ty) + tx;
	const int bbase = bd.base + bd.stride*tx + gx + ty;
	for (int t = 0; t < maxt; t += TS) {
		asub[ty][tx] = a[abase + t];
		bsub[ty][tx] = b[bbase + bd.stride*t];
		barrier(CLK_LOCAL_MEM_FENCE);
		MULBLK(sum, asub, bsub, tx, ty);
	}
	OPOS; m[P(md, pos.x, pos.y)] = sum;
}`,
	`__kernel void mulAT(const Dims ad, const __global float* a, const Dims bd, const __global float* b,
		const Dims md, __global float* m, int oTrans) {
	MHEAD
	const int maxt = (1 + (ad.rows-1)/TS) * TS;
	const int abase = ad.base + ad.stride*tx + gy + ty;
	const int bbase = bd.base + bd.stride*tx + gx + ty;
	for (int t = 0; t < maxt; t += TS) {
		asub[ty][tx] = a[abase + ad.stride*t];
		bsub[ty][tx] = b[bbase + bd.stride*t];
		barrier(CLK_LOCAL_MEM_FENCE);
		MULBLK(sum, asub, bsub, tx, ty);
	}
	OPOS; m[P(md, pos.x, pos.y)] = sum;
}`,
	`__kernel void mulBT(const Dims ad, const __global float* a, const Dims bd, const __global float* b,
		const Dims md, __global float* m, int oTrans) {
	MHEAD
	const int maxt = ad.stride;
	const int abase = ad.base + ad.stride*(gy+ty) + tx;
	const int bbase = bd.base + bd.stride*(gx+ty) + tx;
	for (int t = 0; t < maxt; t += TS) {
		asub[ty][tx] = a[abase + t];
		bsub[ty][tx] = b[bbase + t];
		barrier(CLK_LOCAL_MEM_FENCE);
		MULBLK(sum, asub, bsub, tx, ty);
	}
	OPOS; m[P(md, pos.x, pos.y)] = sum;
}`,
	`__kernel void mulABT(const Dims ad, const __global float* a, const Dims bd, const __global float* b,
		const Dims md, __global float* m, int oTrans) {
	MHEAD
	const int maxt = bd.stride;
	const int abase = ad.base + ad.stride*tx + gy + ty;
	const int bbase = bd.base + bd.stride*(gx+ty) + tx;
	for (int t = 0; t < maxt; t += TS) {
		asub[ty][tx] = a[abase + ad.stride*t];
		bsub[ty][tx] = b[bbase + t];
		barrier(CLK_LOCAL_MEM_FENCE);
		MULBLK(sum, asub, bsub, tx, ty);
	}
	OPOS; m[P(md, pos.x, pos.y)] = sum;
}`,
	`__kernel void loadImage(__write_only image2d_array_t img, const Dims md, const __global float* m) {
	ARG
   	const int width = get_image_width(img);
	const int nimg = get_global_id(2);
	float4 val = (float4)(m[P(md,nimg,row*width+col)], 0, 0, 0);
	write_imagef(img, (int4)(col,row,nimg,0), val);
}`,
	`__kernel void loadImage2(__write_only image2d_array_t img, const Dims ad, const __global float* a,
			const Dims bd, const __global float* b) {
	ARG
   	const int width = get_image_width(img);
	const int nimg = get_global_id(2);
	const int pos = row*width+col;
	float4 val = (float4)(a[P(ad,nimg,pos)], b[P(bd,nimg,pos)], 0, 0);
	write_imagef(img, (int4)(col,row,nimg,0), val);
}`,
	`__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_LINEAR;

	__kernel void approx(__read_only image2d_array_t img, const Dims xd, const __global float* x,
			const Dims yd, const __global float* y, const Dims md, __global float* m) {
 	ARG
 	const int xy = row*get_image_width(img)+col;
	const int nimg = get_global_id(2);
	float4 pos;
	pos.x = col + x[P(xd,nimg,xy)] + .5f;
	pos.y = row + y[P(yd,nimg,xy)] + .5f;
	pos.z = nimg;
	float4 val = read_imagef(img, sampler, pos);
	m[P(md,nimg,xy)] = val.x;
}`,
	`__kernel void scaleImage(const float x0, const float y0,
			const Dims sxd, const __global float* sx, const Dims syd, const __global float* sy,
			const Dims xd, __global float* x, const Dims yd, __global float* y) {
	ARG
 	const int xy = row*get_global_size(0)+col;
	const int nimg = get_global_id(2);
	x[P(xd,nimg,xy)] += (col-x0) * sx[P(sxd,0,nimg)];
	y[P(yd,nimg,xy)] += (row-y0) * sy[P(syd,0,nimg)];	
}`,
	`__kernel void rotateImage(const float x0, const float y0, const Dims ad, const __global float* ang, 
			const Dims xd, __global float* x, const Dims yd, __global float* y) {
	ARG
 	const int xy = row*get_global_size(0)+col;
	const int nimg = get_global_id(2);
	float cosa;
	float sina = sincos(ang[P(ad,0,nimg)], &cosa);
	x[P(xd,nimg,xy)] += (cosa-1.f)*(col-x0) - sina*(row-y0);
	y[P(yd,nimg,xy)] += (cosa-1.f)*(row-y0) + sina*(col-x0);
}`,
	`
// simple fast random number generator
// based on http://cas.ee.ic.ac.uk/people/dt10/research/rngs-gpu-mwc64x.html
float rnd(uint2* state) {
	uint x=(*state).x, c=(*state).y;
	uint res = x^c;
	uint hi = mul_hi(x, MWC_A);
	x = x*MWC_A + c;
	c = hi + (x<c);
	*state = (uint2)(x,c);
	return res / 4294967296.0;
}

__kernel void random(const Dims md, __global float* m, const float rmin, const float range, 
		const __global uint2* seedsIn, __global uint2* seedsOut) {
	const int id = get_local_id(0);
	const int pos = get_global_id(0);
	const ulong stream = id*0x100000000;
	uint2 state = seedsIn[id];
	m[P(md, pos/md.cols, pos%md.cols)] = rmin + range * rnd(&state);
	seedsOut[id] = state;
}`,
}

/* generalised multiply
#define RTSX (TSX/WPTX)
#define RTSY (TSY/WPTY)
#define LPTA ((TSK*TSY)/(RTSX*RTSY))
#define LPTB ((TSK*TSX)/(RTSX*RTSY))

__kernel void mul(const Dims ad, const __global float* a, const Dims bd, const __global float* b,
		const Dims md, __global float* m) {
	const int tx = get_local_id(0);
	const int ty = get_local_id(1);
	const int offX = TSX*get_group_id(0);
	const int offY = TSY*get_group_id(1);

	__local float asub[TSY][TSK+2];
	__local float bsub[TSX][TSK+2];

	float sum[WPTY][WPTX] = {};
	const int base = ty*RTSX + tx;

	for (int t = 0; t < ad.w; t += TSK) {
		for (int la = 0; la < LPTA*RTSX*RTSY; la += RTSX*RTSY) {
			int col = (la+base) % TSK;
			int row = (la+base) / TSK;
			asub[row][col] = a[P(ad, offY+row, t+col)];
		}
		for (int lb = 0; lb < LPTB*RTSX*RTSY; lb += RTSX*RTSY) {
			int col = (lb+base) % TSK;
			int row = (lb+base) / TSK;
			bsub[row][col] = b[P(bd, offX+row, t+col)];
		}
		barrier(CLK_LOCAL_MEM_FENCE);

		for (int k = 0; k < TSK; k++) {
			for (int wy = 0; wy < WPTY; wy++) {
				for (int wx = 0; wx < WPTX; wx++) {
					sum[wy][wx] += asub[ty+wy*RTSY][k] * bsub[tx+wx*RTSX][k];
				}
			}
       	}
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	for (int wy = 0; wy < WPTY; wy++) {
		for (int wx = 0; wx < WPTX; wx++) {
			m[P(md, offY+ty+wy*RTSY, offX+tx+wx*RTSX)] = sum[wy][wx];
		}
	}
}
*/
