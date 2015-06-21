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
	approxKernel
	filterKernel
	randomKernel
	numKernels
)

var name = []string{"copy", "copyIx", "set", "scale", "add", "cmp", "sum", "sumrows", "maxcol", "norm",
	"histogram", "mulelem", "transpose", "mul", "mulAT", "mulBT", "mulABT", "loadImage", "approx", "filter", "random"}

var srcHead = `
// Matrix header structure
typedef struct {
	int rows, cols, base, stride;
} Dims;

#define ARG const int col=get_global_id(0); const int row=get_global_id(1);

#define P(d, r, c) (d.base+d.stride*(r)+(c))
`

var mulHead = `
#define RTS (TS/WPT)

#define MHEAD \
	__local float asub[TS][TS+1]; __local float bsub[TS][TS+1];\
	const int tx = get_local_id(0);	const int ty = get_local_id(1);\
	const int gx = TS*get_group_id(0); const int gy = TS*get_group_id(1);\
	float asum[WPT*WPT] = {};

#define MDO(code) for (int wy=0; wy<WPT; wy++) for (int wx=0; wx<WPT; wx++) { code; }

#define MPOS int c=(RTS*RTS*w+ty*RTS+tx)%TS; int r=(RTS*RTS*w+ty*RTS+tx)/TS;

#define OPOS int2 pos = oTrans ? (int2)(gx+tx+wx*RTS, gy+ty+wy*RTS) : (int2)(gy+ty+wy*RTS, gx+tx+wx*RTS);
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
	const int tpos = get_local_id(1)*TRBLK + get_local_id(0);
	__local float buffer[TRBLK*TRBLK];
	buffer[tpos] = m[P(md, row, col)];
	barrier(CLK_LOCAL_MEM_FENCE);
	if (tpos != 0) return;
	float sum = 0.f;
	for (int i = 0; i < TRBLK*TRBLK; i++) sum += buffer[i];
	int ix = get_group_id(1)*get_num_groups(0) + get_group_id(0);
	res[ix] = sum;
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
	int maxt = ad.stride;
	for (int t = 0; t < maxt; t += TS) {
		for (int w = 0; w < WPT*WPT; w++) {
			MPOS
			asub[r][c] = a[P(ad, gy+r, t+c)];
			bsub[r][c] = b[P(bd, t+c, gx+r)];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
		for (int k = 0; k < TS; k++) {
			MDO(asum[wy*WPT+wx] += asub[ty+wy*RTS][k] * bsub[tx+wx*RTS][k])
		}
	}
	MDO(OPOS; m[P(md, pos.x, pos.y)] = asum[wy*WPT+wx];)
}`,
	`__kernel void mulAT(const Dims ad, const __global float* a, const Dims bd, const __global float* b,
		const Dims md, __global float* m, int oTrans) {
	MHEAD
	int maxt = (1 + (ad.rows-1)/TS) * TS;
	for (int t = 0; t < maxt; t += TS) {
		for (int w = 0; w < WPT*WPT; w++) {
			MPOS
			asub[r][c] = a[P(ad, t+c, gy+r)];
			bsub[r][c] = b[P(bd, t+c, gx+r)];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
		for (int k = 0; k < TS; k++) {
			MDO(asum[wy*WPT+wx] += asub[ty+wy*RTS][k] * bsub[tx+wx*RTS][k])
		}
	}
	MDO(OPOS; m[P(md, pos.x, pos.y)] = asum[wy*WPT+wx];)
}`,
	`__kernel void mulBT(const Dims ad, const __global float* a, const Dims bd, const __global float* b,
		const Dims md, __global float* m, int oTrans) {
	MHEAD
	int maxt = ad.stride;
	for (int t = 0; t < maxt; t += TS) {
		for (int w = 0; w < WPT*WPT; w++) {
			MPOS
			asub[r][c] = a[P(ad, gy+r, t+c)];
			bsub[r][c] = b[P(bd, gx+r, t+c)];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
		for (int k = 0; k < TS; k++) {
			MDO(asum[wy*WPT+wx] += asub[ty+wy*RTS][k] * bsub[tx+wx*RTS][k])
		}
	}
	MDO(OPOS; m[P(md, pos.x, pos.y)] = asum[wy*WPT+wx];)
}`,
	`__kernel void mulABT(const Dims ad, const __global float* a, const Dims bd, const __global float* b,
		const Dims md, __global float* m, int oTrans) {
	MHEAD
	int maxt = bd.stride;
	for (int t = 0; t < maxt; t += TS) {
		for (int w = 0; w < WPT*WPT; w++) {
			MPOS
			asub[r][c] = a[P(ad, t+c, gy+r)];
			bsub[r][c] = b[P(bd, gx+r, t+c)];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
		for (int k = 0; k < TS; k++) {
			MDO(asum[wy*WPT+wx] += asub[ty+wy*RTS][k] * bsub[tx+wx*RTS][k])
		}
	}
	MDO(OPOS; m[P(md, pos.x, pos.y)] = asum[wy*WPT+wx];)
}`,
	`__kernel void loadImage(__write_only image2d_array_t img, const Dims md, const __global float* m) {
	ARG
   	const int width = get_image_width(img);
	const int nimg = get_global_id(2);
	int4 ipos = (int4)(col, row, nimg, 0);
	float4 val = (float4)(m[P(md, nimg, row*width+col)], 0, 0, 0);
	write_imagef(img, ipos, val);
}`,
	`__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;

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
	`__kernel void filter(const Dims x1d, const __global float* x1, const Dims y1d, const __global float* y1,
			const Dims x2d, __global float* x2, const Dims y2d, __global float* y2,
			const Dims kd, const __global float* kern) {
	ARG
	const int width = get_global_size(0);
	const int height = x1d.cols / width;
 	const int xy = row*width+col;
	const int nimg = get_global_id(2);
	float xconv = 0.f;
	float yconv = 0.f;
	for (int y = 0; y < kd.rows; y++) {
		int yy = row - height/2 + y;
		for (int x = 0; x < kd.cols; x++) {
			int xx = col - width/2 + x;
			if (xx >= 0 && xx < width && yy >= 0 && yy < height) {
				int pos = P(x1d, nimg, yy*width+xx);
				xconv += x1[pos] * kern[P(kd,y,x)];
				yconv += y1[pos] * kern[P(kd,y,x)];
			}
		}
	}
	x2[P(x2d,nimg,xy)] = xconv;
	y2[P(y2d,nimg,xy)] = yconv;
}`,
	`
// simple fast random number generator
// based on http://cas.ee.ic.ac.uk/people/dt10/research/rngs-gpu-mwc64x.html
#define MWC_A 4294883355U
#define MWC_M 18446383549859758079UL
#define MWC_BASEID 4077358422479273989UL

ulong addMod64(ulong a, ulong b) {
	ulong v = a + b;
	if ((v >= MWC_M) || (v < a)) v = v-MWC_M;
	return v;
}

ulong mulMod64(ulong a, ulong b) {
	ulong r = 0;
	while (a != 0) {
		if (a & 1) r = addMod64(r, b);
		b = addMod64(b, b);
		a = a>>1;
	}
	return r;
}

ulong powMod64(ulong a, ulong e) {
	ulong sqr=a, acc=1;
	while (e != 0) {
		if (e & 1) acc = mulMod64(acc, sqr);
		sqr = mulMod64(sqr, sqr);
		e = e>>1;
	}
	return acc;
}

float rnd(uint2* state) {
	uint x=(*state).x, c=(*state).y;
	uint res = x^c;
	uint hi = mul_hi(x, MWC_A);
	x = x*MWC_A + c;
	c = hi + (x<c);
	*state = (uint2)(x,c);
	return res / 4294967296.0;
}

uint2 seedStreams(ulong stream) {
	ulong m = powMod64(MWC_A, stream);
	ulong x = mulMod64(MWC_BASEID, m);
	return (uint2)((uint)(x / MWC_A), (uint)(x % MWC_A));
}

__kernel void random(const Dims md, __global float* m, const float rmin, const float range,
		const int nseed, __global uint2* seeds) {
	const int id = get_local_id(0);
	const int pos = get_global_id(0);
	uint2 state = (nseed == 0) ? seedStreams(id*0x100000000) : seeds[id];
	m[P(md, pos/md.cols, pos%md.cols)] = rmin + range * rnd(&state);
	seeds[id] = state;
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
