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
	maxColKernel
	normKernel
	transKernel
	mulKernel
	mulATKernel
	mulBTKernel
	mulABTKernel
	mulElemKernel
	numKernels
)

var name = []string{"copy", "copyIx", "set", "scale", "add", "cmp", "sum", "maxcol", "norm",
	"transpose", "mul", "mulAT", "mulBT", "mulABT", "mulelem"}

var srcHead = `
// Matrix header structure
typedef struct {
	int rows, cols, base, stride;
} Dims;

#define ARG const int col=get_global_id(0); const int row=get_global_id(1);

#define P(d, r, c) (d.base+d.stride*(r)+(c))
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
var source = `
__kernel void copy(const Dims ad, const __global float* a, const Dims md, __global float* m) {
	ARG m[P(md,row,col)] = a[P(ad,row,col)];
}

__kernel void copyIx(const Dims ad, const __global float* a, const Dims id, __global float* ix, 
		const Dims md, __global float* m) {
	ARG int irow = ix[P(id,row,0)];
	m[P(md,row,col)] = a[P(ad,irow,col)];
}

__kernel void set(const float val, const Dims md, __global float* m) {
	ARG	m[P(md,row,col)] = val;
}

__kernel void scale(const float sc, const Dims md, __global float* m) {
	ARG	m[P(md,row,col)] *= sc;
}

__kernel void add(const float sc, const Dims ad, const __global float* a, const Dims bd, const __global float* b, 
		const Dims md, __global float* m) {
	ARG	m[P(md,row,col)] = a[P(ad,row,col)] + sc*b[P(bd,row,col)]; 
}

__kernel void cmp(const float eps, const Dims ad, const __global float* a, const Dims bd, const __global float* b, 
		const Dims md, __global float* m) {
	ARG	m[P(md,row,col)] = fabs(a[P(ad,row,col)] - b[P(bd,row,col)]) > eps; 
}

__kernel void sum(const Dims md, const __global float* m, __global float* res) {
	float sum = 0.f;
	for (int r = 0; r < md.rows; r++) {
		for (int c = 0; c < md.cols; c++) {
			sum += m[P(md,r,c)];
		}
	}
	*res = sum;
}

__kernel void maxcol(const Dims ad, const __global float* a, const Dims md, __global float* m) {
	int row = get_global_id(0); int maxcol = 0; 
	float maxval = -1e38f; float v;
	for (int c = 0; c < ad.cols; c++) {	
		if ((v = a[P(ad,row,c)]) > maxval) {
			maxval = v; maxcol = c;
		}
	}
	m[P(md,row,0)] = (float)maxcol;
}

__kernel void norm(const Dims ad, const __global float* a, const Dims md, __global float* m) {
	int row = get_global_id(0);
	float sum = 0.f;
	for (int c = 0; c < ad.cols; c++) {	
		sum += a[P(ad,row,c)];
	}
	for (int c = 0; c < ad.cols; c++) {	
		m[P(ad,row,c)] = a[P(ad,row,c)] / sum;
	}
}

__kernel void mulelem(const Dims ad, const __global float* a, const Dims bd, const __global float* b, 
		const Dims md, __global float* m) {
	ARG	m[P(md,row,col)] = a[P(ad,row,col)] * b[P(bd,row,col)];
}

__kernel void transpose(const Dims ad, const __global float* a, const Dims md, __global float* m) {
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
}

#define RTS (TS/WPT)

#define MHEAD \
	__local float asub[TS][TS+1]; __local float bsub[TS][TS+1];\
	const int tx = get_local_id(0);	const int ty = get_local_id(1);\
	const int gx = TS*get_group_id(0); const int gy = TS*get_group_id(1);\
	float asum[WPT*WPT] = {};

#define MDO(code) for (int wy=0; wy<WPT; wy++) for (int wx=0; wx<WPT; wx++) { code; }

#define MPOS int c=(RTS*RTS*w+ty*RTS+tx)%TS; int r=(RTS*RTS*w+ty*RTS+tx)/TS;

#define OPOS int2 pos = oTrans ? (int2)(gx+tx+wx*RTS, gy+ty+wy*RTS) : (int2)(gy+ty+wy*RTS, gx+tx+wx*RTS);

// matrix multiply
__kernel void mul(const Dims ad, const __global float* a, const Dims bd, const __global float* b,
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
}

// a matrix transposed
__kernel void mulAT(const Dims ad, const __global float* a, const Dims bd, const __global float* b,
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
}

// b matrix transposed
__kernel void mulBT(const Dims ad, const __global float* a, const Dims bd, const __global float* b,
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
}

// a and b matrix transposed
__kernel void mulABT(const Dims ad, const __global float* a, const Dims bd, const __global float* b,
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
}
`

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
