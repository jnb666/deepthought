package blas

const ()

const (
	copyKernel = iota
	setKernel
	scaleKernel
	addKernel
	cmpKernel
	sumKernel
	maxColKernel
	normKernel
	transKernel
	mulKernel
	mulElemKernel
	numKernels
)

var name = []string{"copy", "set", "scale", "add", "cmp", "sum", "maxcol", "norm", "transpose", "mul", "mulelem"}

var srcHead = `
// Header structure rows=.x cols=.y base=.z stride=.w

#define ARG const int col=get_global_id(0); const int row=get_global_id(1);

#define P(d, r, c) (d.z+d.w*(r)+(c))
`
var unarySrc = `
__kernel void unary(const int4 ad, const __global float* a, const int4 md, __global float* m) {
	ARG float x = a[P(ad,row,col)];
`

var binarySrc = `
__kernel void binary(const int4 ad, const __global float* a, const int4 bd, const __global float* b, 
		const int4 md, __global float* m) {
	ARG float x = a[P(ad,row,col)]; float y = b[P(bd,row,col)];
`
var source = `
__kernel void copy(const int4 ad, const __global float* a, const int4 md, __global float* m) {
	ARG m[P(md,row,col)] = a[P(ad,row,col)];
}

__kernel void set(const float val, const int4 md, __global float* m) {
	ARG	m[P(md,row,col)] = val;
}

__kernel void scale(const float sc, const int4 md, __global float* m) {
	ARG	m[P(md,row,col)] *= sc;
}

__kernel void add(const float sc, const int4 ad, const __global float* a, const int4 bd, const __global float* b, 
		const int4 md, __global float* m) {
	ARG	m[P(md,row,col)] = a[P(ad,row,col)] + sc*b[P(bd,row,col)]; 
}

__kernel void cmp(const float eps, const int4 ad, const __global float* a, const int4 bd, const __global float* b, 
		const int4 md, __global float* m) {
	ARG	m[P(md,row,col)] = fabs(a[P(ad,row,col)] - b[P(bd,row,col)]) > eps; 
}

__kernel void sum(const int4 md, const __global float* m, __global float* res) {
	float sum = 0.f;
	for (int r = 0; r < md.x; r++) {
		for (int c = 0; c < md.y; c++) {
			sum += m[P(md,r,c)];
		}
	}
	*res = sum;
}

__kernel void maxcol(const int4 ad, const __global float* a, const int4 md, __global float* m) {
	int row = get_global_id(0); int maxcol = 0; 
	float maxval = -1e38f; float v;
	for (int c = 0; c < ad.y; c++) {	
		if ((v = a[P(ad,row,c)]) > maxval) {
			maxval = v; maxcol = c;
		}
	}
	m[P(md,row,0)] = (float)maxcol;
}

__kernel void norm(const int4 ad, const __global float* a, const int4 md, __global float* m) {
	int row = get_global_id(0);
	float sum = 0.f;
	for (int c = 0; c < ad.y; c++) {	
		sum += a[P(ad,row,c)];
	}
	for (int c = 0; c < ad.y; c++) {	
		m[P(ad,row,c)] = a[P(ad,row,c)] / sum;
	}
}

__kernel void mulelem(const int4 ad, const __global float* a, const int4 bd, const __global float* b, 
		const int4 md, __global float* m) {
	ARG	m[P(md,row,col)] = a[P(ad,row,col)] * b[P(bd,row,col)];
}

__kernel void transpose(const int4 ad, const __global float* a, const int4 md, __global float* m) {
  	__local float buffer[TRBLK][TRBLK];
	const int tx = get_local_id(0);
	const int ty = get_local_id(1);
 	const int col = get_group_id(0)*TRBLK + tx;
	const int row = get_group_id(1)*TRBLK + ty;
	if (row < ad.x && col < ad.y) {
		buffer[ty][tx] = a[P(ad,row,col)];
	}
	barrier(CLK_LOCAL_MEM_FENCE);
   	const int col2 = get_group_id(1)*TRBLK + ty;
	const int row2 = get_group_id(0)*TRBLK + tx;
	if (row2 < md.x && col2 < md.y) {
		m[P(md,row2,col2)] = buffer[ty][tx];
	}
}

#define RTS (TS/WPT)
#define LPT (TSK*WPT/TS)

__kernel void mul(const int4 ad, const __global float* a, const int4 bd, const __global float* b,
		const int4 md, __global float* m) {
	const int tx = get_local_id(0);
	const int ty = get_local_id(1);
	const int col = TS*get_group_id(0) + tx;
	const int row = TS*get_group_id(1) + ty;
	const int arow = TS*get_group_id(1) + ty;
	const int brow = TS*get_group_id(0) + ty;

	__local float asub[TS][TSK+2];
	__local float bsub[TS][TSK+2];

	float sum[WPT] = {};
	const int ntiles = ad.w/TSK;

	for (int t = 0; t < ntiles; t++) {
		for (int l = 0; l < LPT; l++) {
			int xcol = TSK*t + tx + l*RTS;
			asub[ty][tx+l*RTS] = a[P(ad, arow, xcol)];
			bsub[ty][tx+l*RTS] = b[P(bd, brow, xcol)];
		}
		barrier(CLK_LOCAL_MEM_FENCE);

		for (int k = 0; k < TSK; k++) {
			for (int w = 0; w < WPT; w++) {	
				sum[w] += asub[ty][k] * bsub[tx+w*RTS][k];		
			}
       	}
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	for (int w = 0; w < WPT; w++) {
		m[P(md, row, col+w*RTS)] = sum[w];
	}
}

#define RS 8
#define Q(d, r, c) (d.z+d.w*(r)/RS+(c))
#define vdot(a, b) (a.s0*b.s0+a.s1*b.s1+a.s2*b.s2+a.s3*b.s3+a.s4*b.s4+a.s5*b.s5+a.s6*b.s6+a.s7*b.s7)
#define Vec float8

__kernel void mul0(const int4 ad, const __global Vec* a, const int4 bd, const __global Vec* b,
		const int4 md, __global Vec* m) {
	ARG
    Vec areg[RS];
    Vec breg[RS];
    float acc[RS][RS] = {};

    const int ntiles = ad.w/RS;
    for (int t = 0; t < ntiles; t++) {
        for (int y = 0; y < RS; y++) {
        	areg[y] = a[Q(ad, row*RS+y, t)];
         	breg[y] = b[Q(bd, col*RS+y, t)];
        }
        for (int y=0; y<RS; y++) {
            for (int x=0; x<RS; x++) {
                acc[x][y] += vdot(areg[x], breg[y]);
            }
        }
    }
    for (int y=0; y<RS; y++) {
        m[Q(md, RS*row+y, col)] = vload8(0, acc[y]);
    }
}

`
