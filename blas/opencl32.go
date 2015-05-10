package blas

import (
	"fmt"
	"github.com/go-gl/cl/v1.2/cl"
	"github.com/jnb666/deepthought/scl"
	"unsafe"
)

const (
	wSize = 4
	dSize = 16
)

const (
	copyKernel = iota
	setKernel
	scaleKernel
	addKernel
	cmpKernel
	sumKernel
	maxColKernel
	normKernel
	mulKernel1
	mulKernel2
	mulKernel3
	mulKernel4
	mulElemKernel1
	mulElemKernel2
	mulElemKernel3
	mulElemKernel4
	numKernels
)

var (
	hw   *scl.Hardware
	sw   []*scl.Software
	name = []string{"copy", "set", "scale", "add", "cmp", "sum", "maxcol", "norm",
		"mul1", "mul2", "mul3", "mul4", "mulelem1", "mulelem2", "mulelem3", "mulelem4"}
)

var srcHead = `
typedef struct {
	int rows;
	int cols;
	int base;
	int stride;
} Dims;

#define ARG int row = get_global_id(0); int col = get_global_id(1);

#define P(d, r, c) (d.base + d.stride*(r) + (c))
`

var source = `
__kernel void copy(Dims ad, __global float* a, Dims md, __global float* m) {
	ARG	m[P(md,row,col)] = a[P(ad,row,col)];
}

__kernel void set(float val, Dims md, __global float* m) {
	ARG	m[P(md,row,col)] = val;
}

__kernel void scale(float sc, Dims md, __global float* m) {
	ARG	m[P(md,row,col)] *= sc;
}

__kernel void add(float sc, Dims ad, __global float* a, Dims bd, __global float* b, Dims md, __global float* m) {
	ARG	m[P(md,row,col)] = a[P(ad,row,col)] + sc*b[P(bd,row,col)]; 
}

__kernel void cmp(float eps, Dims ad, __global float* a, Dims bd, __global float* b, Dims md, __global float* m) {
	ARG	m[P(md,row,col)] = fabs(a[P(ad,row,col)] - b[P(bd,row,col)]) > eps; 
}

__kernel void sum(Dims md, __global float* m, __global float* res) {
	float sum = 0.f;
	for (int r = 0; r < md.rows; r++) {
		for (int c = 0; c < md.cols; c++) {
			sum += m[P(md,r,c)];
		}
	}
	*res = sum;
}

__kernel void maxcol(Dims ad, __global float* a, Dims md, __global float* m) {
	int row = get_global_id(0); int maxcol = 0; 
	float maxval = -1e38f; float v;
	for (int c = 0; c < ad.cols; c++) {	
		if ((v = a[P(ad,row,c)]) > maxval) {
			maxval = v; maxcol = c;
		}
	}
	m[P(md,row,0)] = (float)maxcol;
}

__kernel void norm(Dims ad, __global float* a, Dims md, __global float* m) {
	int row = get_global_id(0);
	float sum = 0.f;
	for (int c = 0; c < ad.cols; c++) {	
		sum += a[P(ad,row,c)];
	}
	for (int c = 0; c < ad.cols; c++) {	
		m[P(ad,row,c)] = a[P(ad,row,c)] / sum;
	}
}

__kernel void mul1(Dims ad, __global float* a, Dims bd, __global float* b, Dims md, __global float* m) {
	ARG	float sum = 0.f;
	for (int k = 0; k < ad.cols; k++) {
		sum += a[P(ad,row,k)] * b[P(bd,k,col)];
	}
	m[P(md,row,col)] = sum;
}

__kernel void mul2(Dims ad, __global float* a, Dims bd, __global float* b, Dims md, __global float* m) {
	ARG	float sum = 0.f;
	for (int k = 0; k < ad.rows; k++) {
		sum += a[P(ad,k,row)] * b[P(bd,k,col)];
	}
	m[P(md,row,col)] = sum;
}

__kernel void mul3(Dims ad, __global float* a, Dims bd, __global float* b, Dims md, __global float* m) {
	ARG	float sum = 0.f;
	for (int k = 0; k < ad.cols; k++) {
		sum += a[P(ad,row,k)] * b[P(bd,col,k)];
	}
	m[P(md,row,col)] = sum;
}

__kernel void mul4(Dims ad, __global float* a, Dims bd, __global  float* b, Dims md, __global float* m) {
	ARG	float sum = 0.f;
	for (int k = 0; k < ad.rows; k++) {
		sum += a[P(ad,k,row)] * b[P(bd,col,k)];
	}
	m[P(md,row,col)] = sum;
}

__kernel void mulelem1(Dims ad, __global float* a, Dims bd, __global float* b, Dims md, __global float* m) {
	ARG	m[P(md,row,col)] = a[P(ad,row,col)] * b[P(bd,row,col)];
}

__kernel void mulelem2(Dims ad, __global float* a, Dims bd, __global float* b, Dims md, __global float* m) {
	ARG	m[P(md,row,col)] = a[P(ad,col,row)] * b[P(bd,row,col)];
}

__kernel void mulelem3(Dims ad, __global float* a, Dims bd, __global float* b, Dims md, __global float* m) {
	ARG	m[P(md,row,col)] = a[P(ad,row,col)] * b[P(bd,col,row)];
}

__kernel void mulelem4(Dims ad, __global float* a, Dims bd, __global float* b, Dims md, __global float* m) {
	ARG	m[P(md,row,col)] = a[P(ad,col,row)] * b[P(bd,col,row)];
}
`

// matrix dimensions
type dims struct {
	rows   int32
	cols   int32
	base   int32
	stride int32
}

// matrix of float32 stored internally in row major order
type opencl32 struct {
	dims
	buf    *scl.Buffer
	data   []float32
	format string
}

// initialise opencl
func initCL() {
	var err error
	hw = scl.Devices().Select(0)
	fmt.Println("Init OpenCL:", hw)
	sw = make([]*scl.Software, numKernels)
	for i := range sw {
		if sw[i], err = scl.Compile(hw, srcHead+source, name[i], ""); err != nil {
			panic(err)
		}
	}
}

func releaseCL() {
	for _, s := range sw {
		s.Release()
	}
	hw.Release()
}

// constructor
func newopencl32(rows, cols int) Matrix {
	hostData := make([]float32, rows*cols)
	buffer := scl.NewBuffer(hw, cl.MEM_READ_WRITE, rows*cols*wSize, hostData)
	buffer.Write(hw)
	return &opencl32{
		dims:   dims{rows: int32(rows), cols: int32(cols), stride: int32(cols)},
		buf:    buffer,
		data:   hostData,
		format: "%8.4f",
	}
}

func (m *opencl32) Release() { m.buf.Release() }

// accessors
func (m *opencl32) Rows() int { return int(m.rows) }

func (m *opencl32) Cols() int { return int(m.cols) }

func (m *opencl32) Size() int { return len(m.data) - int(m.base) }

func (m *opencl32) SetFormat(f string) { m.format = f }

func (m *opencl32) String() string {
	return format(m.format, int(m.rows), int(m.cols), m.Data(RowMajor))
}

// Load method initialises a matrix with data from a list of float64 values.
// If the number of values is less than the size then they are repeated to fill the matrix.
func (m *opencl32) Load(order Ordering, vals ...float64) Matrix {
	var row, col int32
	if len(vals) == 0 {
		panic("blas:Load - no data provided")
	}
	if err := m.buf.Read(hw); err != cl.SUCCESS {
		panic(cl.ErrToStr(err))
	}
	next := getNext(vals)
	if order == RowMajor {
		for row = 0; row < m.rows; row++ {
			for col = 0; col < m.cols; col++ {
				m.data[m.base+row*m.stride+col] = float32(next())
			}
		}
	} else {
		for col = 0; col < m.cols; col++ {
			for row = 0; row < m.rows; row++ {
				m.data[m.base+row*m.stride+col] = float32(next())
			}
		}
	}
	m.buf.Write(hw)
	return m
}

// Data method returns a copy of the matrix data as a slice
func (m *opencl32) Data(order Ordering) []float64 {
	var row, col int32
	if err := m.buf.Read(hw); err != cl.SUCCESS {
		panic(cl.ErrToStr(err))
	}
	data := make([]float64, m.rows*m.cols)
	i := 0
	if order == RowMajor {
		for row = 0; row < m.rows; row++ {
			for col = 0; col < m.cols; col++ {
				data[i] = float64(m.data[m.base+row*m.stride+col])
				i++
			}
		}
	} else {
		for col = 0; col < m.cols; col++ {
			for row = 0; row < m.rows; row++ {
				data[i] = float64(m.data[m.base+row*m.stride+col])
				i++
			}
		}
	}
	return data
}

// Reshape method changes the dimensions without altering the data
func (m *opencl32) Reshape(rows, cols int, shrink bool) Matrix {
	return m.reshape(int32(rows), int32(cols), shrink)
}

func (m *opencl32) reshape(rows, cols int32, shrink bool) Matrix {
	if m.Size() < int(rows*cols) {
		panic("blas:Reshape - matrix is too small")
	}
	m.rows, m.cols = rows, cols
	if shrink || m.stride < m.cols {
		m.stride = m.cols
	}
	return m
}

// Row method returns a view on the matrix with rows [r1:r2] exclusive.
func (m *opencl32) Row(row1, row2 int) Matrix {
	r1, r2 := int32(row1), int32(row2)
	return &opencl32{
		dims:   dims{rows: r2 - r1, cols: m.cols, base: m.base + r1*m.stride, stride: m.stride},
		buf:    m.buf,
		data:   m.data,
		format: m.format,
	}
}

// Col method returns a view on the matrix with columns [c1:c2] exclusive.
func (m *opencl32) Col(col1, col2 int) Matrix {
	c1, c2 := int32(col1), int32(col2)
	return &opencl32{
		dims:   dims{rows: m.rows, cols: c2 - c1, base: c1, stride: m.stride},
		buf:    m.buf,
		data:   m.data,
		format: m.format,
	}
}

// Copy method returs a copy of the input matrix
func (m *opencl32) Copy(in Matrix) Matrix {
	a := in.(*opencl32)
	m.reshape(a.rows, a.cols, false)
	k := sw[copyKernel]
	setArgMatrix(k, 0, a)
	setArgMatrix(k, 2, m)
	err := k.EnqueueKernel(hw, m.Rows(), m.Cols(), false)
	if err != nil {
		panic(err)
	}
	return m
}

// Set method sets all elements of the matrix to the given value
func (m *opencl32) Set(val float64) Matrix {
	value := float32(val)
	k := sw[setKernel]
	k.SetArg(0, wSize, unsafe.Pointer(&value))
	setArgMatrix(k, 1, m)
	err := k.EnqueueKernel(hw, m.Rows(), m.Cols(), false)
	if err != nil {
		panic(err)
	}
	return m
}

// Scale method muliplies each element of the matrix by a scalar.
func (m *opencl32) Scale(s float64) Matrix {
	sc := float32(s)
	k := sw[scaleKernel]
	k.SetArg(0, wSize, unsafe.Pointer(&sc))
	setArgMatrix(k, 1, m)
	err := k.EnqueueKernel(hw, m.Rows(), m.Cols(), false)
	if err != nil {
		panic(err)
	}
	return m
}

// Add method evaluates a + sc * b and puts the result in m.
func (m *opencl32) Add(m1, m2 Matrix, s float64) Matrix {
	checkEqualSize("blas:Add", m1, m2, m)
	a, b := m1.(*opencl32), m2.(*opencl32)
	sc := float32(s)
	k := sw[addKernel]
	k.SetArg(0, wSize, unsafe.Pointer(&sc))
	setArgMatrix(k, 1, a)
	setArgMatrix(k, 3, b)
	setArgMatrix(k, 5, m)
	err := k.EnqueueKernel(hw, m.Rows(), m.Cols(), false)
	if err != nil {
		panic(err)
	}
	return m
}

// Cmp method compares a and b and returns a matrix with 0 where they are equal else 1.
func (m *opencl32) Cmp(m1, m2 Matrix, epsilon float64) Matrix {
	checkEqualSize("blas:Cmp", m1, m2, m)
	a, b := m1.(*opencl32), m2.(*opencl32)
	eps := float32(epsilon)
	k := sw[cmpKernel]
	k.SetArg(0, wSize, unsafe.Pointer(&eps))
	setArgMatrix(k, 1, a)
	setArgMatrix(k, 3, b)
	setArgMatrix(k, 5, m)
	err := k.EnqueueKernel(hw, m.Rows(), m.Cols(), false)
	if err != nil {
		panic(err)
	}
	return m
}

// MulElem method performs element wise multiplication of the two input matrices and puts the output in m.
// If trans1, trans2 flag is set then input matrices are transposed first.
func (m *opencl32) MulElem(m1, m2 Matrix, trans1, trans2 bool) Matrix {
	a, b := m1.(*opencl32), m2.(*opencl32)
	ac, ar, bc, br, knum := transCL(a, b, trans1, trans2)
	if ac != bc || ar != br {
		panic("blas:MulElem - mismatch in no. of rows and columns in input matrices")
	}
	m.reshape(ar, ac, true)
	k := sw[mulElemKernel1+knum]
	setArgMatrix(k, 0, a)
	setArgMatrix(k, 2, b)
	setArgMatrix(k, 4, m)
	err := k.EnqueueKernel(hw, m.Rows(), m.Cols(), false)
	if err != nil {
		panic(err)
	}
	return m
}

// Mul method multiplies two matrices using regular matrix multiplication and puts the output in m.
// If trans1, trans2 flag is set then input matrices are transposed first.
func (m *opencl32) Mul(m1, m2 Matrix, trans1, trans2 bool) Matrix {
	a, b := m1.(*opencl32), m2.(*opencl32)
	ac, ar, bc, br, knum := transCL(a, b, trans1, trans2)
	if ac != br {
		panic("blas:Mul - mismatch in no. of rows and columns in input matrices")
	}
	m.reshape(ar, bc, true)
	k := sw[mulKernel1+knum]
	setArgMatrix(k, 0, a)
	setArgMatrix(k, 2, b)
	setArgMatrix(k, 4, m)
	err := k.EnqueueKernel(hw, m.Rows(), m.Cols(), false)
	if err != nil {
		panic(err)
	}
	return m
}

// Sum method calculates the sum of the values in the matrix
func (m *opencl32) Sum() float64 {
	var sum float32
	res := scl.NewBuffer(hw, cl.MEM_WRITE_ONLY, wSize, &sum)
	defer res.Release()
	k := sw[sumKernel]
	setArgMatrix(k, 0, m)
	k.SetArgBuffer(2, res)
	err := k.EnqueueKernel(hw, 1, 1, true)
	if err != nil {
		panic(err)
	}
	res.Read(hw)
	return float64(sum)
}

// MaxCol method gets the column number with the maximim value for each row of the input matrix.
func (m *opencl32) MaxCol(in Matrix) Matrix {
	a := in.(*opencl32)
	m.reshape(a.rows, 1, false)
	k := sw[maxColKernel]
	setArgMatrix(k, 0, a)
	setArgMatrix(k, 2, m)
	err := k.EnqueueKernel(hw, m.Rows(), 1, false)
	if err != nil {
		panic(err)
	}
	return m
}

// Norm method divides each element by the sum of the values in that row.
func (m *opencl32) Norm(in Matrix) Matrix {
	a := in.(*opencl32)
	m.reshape(a.rows, a.cols, false)
	k := sw[normKernel]
	setArgMatrix(k, 0, a)
	setArgMatrix(k, 2, m)
	err := k.EnqueueKernel(hw, m.Rows(), 1, false)
	if err != nil {
		panic(err)
	}
	return m
}

// UnaryCL type represents a user function of one variable.
type UnaryCL struct {
	*scl.Software
}

var unarySrc = `
__kernel void unary(Dims ad, __global float* a, Dims md, __global float* m) {
	ARG float x = a[P(ad,row,col)];
`

// NewUnaryCL function compiles a new function of one variable
func NewUnaryCL(text string) UnaryCL {
	src := srcHead + unarySrc + "m[P(md,row,col)] = " + text + "; }"
	sw, err := scl.Compile(hw, src, "unary", "")
	if err != nil {
		panic(err)
	}
	return UnaryCL{sw}
}

// Apply method applies a unary function to a matrix element wise.
func (fn UnaryCL) Apply(in, out Matrix) Matrix {
	a, b := in.(*opencl32), out.(*opencl32)
	b.reshape(a.rows, a.cols, false)
	k := fn.Software
	setArgMatrix(k, 0, a)
	setArgMatrix(k, 2, b)
	err := k.EnqueueKernel(hw, b.Rows(), b.Cols(), false)
	if err != nil {
		panic(err)
	}
	return b
}

// BinaryCL type represents a user function of two variables.
type BinaryCL struct {
	*scl.Software
}

var binarySrc = `
__kernel void binary(Dims ad, __global float* a, Dims bd, __global float* b, Dims md, __global float* m) {
	ARG float x = a[P(ad,row,col)]; float y = b[P(bd,row,col)];
`

// NewUnaryCL function compiles a new function of one variable
func NewBinaryCL(text string) BinaryCL {
	src := srcHead + binarySrc + "m[P(md,row,col)] = " + text + "; }"
	sw, err := scl.Compile(hw, src, "binary", "")
	if err != nil {
		panic(err)
	}
	return BinaryCL{sw}
}

// Apply method applies a unary function to a matrix
func (fn BinaryCL) Apply(m1, m2, out Matrix) Matrix {
	checkEqualSize("binary64:Apply", m1, m2, out)
	a, b, m := m1.(*opencl32), m2.(*opencl32), out.(*opencl32)
	k := fn.Software
	setArgMatrix(k, 0, a)
	setArgMatrix(k, 2, b)
	setArgMatrix(k, 4, m)
	err := k.EnqueueKernel(hw, m.Rows(), m.Cols(), false)
	if err != nil {
		panic(err)
	}
	return b
}

// utils
func transCL(a, b *opencl32, atrans, btrans bool) (ac, ar, bc, br int32, kernel int) {
	ac, ar, bc, br = a.cols, a.rows, b.cols, b.rows
	if atrans {
		ac, ar = ar, ac
		kernel++
	}
	if btrans {
		bc, br = br, bc
		kernel += 2
	}
	return
}

func setArgMatrix(k *scl.Software, argc uint32, m *opencl32) {
	k.SetArg(argc, dSize, unsafe.Pointer(&m.dims))
	k.SetArgBuffer(argc+1, m.buf)
}
