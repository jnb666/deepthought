package blas

import (
	"fmt"
	"github.com/go-gl/cl/v1.2/cl"
	"github.com/jnb666/deepthought/scl"
	"unsafe"
)

const (
	wSize    = 4  // word size
	dSize    = 16 // dims header size
	padSize  = 32 // pad matrix to this size
	trBlock  = 16 // block size for transpose kernel
	mulBlock = 32 // block size for matrix multiply kernel
	mulWG    = 16 // work group size for multiply kernel
)

var (
	hw *scl.Hardware
	sw []*scl.Software
)

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
	opts := fmt.Sprintf("-D TRBLK=%d -D TS=%d -D WPT=%d", trBlock, mulBlock, mulBlock/mulWG)
	var src string
	for i := range sw {
		if i >= mulKernel {
			src = srcHead + mulHead + source[i]
		} else {
			src = srcHead + source[i]
		}
		if sw[i], err = scl.Compile(hw, src, name[i], opts); err != nil {
			panic(fmt.Sprintf("error compiling %s : %s", name[i], err))
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
	nrows, ncols := int32(rows), int32(cols)
	prows, pcols := pad(nrows), pad(ncols)
	hostData := make([]float32, prows*pcols)
	buffer := scl.NewBuffer(hw, cl.MEM_READ_WRITE, int(prows*pcols*wSize), hostData)
	buffer.Write(hw)
	return &opencl32{
		dims:   dims{rows: nrows, cols: ncols, stride: pcols},
		buf:    buffer,
		data:   hostData,
		format: "%8.4f",
	}
}

func (m *opencl32) Release() {
	m.buf.Release()
}

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
		m.stride = pad(m.cols)
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
func (m *opencl32) Copy(in, ix Matrix) Matrix {
	a := in.(*opencl32)
	var k *scl.Software
	if ix == nil {
		m.reshape(a.rows, a.cols, false)
		k = sw[copyKernel]
		setArgMatrix(k, 0, a)
		setArgMatrix(k, 2, m)
	} else {
		mix := ix.(*opencl32)
		m.reshape(mix.rows, a.cols, false)
		k = sw[copyIxKernel]
		setArgMatrix(k, 0, a)
		setArgMatrix(k, 2, mix)
		setArgMatrix(k, 4, m)
	}
	err := k.EnqueueKernel(hw, globalWG(m), nil, false)
	if err != nil {
		panic(err)
	}
	return m
}

// Transpose method returns a transposed copy of the input matrix
func (m *opencl32) Transpose(in Matrix) Matrix {
	a := in.(*opencl32)
	m.reshape(a.cols, a.rows, true)
	k := sw[transKernel]
	setArgMatrix(k, 0, a)
	setArgMatrix(k, 2, m)
	sz := uint64(max(a.rows, a.cols))
	err := k.EnqueueKernel(hw, []uint64{sz, sz}, []uint64{trBlock, trBlock}, false)
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
	err := k.EnqueueKernel(hw, globalWG(m), nil, false)
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
	err := k.EnqueueKernel(hw, globalWG(m), nil, false)
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
	err := k.EnqueueKernel(hw, globalWG(m), nil, false)
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
	err := k.EnqueueKernel(hw, globalWG(m), nil, false)
	if err != nil {
		panic(err)
	}
	return m
}

// MulElem method performs element wise multiplication of the two input matrices and puts the output in m.
func (m *opencl32) MulElem(m1, m2 Matrix) Matrix {
	checkEqualSize("blas:Cmp", m1, m2, m)
	a, b := m1.(*opencl32), m2.(*opencl32)
	k := sw[mulElemKernel]
	setArgMatrix(k, 0, a)
	setArgMatrix(k, 2, b)
	setArgMatrix(k, 4, m)
	err := k.EnqueueKernel(hw, globalWG(m), nil, false)
	if err != nil {
		panic(err)
	}
	return m
}

// Mul method multiplies two matrices using regular matrix multiplication and puts the output in m.
// Second matrix must be transposed first before input to this routine.
func (m *opencl32) Mul(m1, m2 Matrix, aTrans, bTrans, oTrans bool) Matrix {
	a, b := m1.(*opencl32), m2.(*opencl32)
	ar, ac, br, bc := a.rows, a.cols, b.rows, b.cols
	if aTrans {
		ar, ac = ac, ar
	}
	if bTrans {
		br, bc = bc, br
	}
	if ac != br {
		panic("blas:Mul - mismatch in no. of rows and columns in input matrices")
	}
	if oTrans {
		m.reshape(bc, ar, true)
	} else {
		m.reshape(ar, bc, true)
	}
	kernel := mulKernel
	var trans uint32
	if aTrans {
		kernel += 1
	}
	if bTrans {
		kernel += 2
	}
	if oTrans {
		trans = 1
	}
	k := sw[kernel]
	setArgMatrix(k, 0, a)
	setArgMatrix(k, 2, b)
	setArgMatrix(k, 4, m)
	k.SetArg(6, 4, unsafe.Pointer(&trans))
	// local mem kernel
	gSize := []uint64{mulWG * uint64(pad(bc)) / mulBlock, mulWG * uint64(pad(ar)) / mulBlock}
	lSize := []uint64{mulWG, mulWG}
	//fmt.Printf("global size is %+v local size is %+v\n", gSize, lSize)
	err := k.EnqueueKernel(hw, gSize, lSize, false)
	if err != nil {
		panic(err)
	}
	return m
}

// Sum method calculates the sum of the values in the matrix
func (m *opencl32) Sum() float64 {
	gx := 1 + (int(m.cols)-1)/trBlock
	gy := 1 + (int(m.rows)-1)/trBlock
	sums := make([]float32, gx*gy)
	res := scl.NewBuffer(hw, cl.MEM_WRITE_ONLY, wSize*gx*gy, sums)
	defer res.Release()
	k := sw[sumKernel]
	setArgMatrix(k, 0, m)
	k.SetArgBuffer(2, res)
	gSize := []uint64{trBlock * uint64(gx), trBlock * uint64(gy)}
	lSize := []uint64{trBlock, trBlock}
	err := k.EnqueueKernel(hw, gSize, lSize, true)
	if err != nil {
		panic(err)
	}
	clErr := res.Read(hw)
	if clErr != cl.SUCCESS {
		panic(cl.ErrToStr(clErr))
	}
	sum := float32(0)
	for _, s := range sums {
		sum += s
	}
	return float64(sum)
}

// SumRows method returns a column vector with the sum of each row.
func (m *opencl32) SumRows(in Matrix) Matrix {
	a := in.(*opencl32)
	m.reshape(a.rows, 1, false)
	k := sw[sumRowsKernel]
	setArgMatrix(k, 0, a)
	setArgMatrix(k, 2, m)
	err := k.EnqueueKernel(hw, []uint64{uint64(m.rows)}, nil, false)
	if err != nil {
		panic(err)
	}
	return m
}

// MaxCol method gets the column number with the maximim value for each row of the input matrix.
func (m *opencl32) MaxCol(in Matrix) Matrix {
	a := in.(*opencl32)
	m.reshape(a.rows, 1, false)
	k := sw[maxColKernel]
	setArgMatrix(k, 0, a)
	setArgMatrix(k, 2, m)
	err := k.EnqueueKernel(hw, []uint64{uint64(m.rows)}, nil, false)
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
	err := k.EnqueueKernel(hw, []uint64{uint64(m.rows)}, nil, false)
	if err != nil {
		panic(err)
	}
	return m
}

// Histogram method adds bins the values from the input column vector and adds to the histogram.
func (m *opencl32) Histogram(in Matrix, bins int, min, max float64) Matrix {
	a := in.(*opencl32)
	ibins := int32(bins)
	fmin := float32(min)
	scale := float32(bins) / float32(max-min)
	m.reshape(ibins, 1, false)
	k := sw[histKernel]
	k.SetArg(0, wSize, unsafe.Pointer(&ibins))
	k.SetArg(1, wSize, unsafe.Pointer(&fmin))
	k.SetArg(2, wSize, unsafe.Pointer(&scale))
	setArgMatrix(k, 3, a)
	setArgMatrix(k, 5, m)
	k.SetArg(7, uint64(wSize*bins), nil)
	err := k.EnqueueKernel(hw, []uint64{uint64(a.rows)}, nil, false)
	if err != nil {
		panic(err)
	}
	return m
}

// UnaryCL type represents a user function of one variable.
type UnaryCL struct {
	*scl.Software
}

// NewUnaryCL function compiles a new function of one variable
func NewUnaryCL(text string) UnaryCL {
	src := srcHead + unarySrc + text + "m[P(md,row,col)] = y; }"
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
	err := k.EnqueueKernel(hw, globalWG(b), nil, false)
	if err != nil {
		panic(err)
	}
	return b
}

// BinaryCL type represents a user function of two variables.
type BinaryCL struct {
	*scl.Software
}

// NewUnaryCL function compiles a new function of one variable
func NewBinaryCL(text string) BinaryCL {
	src := srcHead + binarySrc + text + "m[P(md,row,col)] = z; }"
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
	err := k.EnqueueKernel(hw, globalWG(m), nil, false)
	if err != nil {
		panic(err)
	}
	return b
}

// Wait for any calcs to complete
func Sync() {
	if implementation == OpenCL32 {
		cl.Finish(hw.Queue)
	}
}

// utils
func pad(n int32) int32 {
	return padSize * (1 + (n-1)/padSize)
}

func globalWG(m *opencl32) []uint64 {
	return []uint64{uint64(m.cols), uint64(m.rows)}
}

func setArgMatrix(k *scl.Software, argc uint32, m *opencl32) {
	k.SetArg(argc, dSize, unsafe.Pointer(&m.dims))
	k.SetArgBuffer(argc+1, m.buf)
}

func max(a, b int32) int32 {
	if a > b {
		return a
	}
	return b
}
