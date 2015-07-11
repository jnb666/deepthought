package blas

import (
	"fmt"
	"github.com/go-gl/cl/v1.2/cl"
	"github.com/jnb666/deepthought/scl"
	"sync"
	"unsafe"
)

const (
	wSize      = 4          // word size
	dSize      = 16         // dims header size
	padSize    = 16         // pad matrix to this size
	trBlock    = 16         // block size for transpose kernel
	mulBlock   = 16         // block size for matrix multiply kernel
	randSize   = 256        // no. of concurrent random seeds
	mwc_A      = 4294883355 // params for random no. generator
	mwc_M      = 18446383549859758079
	mwc_BASEID = 4077358422479273989
)

var (
	hw        *scl.Hardware
	sw        []*scl.Software
	seeds     [2]*scl.Buffer
	seedBuf   []seed
	seedIx    int
	randMutex sync.Mutex
)

type seed struct{ hi, lo uint32 }

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
	opts := fmt.Sprintf("-D TRBLK=%d -D TS=%d -D MWC_A=%dU", trBlock, mulBlock, mwc_A)
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
	buffer.Clear(hw)
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
func (m *opencl32) Load(order Ordering, vals ...float32) Matrix {
	var row, col int32
	if len(vals) == 0 {
		panic("blas:Load - no data provided")
	}
	next := getNext(vals)
	if order == RowMajor {
		for row = 0; row < m.rows; row++ {
			for col = 0; col < m.cols; col++ {
				m.data[m.base+row*m.stride+col] = next()
			}
		}
	} else {
		for col = 0; col < m.cols; col++ {
			for row = 0; row < m.rows; row++ {
				m.data[m.base+row*m.stride+col] = next()
			}
		}
	}
	m.buf.Write(hw)
	return m
}

// Data method returns a copy of the matrix data as a slice
func (m *opencl32) Data(order Ordering) []float32 {
	var row, col int32
	m.buf.Read(hw)
	data := make([]float32, m.rows*m.cols)
	i := 0
	if order == RowMajor {
		for row = 0; row < m.rows; row++ {
			for col = 0; col < m.cols; col++ {
				data[i] = m.data[m.base+row*m.stride+col]
				i++
			}
		}
	} else {
		for col = 0; col < m.cols; col++ {
			for row = 0; row < m.rows; row++ {
				data[i] = m.data[m.base+row*m.stride+col]
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

// Seed the random number generator seeds
func initSeeds(base int32) {
	seedBuf = make([]seed, randSize)
	for i := range seedBuf {
		m := powMod64(mwc_A, uint64(base)+uint64(i)<<32, mwc_M)
		x := mulMod64(mwc_BASEID, m, mwc_M)
		seedBuf[i].hi = uint32(x / mwc_A)
		seedBuf[i].lo = uint32(x % mwc_A)
	}
	seeds[0] = scl.NewBuffer(hw, cl.MEM_READ_WRITE, 8*randSize, seedBuf)
	seeds[0].Write(hw)
	seeds[1] = scl.NewBuffer(hw, cl.MEM_READ_WRITE, 8*randSize, nil)
}

// Random method initialises a matrix with random values in range 0-1.
func (m *opencl32) Random(min, max float32) Matrix {
	if seeds[0] == nil {
		panic("random seeds not initialised - call SeedRandom")
	}
	randMutex.Lock()
	defer randMutex.Unlock()
	rrange := max - min
	k := sw[randomKernel]
	setArgMatrix(k, 0, m)
	k.SetArg(2, 4, unsafe.Pointer(&min))
	k.SetArg(3, 4, unsafe.Pointer(&rrange))
	k.SetArgBuffer(4, seeds[seedIx])
	k.SetArgBuffer(5, seeds[1-seedIx])
	globalSize := uint64(m.rows * m.cols)
	localSize := globalSize
	if localSize > randSize {
		localSize = randSize
	}
	k.EnqueueKernel(hw, []uint64{globalSize}, []uint64{localSize})
	seedIx = 1 - seedIx
	return m
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
	k.EnqueueKernel(hw, globalWG(m), nil)
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
	k.EnqueueKernel(hw, []uint64{sz, sz}, []uint64{trBlock, trBlock})
	return m
}

// Set method sets all elements of the matrix to the given value
func (m *opencl32) Set(val float32) Matrix {
	k := sw[setKernel]
	k.SetArg(0, wSize, unsafe.Pointer(&val))
	setArgMatrix(k, 1, m)
	k.EnqueueKernel(hw, globalWG(m), nil)
	return m
}

// Scale method muliplies each element of the matrix by a scalar.
func (m *opencl32) Scale(sc float32) Matrix {
	k := sw[scaleKernel]
	k.SetArg(0, wSize, unsafe.Pointer(&sc))
	setArgMatrix(k, 1, m)
	k.EnqueueKernel(hw, globalWG(m), nil)
	return m
}

// Add method evaluates a + sc * b and puts the result in m.
func (m *opencl32) Add(m1, m2 Matrix, sc float32) Matrix {
	checkEqualSize("blas:Add", m1, m2, m)
	a, b := m1.(*opencl32), m2.(*opencl32)
	k := sw[addKernel]
	k.SetArg(0, wSize, unsafe.Pointer(&sc))
	setArgMatrix(k, 1, a)
	setArgMatrix(k, 3, b)
	setArgMatrix(k, 5, m)
	k.EnqueueKernel(hw, globalWG(m), nil)
	return m
}

// Cmp method compares a and b and returns a matrix with 0 where they are equal else 1.
func (m *opencl32) Cmp(m1, m2 Matrix, eps float32) Matrix {
	checkEqualSize("blas:Cmp", m1, m2, m)
	a, b := m1.(*opencl32), m2.(*opencl32)
	k := sw[cmpKernel]
	k.SetArg(0, wSize, unsafe.Pointer(&eps))
	setArgMatrix(k, 1, a)
	setArgMatrix(k, 3, b)
	setArgMatrix(k, 5, m)
	k.EnqueueKernel(hw, globalWG(m), nil)
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
	k.EnqueueKernel(hw, globalWG(m), nil)
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
	k.EnqueueKernel(hw, []uint64{uint64(pad(bc)), uint64(pad(ar))}, []uint64{mulBlock, mulBlock})
	return m
}

// Sum method calculates the sum of the values in the matrix
func (m *opencl32) Sum() float32 {
	gx := 1 + (int(m.cols)-1)/trBlock
	gy := 1 + (int(m.rows)-1)/trBlock
	sums := make([]float32, gx*gy)
	res := scl.NewBuffer(hw, cl.MEM_WRITE_ONLY, wSize*gx*gy, sums)
	defer res.Release()
	k := sw[sumKernel]
	setArgMatrix(k, 0, m)
	k.SetArgBuffer(2, res)
	gSize := []uint64{trBlock * uint64(gx), trBlock * uint64(gy)}
	k.EnqueueKernel(hw, gSize, []uint64{trBlock, trBlock})
	res.Read(hw)
	sum := float32(0)
	for _, s := range sums {
		sum += s
	}
	return sum
}

// SumRows method returns a column vector with the sum of each row.
func (m *opencl32) SumRows(in Matrix) Matrix {
	a := in.(*opencl32)
	m.reshape(a.rows, 1, false)
	k := sw[sumRowsKernel]
	setArgMatrix(k, 0, a)
	setArgMatrix(k, 2, m)
	k.EnqueueKernel(hw, []uint64{uint64(m.rows)}, nil)
	return m
}

// MaxCol method gets the column number with the maximim value for each row of the input matrix.
func (m *opencl32) MaxCol(in Matrix) Matrix {
	a := in.(*opencl32)
	m.reshape(a.rows, 1, false)
	k := sw[maxColKernel]
	setArgMatrix(k, 0, a)
	setArgMatrix(k, 2, m)
	k.EnqueueKernel(hw, []uint64{uint64(m.rows)}, nil)
	return m
}

// Norm method divides each element by the sum of the values in that row.
func (m *opencl32) Norm(in Matrix) Matrix {
	a := in.(*opencl32)
	m.reshape(a.rows, a.cols, false)
	k := sw[normKernel]
	setArgMatrix(k, 0, a)
	setArgMatrix(k, 2, m)
	k.EnqueueKernel(hw, []uint64{uint64(m.rows)}, nil)
	return m
}

// Histogram method adds bins the values from the input column vector and adds to the histogram.
func (m *opencl32) Histogram(in Matrix, bins int, min, max float32) Matrix {
	a := in.(*opencl32)
	ibins := int32(bins)
	scale := float32(bins) / (max - min)
	m.reshape(ibins, 1, false)
	k := sw[histKernel]
	k.SetArg(0, wSize, unsafe.Pointer(&ibins))
	k.SetArg(1, wSize, unsafe.Pointer(&min))
	k.SetArg(2, wSize, unsafe.Pointer(&scale))
	setArgMatrix(k, 3, a)
	setArgMatrix(k, 5, m)
	k.SetArg(7, uint64(wSize*bins), nil)
	k.EnqueueKernel(hw, []uint64{uint64(a.rows)}, nil)
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
	k.EnqueueKernel(hw, globalWG(b), nil)
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
	k.EnqueueKernel(hw, globalWG(m), nil)
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

func pad64(n int) uint64 {
	return uint64(padSize * (1 + (n-1)/padSize))
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
