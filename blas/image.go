package blas

import (
	"github.com/go-gl/cl/v1.2/cl"
	"math"
	"unsafe"
)

// Image interface type represents a 2 dimensional image.
type Image interface {
	Load(Matrix) Image
	Approx(xv, yv, out Matrix) Image
	Release()
}

type imagecl struct {
	width  int
	height int
	nimage int
	buf    cl.Mem
}

// NewImage creates a new OpenCL image array
func NewImage(width, height, nimage int) Image {
	var err cl.ErrorCode
	if implementation != OpenCL32 {
		panic("image only implemented for OpenCL32!")
	}
	format := cl.ImageFormat{cl.R, cl.FLOAT}
	desc := cl.ImageDesc{
		ImageType:      cl.MEM_OBJECT_IMAGE2D_ARRAY,
		ImageWidth:     uint64(width),
		ImageHeight:    uint64(height),
		ImageArraySize: uint64(nimage),
	}
	img := cl.CreateImage(hw.Context, cl.MEM_READ_WRITE, format, desc, nil, &err)
	if err != cl.SUCCESS {
		panic(cl.ErrToStr(err))
	}
	return &imagecl{
		width:  width,
		height: height,
		nimage: nimage,
		buf:    img,
	}
}

func (img *imagecl) Release() {
	cl.ReleaseMemObject(img.buf)
}

// Load from a buffer into the image array where each row in the buffer contains an image
func (img *imagecl) Load(m Matrix) Image {
	in := m.(*opencl32)
	k := sw[loadImageKernel]
	k.SetArg(0, 8, unsafe.Pointer(&img.buf))
	setArgMatrix(k, 1, in)
	gSize := []uint64{uint64(img.width), uint64(img.height), uint64(img.nimage)}
	err := k.EnqueueKernel(hw, gSize, nil, false)
	if err != nil {
		panic(err)
	}
	return img
}

// Apply 2D linear interpolation to and copy results to out matrix
func (img *imagecl) Approx(xv, yv, out Matrix) Image {
	x, y, m := xv.(*opencl32), yv.(*opencl32), out.(*opencl32)
	k := sw[approxKernel]
	k.SetArg(0, 8, unsafe.Pointer(&img.buf))
	setArgMatrix(k, 1, x)
	setArgMatrix(k, 3, y)
	setArgMatrix(k, 5, m)
	gSize := []uint64{uint64(img.width), uint64(img.height), uint64(img.nimage)}
	err := k.EnqueueKernel(hw, gSize, nil, false)
	if err != nil {
		panic(err)
	}
	return img
}

// Apply a convolution kernel to a distribution to generate a set of distorion vectors
func Filter(width int, kernel, xin, yin, xout, yout Matrix) {
	kern := kernel.(*opencl32)
	checkEqualSize("blas:Filter", xin, yin, xout)
	yout.Reshape(yin.Rows(), yin.Cols(), false)
	x1, y1 := xin.(*opencl32), yin.(*opencl32)
	x2, y2 := xout.(*opencl32), yout.(*opencl32)
	k := sw[filterKernel]
	setArgMatrix(k, 0, x1)
	setArgMatrix(k, 2, y1)
	setArgMatrix(k, 4, x2)
	setArgMatrix(k, 6, y2)
	setArgMatrix(k, 8, kern)
	gSize := []uint64{uint64(width), uint64(x1.Cols() / width), uint64(x1.rows)}
	err := k.EnqueueKernel(hw, gSize, nil, false)
	if err != nil {
		panic(err)
	}
}

// Create a gaussian kernel with given size and standard deviation, xs and ys should be odd
func GaussianKernel(xs, ys int, sigma float64) Matrix {
	xmid, ymid := float64(xs/2), float64(ys/2)
	twoPiSigma := math.Sqrt(2*math.Pi) / sigma
	twoSigmaSq := 1.0 / (2 * sigma * sigma)
	data := make([]float64, xs*ys)
	for y := 0; y < ys; y++ {
		for x := 0; x < xs; x++ {
			xf, yf := float64(x), float64(y)
			data[y*xs+x] = twoPiSigma * math.Exp(-twoSigmaSq*((xf-xmid)*(xf-xmid)+(yf-ymid)*(yf-ymid)))
		}
	}
	return New(xs, ys).Load(ColMajor, data...)
}
