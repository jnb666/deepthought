package blas

import (
	"github.com/go-gl/cl/v1.2/cl"
	"math"
	"unsafe"
)

// Image interface type represents a 2 dimensional image.
type Image interface {
	Import(in Matrix)
	Export(xv, yv, out Matrix)
	SetOrigin(x0, y0 float64) Image
	Filter(kernel, xv, yv, dx, dy Matrix)
	Scale(xscale, yscale, dx, dy Matrix)
	Rotate(angle, dx, dy Matrix)
	Release()
}

type imagecl struct {
	width  int
	height int
	nimage int
	buf    cl.Mem
	x0, y0 float32
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
		x0:     float32(width-1) / 2,
		y0:     float32(height-1) / 2,
	}
}

func (img *imagecl) Release() {
	cl.ReleaseMemObject(img.buf)
}

// Load from a buffer into the image array where each row in the buffer contains an image
func (img *imagecl) Import(m Matrix) {
	in := m.(*opencl32)
	k := sw[loadImageKernel]
	k.SetArg(0, 8, unsafe.Pointer(&img.buf))
	setArgMatrix(k, 1, in)
	gSize := []uint64{uint64(img.width), uint64(img.height), uint64(img.nimage)}
	err := k.EnqueueKernel(hw, gSize, nil, false)
	if err != nil {
		panic(err)
	}
}

// Apply 2D linear interpolation to and copy results to out matrix
func (img *imagecl) Export(xv, yv, out Matrix) {
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
}

// Set the image center point for scaling and rotation
func (img *imagecl) SetOrigin(x0, y0 float64) Image {
	img.x0, img.y0 = float32(x0), float32(y0)
	return img
}

// Generate distortion vectors for a scaling transformation
func (img *imagecl) Scale(xscale, yscale, dx, dy Matrix) {
	sx, sy := xscale.(*opencl32), yscale.(*opencl32)
	xd, yd := dx.(*opencl32), dy.(*opencl32)
	k := sw[scaleImageKernel]
	k.SetArg(0, 4, unsafe.Pointer(&img.x0))
	k.SetArg(1, 4, unsafe.Pointer(&img.y0))
	setArgMatrix(k, 2, sx)
	setArgMatrix(k, 4, sy)
	setArgMatrix(k, 6, xd)
	setArgMatrix(k, 8, yd)
	gSize := []uint64{uint64(img.width), uint64(img.height), uint64(img.nimage)}
	err := k.EnqueueKernel(hw, gSize, nil, false)
	if err != nil {
		panic(err)
	}
}

// Generate distortion vectors for rotation transformation
func (img *imagecl) Rotate(angle, dx, dy Matrix) {
	ang := angle.(*opencl32)
	xd, yd := dx.(*opencl32), dy.(*opencl32)
	k := sw[rotateKernel]
	k.SetArg(0, 4, unsafe.Pointer(&img.x0))
	k.SetArg(1, 4, unsafe.Pointer(&img.y0))
	setArgMatrix(k, 2, ang)
	setArgMatrix(k, 4, xd)
	setArgMatrix(k, 6, yd)
	gSize := []uint64{uint64(img.width), uint64(img.height), uint64(img.nimage)}
	err := k.EnqueueKernel(hw, gSize, nil, false)
	if err != nil {
		panic(err)
	}
}

// Apply a convolution kernel to a distribution to generate a set of distorion vectors
func (img *imagecl) Filter(kernel, xin, yin, xout, yout Matrix) {
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
	gSize := []uint64{uint64(img.width), uint64(x1.Cols() / img.width), uint64(x1.rows)}
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
