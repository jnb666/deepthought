package blas

import (
	"fmt"
	"github.com/go-gl/cl/v1.2/cl"
	"github.com/jnb666/deepthought/scl"
	"math"
	"unsafe"
)

// Image interface type represents a 2 dimensional image.
type Image interface {
	Import(in ...Matrix)
	Export(xv, yv, out Matrix)
	SetOrigin(x0, y0 float32) Image
	Scale(xscale, yscale, dx, dy Matrix)
	Rotate(angle, dx, dy Matrix)
	Release()
}

type imagecl struct {
	width    int
	height   int
	nimage   int
	channels int
	buf      cl.Mem
	x0, y0   float32
}

// NewImage creates a new OpenCL image array
func NewImage(width, height, nimage, channels int) Image {
	var err cl.ErrorCode
	if implementation != OpenCL32 {
		panic("image only implemented for OpenCL32!")
	}
	if channels < 1 || channels > 2 {
		panic("image must have 1 or 2 channels!")
	}
	format := cl.ImageFormat{
		ImageChannelOrder:    []cl.ChannelOrder{0, cl.R, cl.RG}[channels],
		ImageChannelDataType: cl.FLOAT,
	}
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
		width:    width,
		height:   height,
		nimage:   nimage,
		channels: channels,
		buf:      img,
		x0:       float32(width-1) / 2,
		y0:       float32(height-1) / 2,
	}
}

func (img *imagecl) Release() {
	cl.ReleaseMemObject(img.buf)
}

// Load from a buffer into the image array where each row in the buffer contains an image
func (img *imagecl) Import(m ...Matrix) {
	k := sw[loadImageKernel+img.channels-1]
	k.SetArg(0, 8, unsafe.Pointer(&img.buf))
	for i := 0; i < img.channels; i++ {
		setArgMatrix(k, uint32(2*i+1), m[i].(*opencl32))
	}
	k.EnqueueKernel(hw, []uint64{uint64(img.width), uint64(img.height), uint64(img.nimage)}, nil)
}

// Apply 2D linear interpolation to and copy results to out matrix
func (img *imagecl) Export(xv, yv, out Matrix) {
	x, y, m := xv.(*opencl32), yv.(*opencl32), out.(*opencl32)
	k := sw[approxKernel]
	k.SetArg(0, 8, unsafe.Pointer(&img.buf))
	setArgMatrix(k, 1, x)
	setArgMatrix(k, 3, y)
	setArgMatrix(k, 5, m)
	k.EnqueueKernel(hw, []uint64{uint64(img.width), uint64(img.height), uint64(img.nimage)}, nil)
}

// Set the image center point for scaling and rotation
func (img *imagecl) SetOrigin(x, y float32) Image {
	img.x0, img.y0 = x, y
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
	k.EnqueueKernel(hw, []uint64{uint64(img.width), uint64(img.height), uint64(img.nimage)}, nil)
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
	k.EnqueueKernel(hw, []uint64{uint64(img.width), uint64(img.height), uint64(img.nimage)}, nil)
}

type Filter struct {
	*scl.Software
}

// Create a new filter with given size
func NewFilter(size int) Filter {
	src := srcHead + filterHead
	for i := 0; i < size; i++ {
		src += fmt.Sprintf(filterLoop, i, i)
	}
	src += filterTail
	//fmt.Println(src)
	opts := fmt.Sprintf("-D FILTER_SIZE=%d -D FILTER_CENTER=%d.5f -D FILTER_STRIDE=%d", size, size/2, pad(int32(size)))
	sw, err := scl.Compile(hw, src, "filter", opts)
	if err != nil {
		panic(err)
	}
	return Filter{sw}
}

// Apply a convolution kernel to a distribution to generate a set of distorion vectors
func (f *Filter) Apply(in Image, kernel, dx, dy Matrix) {
	kern := kernel.(*opencl32)
	img := in.(*imagecl)
	m1, m2 := dx.(*opencl32), dy.(*opencl32)
	m1.Reshape(img.nimage, img.width*img.height, false)
	m2.Reshape(img.nimage, img.width*img.height, false)
	k := f.Software
	k.SetArg(0, 8, unsafe.Pointer(&img.buf))
	setArgMatrix(k, 1, kern)
	setArgMatrix(k, 3, m1)
	setArgMatrix(k, 5, m2)
	globalSize := []uint64{uint64(img.width), uint64(img.height), uint64(img.nimage)}
	k.EnqueueKernel(hw, globalSize, nil)
}

// Create a gaussian kernel with given size and standard deviation, xs and ys should be odd
func GaussianKernel(xs, ys int, sigma float64) Matrix {
	xmid, ymid := float64(xs/2), float64(ys/2)
	twoPiSigma := math.Sqrt(2*math.Pi) / sigma
	twoSigmaSq := 1.0 / (2 * sigma * sigma)
	data := make([]float32, xs*ys)
	for y := 0; y < ys; y++ {
		for x := 0; x < xs; x++ {
			xf, yf := float64(x), float64(y)
			val := twoPiSigma * math.Exp(-twoSigmaSq*((xf-xmid)*(xf-xmid)+(yf-ymid)*(yf-ymid)))
			data[y*xs+x] = float32(val)
		}
	}
	return New(xs, ys).Load(ColMajor, data...)
}
