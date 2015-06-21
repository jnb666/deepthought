package mnist

import (
	//"fmt"
	"github.com/jnb666/deepthought/blas"
	"github.com/jnb666/deepthought/network"
	"math"
)

const (
	size         = 28
	size2        = size * size
	severity     = 1.0
	scale        = 0.15
	rotate       = 15.0 * math.Pi / 180
	elasticSigma = 8.0
	elasticScale = 0.5
	kernelSize   = 21
)

var (
	sina, cosa blas.UnaryFunction
)

// supported types of distortion
const (
	Scale   = 1
	Rotate  = 2
	Elastic = 4
)

type Loader struct {
	img    blas.Image
	xm, ym blas.Matrix
	xscale blas.Matrix
	yscale blas.Matrix
	angle  blas.Matrix
	sina   blas.Matrix
	cosa   blas.Matrix
	tx, ty blas.Matrix
	txy    blas.Matrix
	kernel blas.Matrix
	ux, uy blas.Matrix
	dx, dy blas.Matrix
}

func (l *Loader) init(batch int) {
	l.img = blas.NewImage(size, size, batch)
	// image mid point
	l.xm = blas.New(batch, size2)
	l.ym = blas.New(batch, size2)
	xdata := make([]float64, size2)
	ydata := make([]float64, size2)
	for y := 0; y < size; y++ {
		for x := 0; x < size; x++ {
			xdata[y*size+x] = -float64(size-1)/2 + float64(x)
			ydata[y*size+x] = -float64(size-1)/2 + float64(y)
		}
	}
	l.xm.Load(blas.RowMajor, xdata...)
	l.ym.Load(blas.RowMajor, ydata...)
	// arrays for scaling and rotation
	l.xscale = blas.New(1, batch)
	l.yscale = blas.New(1, batch)
	l.angle = blas.New(1, batch)
	l.sina = blas.New(1, batch)
	l.cosa = blas.New(1, batch)
	sina = blas.NewUnaryCL("float y = sin(x);")
	cosa = blas.NewUnaryCL("float y = cos(x)-1.f;")
	l.tx = blas.New(batch, size2)
	l.ty = blas.New(batch, size2)
	l.txy = blas.New(batch, size2)
	// convolution kernel and elastic distortion arrays
	l.kernel = blas.GaussianKernel(kernelSize, kernelSize, elasticSigma)
	//l.kernel.SetFormat("%4.2f")
	//fmt.Printf("convolve kernel:\n%s\n", l.kernel)
	l.ux = blas.New(batch, size2)
	l.uy = blas.New(batch, size2)
	l.dx = blas.New(batch, size2)
	l.dy = blas.New(batch, size2)
}

// DistortTypes returns the supported types of distortions
func (l *Loader) DistortTypes() []network.Distortion {
	return []network.Distortion{
		{Scale, "scale"},
		{Rotate, "rotate"},
		{Elastic, "elastic"},
	}
}

// Distort method is used to apply distortions to a batch of images.
// mask of -1 indicates all distortions are to be applied.
func (l *Loader) Distort(in, out blas.Matrix, mask int, severity float64) {
	//fmt.Println("distort: severity=", severity, "mask=", mask, "batch=", in.Rows())
	if l.img == nil {
		l.init(in.Rows())
	}
	l.img.Load(in)
	if mask < 0 || mask&Elastic != 0 {
		l.ux.Random(-1, 1)
		l.uy.Random(-1, 1)
		blas.Filter(size, l.kernel, l.ux, l.uy, l.dx, l.dy)
		l.dx.Scale(elasticScale * severity)
		l.dy.Scale(elasticScale * severity)
		//l.dx.SetFormat("%5.2f")
		//l.dy.SetFormat("%5.2f")
		//fmt.Printf("elastic:\n%s\n%s\n", l.dx, l.dy)
	} else {
		l.dx.Set(0)
		l.dy.Set(0)
	}
	if mask < 0 || mask&Scale != 0 {
		l.xscale.Random(-severity*scale, severity*scale)
		l.yscale.Random(-severity*scale, severity*scale)
		//fmt.Printf("scale %s %s\n", l.xscale, l.yscale)
	} else {
		l.xscale.Set(0)
		l.yscale.Set(0)
	}
	if mask < 0 || mask&Rotate != 0 {
		l.angle.Random(-severity*rotate, severity*rotate)
		//fmt.Printf("rotate %s\n", l.angle)
		sina.Apply(l.angle, l.sina)
		cosa.Apply(l.angle, l.cosa)
		l.xscale.Add(l.xscale, l.cosa, 1)
		l.yscale.Add(l.yscale, l.cosa, 1)
	} else {
		l.sina.Set(0)
	}
	if mask < 0 || mask&Scale+mask&Rotate != 0 {
		l.tx.Mul(l.xscale, l.xm, false, false, false)
		l.txy.Mul(l.sina, l.ym, false, false, false)
		l.dx.Add(l.tx, l.txy, -1)
		l.ty.Mul(l.yscale, l.ym, false, false, false)
		l.txy.Mul(l.sina, l.xm, false, false, false)
		l.dy.Add(l.ty, l.txy, 1)
	}
	l.img.Approx(l.dx, l.dy, out)
}

func (l *Loader) Release() {
	if l.img != nil {
		l.img.Release()
		l.xm.Release()
		l.ym.Release()
		l.xscale.Release()
		l.yscale.Release()
		l.angle.Release()
		l.cosa.Release()
		l.sina.Release()
		l.tx.Release()
		l.ty.Release()
		l.txy.Release()
		l.kernel.Release()
		l.ux.Release()
		l.uy.Release()
		l.dx.Release()
		l.dy.Release()
	}
	l.img = nil
}
