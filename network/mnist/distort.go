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
	rotate       = 15.0 * math.Pi / 180.0
	elasticSigma = 8.0
	elasticScale = 0.5
	kernelSize   = 21
)

// supported types of distortion
const (
	Scale   = 1
	Rotate  = 2
	Elastic = 4
)

type Loader struct {
	img    blas.Image
	xscale blas.Matrix
	yscale blas.Matrix
	angle  blas.Matrix
	kernel blas.Matrix
	ux, uy blas.Matrix
	dx, dy blas.Matrix
}

func (l *Loader) init(batch int) {
	l.img = blas.NewImage(size, size, batch)
	// arrays for scaling and rotation
	l.xscale = blas.New(1, batch)
	l.yscale = blas.New(1, batch)
	l.angle = blas.New(1, batch)
	// convolution kernel and elastic distortion arrays
	l.kernel = blas.GaussianKernel(kernelSize, kernelSize, elasticSigma)
	//l.kernel.SetFormat("%4.2f")
	//fmt.Printf("convolve kernel:\n%s\n", l.kernel)
	l.ux = blas.New(batch, size2)
	l.uy = blas.New(batch, size2)
	// distortion map in x and y dimensions
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
	l.img.Import(in)
	if mask < 0 || mask&Elastic != 0 {
		l.ux.Random(-1, 1)
		l.uy.Random(-1, 1)
		l.img.Filter(l.kernel, l.ux, l.uy, l.dx, l.dy)
		l.dx.Scale(elasticScale * severity)
		l.dy.Scale(elasticScale * severity)
		//l.dx.SetFormat("%5.2f")
		//l.dy.SetFormat("%5.2f")
		//fmt.Printf("elastic:\n%s\n%s\n", l.dx, l.dy)
	} else {
		l.dx.Set(0)
		l.dy.Set(0)
	}
	if mask < 0 || mask&Rotate != 0 {
		l.angle.Random(-severity*rotate, severity*rotate)
		//fmt.Printf("rotate %s\n", l.angle)
		l.img.Rotate(l.angle, l.dx, l.dy)
	}
	if mask < 0 || mask&Scale != 0 {
		l.xscale.Random(-severity*scale, severity*scale)
		l.yscale.Random(-severity*scale, severity*scale)
		//fmt.Printf("scale %s %s\n", l.xscale, l.yscale)
		l.img.Scale(l.xscale, l.yscale, l.dx, l.dy)
	}
	l.img.Export(l.dx, l.dy, out)
}

func (l *Loader) Release() {
	if l.img != nil {
		l.img.Release()
		l.xscale.Release()
		l.yscale.Release()
		l.angle.Release()
		l.kernel.Release()
		l.ux.Release()
		l.uy.Release()
		l.dx.Release()
		l.dy.Release()
	}
	l.img = nil
}
