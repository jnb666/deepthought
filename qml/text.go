package qml

import (
	"code.google.com/p/freetype-go/freetype"
	"code.google.com/p/freetype-go/freetype/truetype"
	"fmt"
	"github.com/gonum/plot/vg"
	"gopkg.in/qml.v1/gl/2.1"
	"gopkg.in/qml.v1/gl/glbase"
	"image"
	"io/ioutil"
	"os"
	"path/filepath"
	"sync"
)

const (
	lowChar  = 32
	highChar = 127
)

var (
	DefaultFontName  = "NimbusSanL-Bold"
	DefaultFontScale = 14
	fontCache        = map[string]*fontTex{}
	mutex            sync.Mutex
)

// A Font allows rendering of text to an OpenGL context.
type Font struct {
	*fontTex
	list uint32
}

type fontTex struct {
	tex    glbase.Texture
	chars  []glyph
	height int
	width  int
}

type glyph struct {
	x, adv int
}

// draw truetype font to an image
func getFontTex(gl *GL.GL, fontName string, scale int) *fontTex {
	f := new(fontTex)
	fontFile, err := fontPath(fontName)
	if err != nil {
		panic(err)
	}
	r, err := os.Open(fontFile)
	if err != nil {
		panic(err)
	}
	defer r.Close()
	data, err := ioutil.ReadAll(r)
	if err != nil {
		panic(err)
	}
	ttf, err := truetype.Parse(data)
	if err != nil {
		panic(err)
	}
	// get metrics
	gb := ttf.Bounds(int32(scale))
	f.width = int(gb.XMax - gb.XMin)
	f.height = int(gb.YMax - gb.YMin)
	// draw chars to image
	count := highChar - lowChar + 1
	f.chars = make([]glyph, count)
	imgX, imgY := pow2(count*f.width), pow2(f.height)
	img := image.NewRGBA(image.Rect(0, 0, imgX, imgY))
	c := freetype.NewContext()
	c.SetDPI(72)
	c.SetFont(ttf)
	c.SetFontSize(float64(scale))
	c.SetClip(img.Bounds())
	c.SetDst(img)
	c.SetSrc(image.White)
	gx := 0
	ch := rune(lowChar)
	for i := range f.chars {
		index := ttf.Index(ch)
		metric := ttf.HMetric(int32(scale), index)
		f.chars[i].adv = int(metric.AdvanceWidth)
		f.chars[i].x = int(gx)
		pt := freetype.Pt(int(gx), imgY-5)
		c.DrawString(string(ch), pt)
		gx += f.width
		ch++
	}
	// create texture
	tex := gl.GenTextures(1)
	f.tex = tex[0]
	gl.BindTexture(GL.TEXTURE_2D, f.tex)
	gl.TexParameteri(GL.TEXTURE_2D, GL.TEXTURE_MIN_FILTER, GL.NEAREST)
	gl.TexParameteri(GL.TEXTURE_2D, GL.TEXTURE_MAG_FILTER, GL.NEAREST)
	gl.TexImage2D(GL.TEXTURE_2D, 0, GL.RGBA, imgX, imgY, 0, GL.RGBA, GL.UNSIGNED_BYTE, img.Pix)
	return f
}

// LoadFont loads a truetype font from the given stream and applies the given font scale in points.
func loadFont(gl *GL.GL, fontName string, scale int, ptx, pty func(int) float32) *Font {
	mutex.Lock()
	defer mutex.Unlock()
	name := fmt.Sprintf("%s-%d", fontName, scale)
	if _, ok := fontCache[name]; !ok {
		fontCache[name] = getFontTex(gl, fontName, scale)
	}
	f := fontCache[name]
	// create command lists for this font
	count := highChar - lowChar + 1
	base := gl.GenLists(int32(count))
	imgX, imgY := pow2(count*f.width), pow2(f.height)
	ghf := pty(f.height)
	thf := float32(f.height) / float32(imgY)
	for i, g := range f.chars {
		gwf := ptx(g.adv)
		twf := float32(g.adv) / float32(imgX)
		txf := float32(g.x) / float32(imgX)
		gl.NewList(base+uint32(i), GL.COMPILE)
		gl.Begin(GL.QUADS)
		gl.TexCoord2f(txf, 1)
		gl.Vertex2f(0, -ghf)
		gl.TexCoord2f(txf+twf, 1)
		gl.Vertex2f(gwf, -ghf)
		gl.TexCoord2f(txf+twf, 1-thf)
		gl.Vertex2f(gwf, 0)
		gl.TexCoord2f(txf, 1-thf)
		gl.Vertex2f(0, 0)
		gl.End()
		gl.Translatef(gwf, 0, 0)
		gl.EndList()
	}
	if errno := gl.GetError(); errno != GL.NO_ERROR {
		panic(fmt.Errorf("GL error %d loading fonts\n", errno))
	}
	return &Font{
		fontTex: f,
		list:    base,
	}
}

// Height method returns font height in points
func (f *Font) Height() int {
	return f.height
}

// Size method returns the horizontal size of string in points
func (f *Font) Size(s string) (w int) {
	for _, ch := range s {
		w += f.chars[ch-lowChar].adv
	}
	return
}

// DrawText method draws the given string at the specified coordinates.
func (f *Font) DrawText(gl *GL.GL, xpos, ypos, angle float32, str string) error {
	if len(str) == 0 {
		return nil
	}
	indices := []rune(str)
	for i := range indices {
		indices[i] -= lowChar
	}
	gl.PushAttrib(GL.ENABLE_BIT)
	gl.Enable(GL.TEXTURE_2D)
	gl.Enable(GL.BLEND)
	gl.BlendFunc(GL.SRC_ALPHA, GL.ONE_MINUS_SRC_ALPHA)
	gl.TexEnvf(GL.TEXTURE_ENV, GL.TEXTURE_ENV_MODE, GL.MODULATE)
	gl.BindTexture(GL.TEXTURE_2D, f.tex)

	gl.PushMatrix()
	gl.Translatef(xpos, ypos, 0)
	if angle != 0 {
		gl.Rotatef(angle, 0, 0, 1)
	}
	gl.ListBase(f.list)
	gl.CallLists(len(indices), GL.UNSIGNED_INT, indices)
	gl.PopMatrix()
	gl.PopAttrib()
	if errno := gl.GetError(); errno != GL.NO_ERROR {
		return fmt.Errorf("GL error %d drawing text\n", errno)
	}
	return nil
}

// pow2 returns the first power-of-two value >= to n.
// This can be used to create suitable texture dimensions.
func pow2(x int) int {
	x--
	x |= x >> 1
	x |= x >> 2
	x |= x >> 4
	x |= x >> 8
	x |= x >> 16
	return int(x + 1)
}

// locate font file on disk
func fontPath(name string) (string, error) {
	name += ".ttf"
	for _, d := range vg.FontDirs {
		p := filepath.Join(d, name)
		if _, err := os.Stat(p); err != nil {
			continue
		}
		return p, nil
	}
	return "", fmt.Errorf("Failed to locate a font file %s", name)
}
