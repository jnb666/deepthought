package scl

import (
	"fmt"
	"github.com/go-gl/cl/v1.2/cl"
	"math"
	"strings"
	"testing"
)

var goodCode = `
__kernel void good(__global float *a, __global float *b, __global float *c) {
	*c = *a + *b;
}
`
var badCode = `
__kernel void bad(__global float *a, __global float *b, __global float *c) {
	*c = *a + *d;
}
`
var helloCode = `
__kernel void hello(__global char* buf, __global char* buf2) {
	int i = get_global_id(0);
	buf2[i] = buf[i];
}
`
var matrixCode = `
__kernel void matvec_mult(__global float4* matrix, __global float4* vector, __global float* result) {
	int i = get_global_id(0);
	result[i] = dot(matrix[i], vector[0]);
}
`
var matrixCodeDouble = `
__kernel void matvec_mult(__global double4* matrix, __global double4* vector, __global double* result) {
	int i = get_global_id(0);
	result[i] = dot(matrix[i], vector[0]);
}
`
var scaleCode = `
__kernel void scalevec(__global float *vector, float value, __local float *valueLocal) {
	int local_x = get_local_id(0);
	int global_x = get_global_id(0);
	if (local_x == 0) *valueLocal=value;
	barrier(CLK_LOCAL_MEM_FENCE); 
	vector[global_x] = vector[global_x] * *valueLocal;
}
`

func TestDevices(t *testing.T) {
	devs := Devices()
	if len(devs) == 0 {
		t.Error("no devices found!")
	}
	for _, d := range devs {
		t.Log(d)
	}
}

func TestSelect(t *testing.T) {
	devs := Devices()
	hw := devs.Select(0)
	t.Log("best:", hw)
	if hw.Type&cl.DEVICE_TYPE_GPU == 0 {
		t.Error("expecting GPU type")
	} else {
		hw.Release()
	}
	hw = devs.Select(cl.DEVICE_TYPE_CPU)
	t.Log("bestCPU:", hw)
	if hw.Type&cl.DEVICE_TYPE_CPU == 0 {
		t.Error("expecting CPU type")
	} else {
		hw.Release()
	}
}

func TestBuffer(t *testing.T) {
	chars := []byte("Hello world")
	hw := Devices().Select(0)
	buf := NewBuffer(hw, cl.MEM_READ_ONLY, len(chars), chars)
	buf.Write(hw)
	t.Log(buf)
	if buf.String() != "<< size=11: 72 101 108 108 111 32 119 111 114 108 100 >>" {
		t.Error("buffer data invalid!")
	}
	buf.Release()
	hw.Release()
}

func TestCompileOK(t *testing.T) {
	hw := Devices().Select(0)
	t.Log(goodCode)
	sw, err := Compile(hw, goodCode, "good", "")
	if err != nil {
		t.Error(err)
	} else {
		sw.Release()
	}
	hw.Release()
}

func TestCompileError(t *testing.T) {
	hw := Devices().Select(0)
	t.Log(badCode)
	_, err := Compile(hw, badCode, "bad", "")
	t.Log(err)
	if err == nil || !strings.Contains(fmt.Sprint(err), "BUILD_PROGRAM_FAILURE") {
		t.Error("expected compile error")
	}
	hw.Release()
}

func TestHelloWorld(t *testing.T) {
	hw := Devices().Select(0)
	//t.Log(helloCode)
	sw, err := Compile(hw, helloCode, "hello", "")
	if err != nil {
		t.Error(err)
	} else {
		// run it!
		buff := []byte("Hello, world!")
		buf2 := []byte(".............")
		t.Logf("before: %s %s", buff, buf2)
		size := len(buff)
		err := Run(hw, sw, size, 1, " %r %w ", size, buff, size, buf2)
		if err != nil {
			t.Error(err)
		} else {
			t.Logf("after : %s %s", buff, buf2)
			if string(buff) != string(buf2) {
				t.Errorf("got %s - expected %s", buf2, buff)
			}
		}
		sw.Release()
	}
	hw.Release()
}

func TestVectorMatrixMultiplyFloat(t *testing.T) {
	hw := Devices().Select(0)
	//t.Log(matrixCode)
	sw, err := Compile(hw, matrixCode, "matvec_mult", "")
	if err != nil {
		t.Error(err)
	} else {
		// run it!
		matrix := make([]float32, 16)
		vector := make([]float32, 4)
		result := make([]float32, 4)
		expect := make([]float32, 4)
		for i := range matrix {
			matrix[i] = (0.1 + float32(i)) * 2
		}
		for i := range vector {
			vector[i] = float32(i) * 3.0
			expect[0] += matrix[i] * vector[i]
			expect[1] += matrix[i+4] * vector[i]
			expect[2] += matrix[i+8] * vector[i]
			expect[3] += matrix[i+12] * vector[i]
		}
		t.Logf("input : %v x %v", matrix, vector)
		t.Logf("expect: %.4g", expect)
		err := Run(hw, sw, 4, 1, " %r %r %w ", 16*4, matrix, 4*4, vector, 4*4, result)
		if err != nil {
			t.Error(err)
		} else {
			var err error
			diff := make([]float64, 4)
			for i := range result {
				diff[i] = float64(result[i] - expect[i])
				if math.Abs(diff[i]) > 1e-4 {
					err = fmt.Errorf("got %v - expected %v", result, expect)
				}
			}
			t.Logf("output: %.4g  diff: %v", result, diff)
			if err != nil {
				t.Error(err)
			}
		}
		sw.Release()
	}
	hw.Release()
}

func TestVectorMatrixMultiplyDouble(t *testing.T) {
	hw := Devices().Select(0)
	//t.Log(matrixCodeDouble)
	sw, err := Compile(hw, matrixCodeDouble, "matvec_mult", "")
	if err != nil {
		t.Error(err)
	} else {
		// run it!
		matrix := make([]float64, 16)
		vector := make([]float64, 4)
		result := make([]float64, 4)
		expect := make([]float64, 4)
		for i := range matrix {
			matrix[i] = (0.1 + float64(i)) * 2
		}
		for i := range vector {
			vector[i] = float64(i) * 3.0
			expect[0] += matrix[i] * vector[i]
			expect[1] += matrix[i+4] * vector[i]
			expect[2] += matrix[i+8] * vector[i]
			expect[3] += matrix[i+12] * vector[i]
		}
		t.Logf("input : %v x %v", matrix, vector)
		t.Logf("expect: %.4g", expect)
		err := Run(hw, sw, 4, 1, " %r %r %w ", 16*8, matrix, 4*8, vector, 4*8, result)
		if err != nil {
			t.Error(err)
		} else {
			var err error
			diff := make([]float64, 4)
			for i := range result {
				diff[i] = result[i] - expect[i]
				if math.Abs(diff[i]) > 1e-8 {
					err = fmt.Errorf("got %v - expected %v", result, expect)
				}
			}
			t.Logf("output: %.4g  diff: %v", result, diff)
			if err != nil {
				t.Error(err)
			}
		}
		sw.Release()
	}
	hw.Release()
}

func TestScaleVector(t *testing.T) {
	hw := Devices().Select(0)
	//t.Log(scaleCode)
	sw, err := Compile(hw, scaleCode, "scalevec", "")
	if err != nil {
		t.Error(err)
	} else {
		// input data
		size := 134217728
		vec := make([]float32, size)
		for i := range vec {
			vec[i] = float32(i)
		}
		buf := NewBuffer(hw, cl.MEM_READ_WRITE, size*4, vec)
		defer buf.Release()
		buf.Write(hw)
		scale := float32(3)
		t.Logf("before: 2=>%g 100=>%g 999=>%g  scale=%g", vec[2], vec[100], vec[999], scale)

		// run it!
		for run := 0; run < 3; run++ {
			err := Run(hw, sw, size, 1, " %b %a %N ", buf, 4, &scale, 4)
			buf.Read(hw)
			t.Logf("run %d: 2=>%g 100=>%g 999=>%g", run, vec[2], vec[100], vec[999])
			if err != nil {
				t.Error(err)
				break
			}
		}
		if vec[2] != 54 || vec[100] != 2700 || vec[999] != 26973 {
			t.Error("wrong output!")
		}
		sw.Release()
	}
	hw.Release()
}
