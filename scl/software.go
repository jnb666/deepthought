package scl

import (
	"errors"
	"fmt"
	"github.com/go-gl/cl/v1.2/cl"
	"io/ioutil"
	"os"
	"unsafe"
)

// Software is a struct encapulating program and kernel data.
type Software struct {
	Program    cl.Program
	Kernel     cl.Kernel
	KernelName string
}

// Compile function compiles a kernel from source string and creates a new software object.
func Compile(h *Hardware, source, name string, opts string) (*Software, error) {
	s := new(Software)
	var err cl.ErrorCode
	srcptr := cl.Str(source + "\x00")
	s.Program = cl.CreateProgramWithSource(h.Context, 1, &srcptr, nil, &err)
	checkErr(err)
	opts += " -cl-fp32-correctly-rounded-divide-sqrt -cl-opt-disable"
	err = cl.BuildProgram(s.Program, 1, &h.DeviceId, cl.Str(opts+"\x00"), nil, nil)
	if err != cl.SUCCESS {
		var size uint64
		cl.GetProgramBuildInfo(s.Program, h.DeviceId, cl.PROGRAM_BUILD_LOG, 0, nil, &size)
		log := make([]byte, size+1)
		cl.GetProgramBuildInfo(s.Program, h.DeviceId, cl.PROGRAM_BUILD_LOG, size, unsafe.Pointer(&log[0]), nil)
		cl.ReleaseProgram(s.Program)
		return nil, fmt.Errorf("%s\n%s", cl.ErrToStr(err), log)
	}
	s.Kernel = cl.CreateKernel(s.Program, cl.Str(name+"\x00"), &err)
	if err != cl.SUCCESS {
		cl.ReleaseProgram(s.Program)
		return nil, errors.New(cl.ErrToStr(err))
	}
	s.KernelName = name
	return s, nil
}

// CompileFile function compiles a kernel from source file and creates a new software object.
func CompileFile(h *Hardware, source, name, opts string) (*Software, error) {
	f, err := os.Open(source)
	if err != nil {
		return nil, err
	}
	buf, err := ioutil.ReadAll(f)
	if err != nil {
		return nil, err
	}
	return Compile(h, string(buf), name, opts)
}

// Release method frees any resources tied to the software object.
func (s *Software) Release() {
	cl.ReleaseKernel(s.Kernel)
	cl.ReleaseProgram(s.Program)
}

// SetArg method sets an argument to the kernel
func (s *Software) SetArg(argc uint32, size uint64, ptr unsafe.Pointer) {
	err := cl.SetKernelArg(s.Kernel, argc, size, ptr)
	if err != cl.SUCCESS {
		panic(cl.ErrToStr(err))
	}
}

func (s *Software) SetArgBuffer(argc uint32, b *Buffer) {
	err := cl.SetKernelArg(s.Kernel, argc, 8, unsafe.Pointer(&b.Buf))
	if err != cl.SUCCESS {
		panic(cl.ErrToStr(err))
	}
}

// The Run function launches a kernel with the given argument list.
// The format string has a printf style list describing the type of each argument:
//
//		%r	read only buffer: size in bytes + slice of values
//		%w	write only buffer: size in bytes + slice of values
//		%R	read and write buffer: size in bytes + slice of values
//		%a	pass by value argument: size in bytes + pointer to value
//		%b	buffer object: pointer to a scl.Buffer only
//		%N	null argument for local memory: size in bytes only
//
func Run(h *Hardware, s *Software, workSizeX, workSizeY int, format string, args ...interface{}) error {
	var err cl.ErrorCode
	var outBuf []*Buffer
	// parse arguments
	p := 0
	argc := uint32(0)
	for i := 0; i < len(format); i++ {
		if format[i] == '%' {
			i++
			switch format[i] {
			case 'r':
				size := args[p].(int)
				//fmt.Println("arg[r]", argc, size, args[p+1])
				b := NewBuffer(h, cl.MEM_READ_ONLY, size, args[p+1])
				defer b.Release()
				b.Write(h)
				s.SetArgBuffer(argc, b)
				argc++
				p += 2
			case 'w':
				size := args[p].(int)
				//fmt.Println("arg[w]", argc, size, args[p+1])
				b := NewBuffer(h, cl.MEM_WRITE_ONLY, size, args[p+1])
				defer b.Release()
				outBuf = append(outBuf, b)
				s.SetArgBuffer(argc, b)
				argc++
				p += 2
			case 'R':
				size := args[p].(int)
				//fmt.Println("arg[R]", argc, size, args[p+1])
				b := NewBuffer(h, cl.MEM_READ_WRITE, size, args[p+1])
				defer b.Release()
				b.Write(h)
				outBuf = append(outBuf, b)
				s.SetArgBuffer(argc, b)
				argc++
				p += 2
			case 'a':
				size := uint64(args[p].(int))
				//fmt.Println("arg[a]", argc, size, args[p+1])
				err = cl.SetKernelArg(s.Kernel, argc, size, getPointer(args[p+1]))
				argc++
				p += 2
			case 'b':
				b := args[p].(*Buffer)
				//fmt.Println("arg[b]", argc, b)
				s.SetArgBuffer(argc, b)
				argc++
				p++
			case 'N':
				size := uint64(args[p].(int))
				//fmt.Println("arg[N]", argc, size)
				err = cl.SetKernelArg(s.Kernel, argc, size, nil)
				argc++
				p++
			default:
				return fmt.Errorf("unexpected format flag: %s in %s", format[i-1:i+1], format)
			}
			if err != cl.SUCCESS {
				return errors.New(cl.ErrToStr(err))
			}
		}
	}

	// enqueue the kernel
	//fmt.Println("enqueue kernel")
	s.EnqueueKernel(h, []uint64{uint64(workSizeX), uint64(workSizeY)}, nil)

	// read the output
	//fmt.Println("read output buffers")
	for _, b := range outBuf {
		b.Read(h)
	}
	return nil
}

// Enqueue kernel method runs an NDRange kernel
func (s *Software) EnqueueKernel(h *Hardware, globalSize, localSize []uint64) {
	var err cl.ErrorCode
	if localSize != nil {
		err = cl.EnqueueNDRangeKernel(h.Queue, s.Kernel, uint32(len(globalSize)), nil, &globalSize[0], &localSize[0], 0, nil, nil)
	} else {
		err = cl.EnqueueNDRangeKernel(h.Queue, s.Kernel, uint32(len(globalSize)), nil, &globalSize[0], nil, 0, nil, nil)
	}
	if err != cl.SUCCESS {
		panic(cl.ErrToStr(err))
	}
}
