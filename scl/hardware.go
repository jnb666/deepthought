/*
Package scl is a set of bindings to make it easy to execute OpenCL kernels.
It is inspired by the SimpleOpenCL C library by Oscar Amoros.

References:
	http://github.com/go-gl/cl
	http://code.google.com/p/simple-opencl/
	http://www.khronos.org/opencl/
*/
package scl

import (
	"fmt"
	"github.com/go-gl/cl/v1.2/cl"
	"unsafe"
)

// Device has information about the OpenCL platform and device.
type Device struct {
	Platform cl.PlatformID
	DeviceId cl.DeviceId
	Units    int
	Memory   uint64
	Type     cl.DeviceType
	Name     string
}

// DeviceList is a list of devices
type DeviceList []*Device

// Hardware is a struct with an OpenCL device and associated context and queue.
type Hardware struct {
	*Device
	Context cl.Context
	Queue   cl.CommandQueue
}

// Devices function returns a list with all available devices.
func Devices() DeviceList {
	list := []*Device{}
	// get platform id
	var nPlatform, nDevices uint32
	var platforms [8]cl.PlatformID
	err := cl.GetPlatformIDs(8, &platforms[0], &nPlatform)
	checkErr(err)
	if nPlatform <= 0 {
		panic("no OpenCL platform found")
	}
	var devices [16]cl.DeviceId
	for i := 0; i < int(nPlatform); i++ {
		// get device info
		err = cl.GetDeviceIDs(platforms[i], cl.DEVICE_TYPE_ALL, 16, &devices[0], &nDevices)
		checkErr(err)
		for j := 0; j < int(nDevices); j++ {
			list = append(list, &Device{
				Platform: platforms[i],
				DeviceId: devices[j],
				Units:    computeUnits(devices[j]),
				Memory:   memorySize(devices[j]),
				Type:     deviceType(devices[j]),
				Name:     deviceName(devices[j]),
			})
		}
	}
	return list
}

// get info about a device
func computeUnits(id cl.DeviceId) int {
	var units uint32
	err := cl.GetDeviceInfo(id, cl.DEVICE_MAX_COMPUTE_UNITS, 4, unsafe.Pointer(&units), nil)
	checkErr(err)
	return int(units)
}

func memorySize(id cl.DeviceId) uint64 {
	var size uint64
	err := cl.GetDeviceInfo(id, cl.DEVICE_MAX_MEM_ALLOC_SIZE, 8, unsafe.Pointer(&size), nil)
	checkErr(err)
	return size
}

func deviceType(id cl.DeviceId) cl.DeviceType {
	var size uint64
	err := cl.GetDeviceInfo(id, cl.DEVICE_TYPE, 0, nil, &size)
	checkErr(err)
	var typ cl.DeviceType
	err = cl.GetDeviceInfo(id, cl.DEVICE_TYPE, size, unsafe.Pointer(&typ), nil)
	checkErr(err)
	return typ
}

func deviceName(id cl.DeviceId) string {
	var size uint64
	err := cl.GetDeviceInfo(id, cl.DEVICE_NAME, 0, nil, &size)
	checkErr(err)
	buf := make([]byte, size)
	err = cl.GetDeviceInfo(id, cl.DEVICE_NAME, size, unsafe.Pointer(&buf[0]), nil)
	checkErr(err)
	return string(buf)
}

// Select method returns the "best" hardware of given type from list of devices.
func (devs DeviceList) Select(devType cl.DeviceType) (h *Hardware) {
	var err cl.ErrorCode
	maxu := 0
	for _, d := range devs {
		if (devType == 0 || (devType&d.Type > 0)) && d.Units > maxu {
			maxu = d.Units
			h = &Hardware{Device: d}
		}
	}
	if h == nil {
		panic("no OpenCL device selected")
	}
	h.Context = cl.CreateContext(nil, 1, &h.DeviceId, nil, nil, &err)
	checkErr(err)

	h.Queue = cl.CreateCommandQueue(h.Context, h.DeviceId, 0, &err)
	checkErr(err)
	return h
}

// Release method frees any resources tied to the hardware.
func (h *Hardware) Release() {
	cl.ReleaseCommandQueue(h.Queue)
	cl.ReleaseContext(h.Context)
}

// String formats a human readable hardware description of the device.
func (d *Device) String() string {
	typeName := map[int](string){2: "CPU", 4: "GPU"}
	return fmt.Sprintf("%s Units=%d Memory=%dM Type=%s",
		d.Name, d.Units, d.Memory/(1024*1024), typeName[int(d.Type)])
}

// check OpenCL return code
func checkErr(err cl.ErrorCode) {
	if err != cl.SUCCESS {
		panic(cl.ErrToStr(err))
	}
}
