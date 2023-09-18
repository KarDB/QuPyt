# Copyright (c) 2023 SpinCore Technologies, Inc.
# http://www.spincore.com
#
# This software is provided 'as-is', without any express or implied warranty.
# In no event will the authors be held liable for any damages arising from the
# use of this software.
#
# Permission is granted to anyone to use this software for any purpose,
# including commercial applications, and to alter it and redistribute it
# freely, subject to the following restrictions:
#
# 1. The origin of this software must not be misrepresented; you must not
# claim that you wrote the original software. If you use this software in a
# product, an acknowledgement in the product documentation would be appreciated
# but is not required.
# 2. Altered source versions must be plainly marked as such, and must not be
# misrepresented as being the original software.
# 3. This notice may not be removed or altered from any source distribution.

# Version 20230707

# The following version of the file is NOT THE ORIGINAL version.
# It has been altered subject to the original license above.
# The changes made consiste mainly in formatting and type hints.

import ctypes
from typing import Any

PULSE_PROGRAM = 0
FREQ_REGS = 1

try:
    spinapi = ctypes.CDLL("spinapi64")
except:
    try:
        spinapi = ctypes.CDLL("spinapi")
    except:
        print("Failed to load spinapi library.")


def enum(**enums: Any) -> type:
    return type('Enum', (), enums)


ns = 1.0
us = 1000.0
ms = 1000000.0

MHz = 1.0
kHz = 0.001
Hz = 0.000001

# Defines for status bits
STATUS_STOPPED = 1
STATUS_RESET = 2
STATUS_RUNNING = 4
STATUS_WAITING = 8

# Instruction enum
Inst = enum(
    CONTINUE=0,
    STOP=1,
    LOOP=2,
    END_LOOP=3,
    JSR=4,
    RTS=5,
    BRANCH=6,
    LONG_DELAY=7,
    WAIT=8,
    RTI=9
)

CONTINUE = 0
STOP = 1
LOOP = 2
END_LOOP = 3
JSR = 4
RTS = 5
BRANCH = 6
LONG_DELAY = 7
WAIT = 8
RTI = 9

ONE_PERIOD = 0x200000
TWO_PERIOD = 0x400000
THREE_PERIOD = 0x600000
FOUR_PERIOD = 0x800000
FIVE_PERIOD = 0xA00000
ON = 0xE00000

REG_SHORTPULSE_DISABLE = 0x06
REG_START_ADDRESS = 0x07
REG_DEFAULT_FLAGS = 0x08

# Defines for start_programming

FREQ_REGS = 1

PHASE_REGS = 2
TX_PHASE_REGS = 2
PHASE_REGS_1 = 2

RX_PHASE_REGS = 3
PHASE_REGS_0 = 3

# These are names used by RadioProcessor
COS_PHASE_REGS = 51
SIN_PHASE_REGS = 50

# For specifying which device in pb_dds_load
DEVICE_SHAPE = 0x099000
DEVICE_DDS = 0x099001

# Defines for enabling analog output
ANALOG_ON = 1
ANALOG_OFF = 0
TX_ANALOG_ON = 1
TX_ANALOG_OFF = 0
RX_ANALOG_ON = 1
RX_ANALOG_OFF = 0


# RadioProcessor control word defines
TRIGGER = 0x0001
PCI_READ = 0x0002
BYPASS_AVERAGE = 0x0004
NARROW_BW = 0x0008
FORCE_AVG = 0x0010
BNC0_CLK = 0x0020
DO_ZERO = 0x0040
BYPASS_CIC = 0x0080
BYPASS_FIR = 0x0100
BYPASS_MULT = 0x0200
SELECT_AUX_DDS = 0x0400
DDS_DIRECT = 0x0800
SELECT_INTERNAL_DDS = 0x1000
DAC_FEEDTHROUGH = 0x2000
OVERFLOW_RESET = 0x4000
RAM_DIRECT = 0x8000 | BYPASS_CIC | BYPASS_FIR | BYPASS_MULT


spinapi.pb_get_version.restype = ctypes.c_char_p
spinapi.pb_get_error.restype = ctypes.c_char_p

spinapi.pb_count_boards.restype = ctypes.c_int

spinapi.pb_init.restype = ctypes.c_int

spinapi.pb_select_board.argtype = ctypes.c_int
spinapi.pb_select_board.restype = ctypes.c_int

spinapi.pb_set_debug.argtype = ctypes.c_int
spinapi.pb_set_debug.restype = ctypes.c_int

spinapi.pb_set_defaults.restype = ctypes.c_int

spinapi.pb_set_freq.argtype = ctypes.c_double
spinapi.pb_set_freq.restype = ctypes.c_int

spinapi.pb_set_phase.argtype = ctypes.c_double
spinapi.pb_set_phase.restype = ctypes.c_int

spinapi.pb_set_amp.argtype = ctypes.c_float, ctypes.c_int
spinapi.pb_set_amp.restype = ctypes.c_int

spinapi.pb_overflow.argtype = ctypes.c_int, ctypes.c_int
spinapi.pb_overflow.restype = ctypes.c_int

spinapi.pb_scan_count.argtype = ctypes.c_int
spinapi.pb_scan_count.restype = ctypes.c_int

spinapi.pb_set_num_points.argtype = ctypes.c_int
spinapi.pb_set_num_points.restype = ctypes.c_int

spinapi.pb_set_radio_control.argtype = ctypes.c_int
spinapi.pb_set_radio_control.restype = ctypes.c_int

spinapi.pb_core_clock.argtype = ctypes.c_double
spinapi.pb_core_clock.restype = ctypes.c_int

spinapi.pb_write_register.argtype = ctypes.c_int, ctypes.c_int
spinapi.pb_write_register.restype = ctypes.c_int

spinapi.pb_start_programming.argtype = ctypes.c_int
spinapi.pb_start_programming.restype = ctypes.c_int

spinapi.pb_stop_programming.restype = ctypes.c_int

spinapi.pb_start.restype = ctypes.c_int

spinapi.pb_stop.restype = ctypes.c_int

spinapi.pb_reset.restype = ctypes.c_int

spinapi.pb_close.restype = ctypes.c_int

spinapi.pb_read_status.restype = ctypes.c_int

spinapi.pb_status_message.restype = ctypes.c_char_p

spinapi.pb_get_firmware_id.restype = ctypes.c_int

spinapi.pb_sleep_ms.argtype = ctypes.c_int
spinapi.pb_sleep_ms.restype = ctypes.c_int

spinapi.pb_get_data.argtype = (
    ctypes.c_int,  # num_points Number of complex points to read from RAM
    # real_data Real data from RAM is stored into this array
    ctypes.POINTER(ctypes.c_int),
    # imag_data Imag data from RAM is stored into this array
    ctypes.POINTER(ctypes.c_int),
)
spinapi.pb_get_data.restype = ctypes.c_int

spinapi.pb_get_data_direct.argtype = (
    ctypes.c_int, ctypes.POINTER(ctypes.c_short))
spinapi.pb_get_data_direct.restype = ctypes.c_int

spinapi.pb_unset_radio_control.argtype = ctypes.c_int
spinapi.pb_unset_radio_control.restype = ctypes.c_int

spinapi.pb_inst_pbonly.argtype = (
    ctypes.c_int,  # flags
    ctypes.c_int,  # inst
    ctypes.c_int,  # inst data
    ctypes.c_double,  # length (double)
)
spinapi.pb_inst_pbonly.restype = ctypes.c_int

spinapi.pb_dds_load.argtype = ctypes.c_float, ctypes.c_int
spinapi.pb_dds_load.restype = ctypes.c_int

spinapi.pb_inst_radio.argtype = (
    ctypes.c_int,  # Frequency register
    ctypes.c_int,  # Cosine phase
    ctypes.c_int,  # Sin phase
    ctypes.c_int,  # tx phase
    ctypes.c_int,  # tx enable
    ctypes.c_int,  # phase reset
    ctypes.c_int,  # trigger scan
    ctypes.c_int,  # flags
    ctypes.c_int,  # inst
    ctypes.c_int,  # inst data
    ctypes.c_double,  # length (double)
)
spinapi.pb_inst_radio.restype = ctypes.c_int

spinapi.pb_inst_radio_shape.argtype = (
    ctypes.c_int,  # Frequency register
    ctypes.c_int,  # cos phase
    ctypes.c_int,  # sin phase
    ctypes.c_int,  # tx phase
    ctypes.c_int,  # tx enable
    ctypes.c_int,  # phase reset
    ctypes.c_int,  # trigger scan
    ctypes.c_int,  # useshape
    ctypes.c_int,  # amp
    ctypes.c_int,  # flags
    ctypes.c_int,  # inst
    ctypes.c_int,  # inst data
    ctypes.c_double,  # length (double)
)
spinapi.pb_inst_radio_shape.restype = ctypes.c_int

spinapi.pb_inst_dds2.argtype = (
    ctypes.c_int,  # Frequency register DDS0
    ctypes.c_int,  # Phase register DDS0
    ctypes.c_int,  # Amplitude register DDS0
    ctypes.c_int,  # Output enable DDS0
    ctypes.c_int,  # Phase reset DDS0
    ctypes.c_int,  # Frequency register DDS1
    ctypes.c_int,  # Phase register DDS1
    ctypes.c_int,  # Amplitude register DDS1
    ctypes.c_int,  # Output enable DDS1,
    ctypes.c_int,  # Phase reset DDS1,
    ctypes.c_int,  # Flags
    ctypes.c_int,  # inst
    ctypes.c_int,  # inst data
    ctypes.c_double,  # timing value (double)
)
spinapi.pb_inst_dds2.restype = ctypes.c_int

spinapi.pb_write_felix.argtype = (
    ctypes.c_char_p,  # fnameout The filename for the Felix file you want to create
    ctypes.c_char_p,  # title_string 	Large string with all parameter information to include in Felix Title Block
    ctypes.c_int,  # num_points Number of points to write to the file
    ctypes.c_float,  # SW Spectral width of the baseband data in Hz
    ctypes.c_float,  # SF Spectrometer frequency in MHz
    ctypes.c_int,  # real_data Integer array containing the real portion of the data points
    ctypes.c_int,  # imag_data Integer array containing the imaginary portion of the data points
)
spinapi.pb_write_felix.restype = ctypes.c_int

spinapi.pb_setup_filters.argtype = (
    ctypes.c_double,  # spectral_width
    ctypes.c_int,  # scan_repetitions
    ctypes.c_int,  # cmd
)
spinapi.pb_setup_filters.restype = ctypes.c_int

spinapi.pb_inst_radio_shape_cyclops.argtype = (
    ctypes.c_int,  # Frequency register
    ctypes.c_int,  # cos phase
    ctypes.c_int,  # sin phase
    ctypes.c_int,  # tx phase
    ctypes.c_int,  # tx enable
    ctypes.c_int,  # phase reset
    ctypes.c_int,  # trigger scan
    ctypes.c_int,  # useshape
    ctypes.c_int,  # amp
    ctypes.c_int,  # real_add_sub
    ctypes.c_int,  # imag_add_sub
    ctypes.c_int,  # channel_swap
    ctypes.c_int,  # flags
    ctypes.c_int,  # inst
    ctypes.c_int,  # inst data
    ctypes.c_double,  # length (double)
)
spinapi.pb_inst_radio_shape_cyclops.restype = ctypes.c_int

spinapi.pb_fft_find_resonance.argtypes = (
    ctypes.c_int,                    # num_points: Number of complex data points
    # SF: Spectrometer Frequency used for the experiment (in Hz)
    ctypes.c_double,
    # SW: Spectral Width used for data acquisition (in Hz)
    ctypes.c_double,
    # real: Array of the real part of the complex data points
    ctypes.POINTER(ctypes.c_int),
    # imag: Array of the imaginary part of the complex data points
    ctypes.POINTER(ctypes.c_int)
)
spinapi.pb_fft_find_resonance.restype = ctypes.c_double

spinapi.pb_write_ascii.argtype = (
    ctypes.c_char_p,  # fname
    ctypes.c_int,     # num_points
    ctypes.c_float,   # SW
    ctypes.POINTER(ctypes.c_int),  # real_data
    ctypes.POINTER(ctypes.c_int)   # imag_data
)
spinapi.pb_write_ascii.restype = ctypes.c_int

spinapi.pb_write_ascii_verbose.argtype = (
    ctypes.c_char_p,  # fname
    ctypes.c_int,     # num_points
    ctypes.c_float,   # SW
    ctypes.c_float,    # SF
    ctypes.POINTER(ctypes.c_int),  # real_data
    ctypes.POINTER(ctypes.c_int)   # imag_data
)
spinapi.pb_write_ascii_verbose.restype = ctypes.c_int

spinapi.pb_write_jcamp.argtypes = (
    ctypes.c_char_p,  # fname
    ctypes.c_int,     # num_points
    ctypes.c_float,   # SW
    ctypes.c_float,   # SF
    ctypes.POINTER(ctypes.c_int),  # real_data
    ctypes.POINTER(ctypes.c_int)   # imag_data
)
spinapi.pb_write_jcamp.restype = ctypes.c_int

spinapi.pb_set_scan_segments.argtype = ctypes.c_int
spinapi.pb_set_scan_segments.restype = ctypes.c_int


def pb_get_version() -> str:
    """Return library version as UTF-8 encoded string."""
    ret = spinapi.pb_get_version()
    return str(ctypes.c_char_p(ret).value.decode("utf-8"))


def pb_get_error() -> str:
    """Return library error as UTF-8 encoded string."""
    ret = spinapi.pb_get_error()
    return str(ctypes.c_char_p(ret).value.decode("utf-8"))


def pb_count_boards():
    """Return the number of boards detected in the system."""
    return spinapi.pb_count_boards()


def pb_init():
    """Initialize currently selected board."""
    return spinapi.pb_init()


def pb_set_debug(debug):
    return spinapi.pb_set_debug(debug)


def pb_select_board(board_number):
    """Select a specific board number"""
    return spinapi.pb_select_board(board_number)


def pb_set_defaults():
    """Set board defaults. Must be called before using any other board functions."""
    return spinapi.pb_set_defaults()


def pb_set_freq(*args):
    t = list(args)
    # Argument 0 must be a double
    t[0] = ctypes.c_double(t[0])
    args = tuple(t)
    return spinapi.pb_set_freq(*args)


def pb_set_phase(*args):
    t = list(args)
    # Argument 0 must be a double
    t[0] = ctypes.c_double(t[0])
    args = tuple(t)
    return spinapi.pb_set_phase(*args)


def pb_set_amp(*args):
    t = list(args)
    # Argument 0 must be a float
    t[0] = ctypes.c_float(t[0])
    args = tuple(t)
    return spinapi.pb_set_amp(*args)


def pb_overflow(*args):
    return spinapi.pb_overflow(*args)


def pb_scan_count(*args):
    return spinapi.pb_scan_count(*args)


def pb_set_num_points(*args):
    return spinapi.pb_set_num_points(*args)


def pb_set_radio_control(*args):
    return spinapi.pb_set_radio_control(*args)


def pb_core_clock(clock):
    return spinapi.pb_core_clock(ctypes.c_double(clock))


def pb_write_register(address, value):
    return spinapi.pb_write_register(address, value)


def pb_start_programming(target):
    return spinapi.pb_start_programming(target)


def pb_stop_programming():
    return spinapi.pb_stop_programming()


def pb_dds_load(*args):
    t = list(args)
    # List of argument 0 must be floats
    t[0] = (ctypes.c_float * len(t[0]))(*t[0])
    args = tuple(t)
    return spinapi.pb_dds_load(*args)


def pb_inst_pbonly(*args):
    t = list(args)
    # Argument 3 must be a double
    t[3] = ctypes.c_double(t[3])
    args = tuple(t)
    return spinapi.pb_inst_pbonly(*args)


def pb_inst_radio(*args):
    t = list(args)
    # Argument 10 must be a double
    t[10] = ctypes.c_double(t[10])
    args = tuple(t)
    return spinapi.pb_inst_radio(*args)


def pb_inst_dds(FREQ, TX_PHASE, TX_ENABLE, PHASE_RESET, FLAGS, INST, INST_DATA, LENGTH):
    return pb_inst_radio(FREQ, 0, 0, TX_PHASE, TX_ENABLE, PHASE_RESET, 0, FLAGS, INST, INST_DATA, LENGTH)


def pb_inst_radio_shape(*args):
    t = list(args)
    # Argument 12 must be a double
    t[12] = ctypes.c_double(t[12])
    args = tuple(t)
    return spinapi.pb_inst_radio_shape(*args)


def pb_inst_dds_shape(FREQ, TX_PHASE, TX_ENABLE, PHASE_RESET, USESHAPE, AMP, FLAGS, INST, INST_DATA, LENGTH):
    return pb_inst_radio_shape(FREQ, 0, 0, TX_PHASE, TX_ENABLE, PHASE_RESET, 0, USESHAPE, AMP, FLAGS, INST, INST_DATA, LENGTH)


def pb_inst_dds2(*args):
    t = list(args)
    # Argument 13 must be a double
    t[13] = ctypes.c_double(t[13])
    args = tuple(t)
    return spinapi.pb_inst_dds2(*args)


def pb_start():
    return spinapi.pb_start()


def pb_stop():
    return spinapi.pb_stop()


def pb_reset():
    return spinapi.pb_reset()


def pb_close():
    return spinapi.pb_close()


def pb_read_status():
    return spinapi.pb_read_status()


def pb_status_message():
    """Return library version as UTF-8 encoded string."""
    ret = spinapi.pb_status_message()
    return str(ctypes.c_char_p(ret).value.decode("utf-8"))


def pb_get_firmware_id():
    return spinapi.pb_get_firmware_id()


def pb_sleep_ms(mlsc):
    return spinapi.pb_sleep_ms(mlsc)


def pb_get_data(num_points, real_data, imag_data):
    # Create arrays to store the data
    c_real_data = (ctypes.c_int * num_points)(*real_data)
    c_imag_data = (ctypes.c_int * num_points)(*imag_data)

    # Assign the memory address of the c_real_data and c_imag_data arrays to the pointer arguments
    real_data_pointer = ctypes.cast(c_real_data, ctypes.POINTER(ctypes.c_int))
    imag_data_pointer = ctypes.cast(c_imag_data, ctypes.POINTER(ctypes.c_int))

    real_data.contents = real_data_pointer
    imag_data.contents = imag_data_pointer

    # Call the C function with the updated data pointers
    result = spinapi.pb_get_data(num_points, real_data, imag_data)

    return result


def pb_get_data_direct(num_points, data):
    # Create a C-style array of shorts using ctypes
    c_data = (ctypes.c_short * num_points)(*data)

    # Assign the memory address of the c_data array to the pointer argument
    data_pointer = ctypes.cast(c_data, ctypes.POINTER(ctypes.c_short))
    data.contents = data_pointer

    # Call the C function with the updated data pointer
    result = spinapi.pb_get_data_direct(num_points, data)

    return result


def pb_unset_radio_control(ctrl):
    return spinapi.pb_unset_radio_control(ctrl)


def pb_write_felix(*args):
    t = list(args)
    t[0] = ctypes.c_char_p(t[0].encode())
    t[1] = ctypes.c_char_p(t[1].encode())
    # Argument 3 must be a float
    t[3] = ctypes.c_float(t[3])
    # Argument 4 must be a float
    t[4] = ctypes.c_float(t[4])
    # List of argument 5 must be integers
    t[5] = (ctypes.c_int * len(t[5]))(*t[5])
    # List of argument 6 must be integers
    t[6] = (ctypes.c_int * len(t[6]))(*t[6])
    args = tuple(t)
    return spinapi.pb_write_felix(*args)


def pb_setup_filters(*args):
    t = list(args)
    # Argument 0 must be a double
    t[0] = ctypes.c_double(t[0])
    args = tuple(t)
    return spinapi.pb_setup_filters(*args)


def pb_inst_radio_shape_cyclops(*args):
    t = list(args)
    # Argument 15 must be a double
    t[15] = ctypes.c_double(t[15])
    args = tuple(t)
    return spinapi.pb_inst_radio_shape_cyclops(*args)


def pb_fft_find_resonance(num_points, SF, SW, real_data, imag_data):
    # Create a C-style array of ints using ctypes
    c_real_data = (ctypes.c_int * num_points)(*real_data)
    c_imag_data = (ctypes.c_int * num_points)(*imag_data)

    # Assign the memory address of the c_data arrays to the pointer arguments
    real_data_pointer = ctypes.cast(c_real_data, ctypes.POINTER(ctypes.c_int))
    imag_data_pointer = ctypes.cast(c_imag_data, ctypes.POINTER(ctypes.c_int))

    # Call the C function with the updated data pointers
    result = spinapi.pb_fft_find_resonance(
        num_points, SF, SW, real_data_pointer, imag_data_pointer)

    return result


def pb_write_ascii(fname, num_points, SW, real_data, imag_data):
    # Convert the file name to a C-style string
    c_fname = ctypes.c_char_p(fname.encode())

    # Convert the Python lists to C-style arrays
    c_real_data = (ctypes.c_int * num_points)(*real_data)
    c_imag_data = (ctypes.c_int * num_points)(*imag_data)

    # Call the C function
    result = spinapi.pb_write_ascii(
        c_fname, num_points, SW, c_real_data, c_imag_data)

    return result


def pb_write_ascii_verbose(fname, num_points, SW, SF, real_data, imag_data):
    # Convert the file name to a C-style string
    c_fname = ctypes.c_char_p(fname.encode())

    # Convert the Python lists to C-style arrays
    c_real_data = (ctypes.c_int * num_points)(*real_data)
    c_imag_data = (ctypes.c_int * num_points)(*imag_data)

    # Call the C function
    SW = ctypes.c_float(SW)
    SF = ctypes.c_float(SF)
    result = spinapi.pb_write_ascii_verbose(
        c_fname, num_points, SW, SF, c_real_data, c_imag_data)

    return result


def pb_write_jcamp(fname, num_points, SW, SF, real_data, imag_data):
    # Convert the file name to a C-style string
    c_fname = ctypes.c_char_p(fname.encode())

    # Convert the Python lists to C-style arrays
    c_real_data = (ctypes.c_int * num_points)(*real_data)
    c_imag_data = (ctypes.c_int * num_points)(*imag_data)

    SF = ctypes.c_float(SF)
    SW = ctypes.c_float(SW)

    # Call the C function
    result = spinapi.pb_write_jcamp(
        c_fname, num_points, SW, SF, c_real_data, c_imag_data)

    return result


def pb_set_scan_segments(num_segments):
    # Call the C function
    return spinapi.pb_set_scan_segments(num_segments)
