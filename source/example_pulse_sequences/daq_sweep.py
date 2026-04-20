
import logging
import math
import numpy as np
from qupyt.pulse_sequences.yaml_sequence import YamlSequence


def generate_sequence(params: dict):
    return DAQ_read(
    int(params.get('number_measurements')),
    float(params.get('AWG_frequency')),
    float(params.get('max_delay')),
    float(params.get('pi')),
    float(params.get('laser_duration')),
    float(params.get('AOM_frequency'))  # Hz
)

def DAQ_read(
    number_measurements: int,
    AWG_frequency: float,
    max_delay: float,
    pi: float,
    laser_duration: float,
    AOM_frequency: float = 250e6

) -> dict:
    number_delays = number_measurements//2
    laser_start = 1
    #mw_start = 35.5 #21.5
    read_duration = 3
    delay_list = np.linspace(0, max_delay, number_delays).tolist()
    pi_pulse_start =0.5
    time_quarter = laser_start + laser_duration + 1
    total_time = 4 * time_quarter

    daqread = YamlSequence(duration = total_time)

    # print("number of delays:", number_delays)
    
    daqread.add_pulse(
        'START',
        0.1,
        0.2,
        sequence_blocks = ['start_block']
        )


    for t, delay in enumerate(delay_list):
        daqread.add_pulse(
            'READ', 
            time_quarter+delay + laser_start,
	        read_duration,
	        sequence_blocks = [f'block_{t}']
            )
    for t, delay in enumerate(delay_list):
      daqread.add_pulse(
          'READ', 
          3*time_quarter+delay + laser_start,
          read_duration,
          sequence_blocks = [f'block_{t}']
          )
    
    for t in range(len(delay_list)):
        daqread.add_pulse(
            'MW', 
            3*time_quarter + pi_pulse_start,
	          pi,
            frequency=AWG_frequency,
	          sequence_blocks = [f'block_{t}']
            )

    for i in range(4): 
        daqread.add_pulse(
            'LASER',  
            i * time_quarter
            + laser_start,      
            laser_duration,
            frequency = AOM_frequency,
            amplitude = 0.2,
            sequence_blocks = [f'block_{t}' for t in range(len(delay_list))]
            )

    for i in range(4):
        daqread.add_pulse(
            'LASER',
            i * time_quarter
            + laser_start,
            laser_duration,
            frequency = AOM_frequency,
            amplitude = 0.2,
            sequence_blocks = ["start_block", "wait_loop"]
            )
    
    
    daqread.sequencing_order = ["wait_loop", "start_block", "wait_loop"] + [f'block_{p}' for p in range(len(delay_list))]  # add the first block at the end to ensure it is executed first
    daqread.sequencing_repeats = [1,2,1] + ([1] *  (len(delay_list))) # execute each block once

    daqread.write()

    return {}


 