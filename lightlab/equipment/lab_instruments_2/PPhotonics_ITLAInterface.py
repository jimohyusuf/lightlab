import re
import time
import numpy as np
from scipy.constants import c as C0

from lightlab.equipment.lab_instruments_2.PPhotonics_ITLAClient import ITLAClient

# Helpers
def wait(time_sec):
    time.sleep(time_sec)
    return

def freq_to_wavl(frequency, speed_of_light=C0):
    return speed_of_light / np.array(frequency)

def wavl_to_freq(wavl, speed_of_light=C0):
    return speed_of_light / np.array(wavl)

def safe_float_regex(value):
    try:
        return float(re.sub(r'[^\d\.]', '', value))
    except ValueError:
        return None


# Laser state representation
LASER_OFF = 0
LASER_ON  = 1

class LaserState:
    def __init__(self):
        self.on = LASER_OFF
        self.power = None       # last requested power in dBm
        self.wavelength = None  # last requested wavelength (nm) or None

class Laser:
    """
    High-level Laser wrapper, no software triggers for clean sweep or jump.
    """
    def __init__(self, serial: str, address: str):
        self.verbose = False
        self.laser_state = LaserState()
        self.client = ITLAClient(serial_number=serial, address=address)

        # Initialize from device
        freq_str = self.client.get_frequency_tera_hz()
        freq_thz = safe_float_regex(freq_str) or 193.1
        freq_ghz = freq_thz * 1000
        self.laser_state.wavelength = freq_to_wavl(freq_ghz)

        power_str = self.client.read_power_dbm()
        p_val = safe_float_regex(power_str) or 10.0
        self.laser_state.power = p_val

    # -------------------------------------------------------------------------
    # Laser On/Off
    # -------------------------------------------------------------------------
    def ensure_on(self, wait_for_stable_power=True):
        if self.laser_state.on == LASER_OFF:
            self.client.enable_laser()
            self.laser_state.on = LASER_ON
        if wait_for_stable_power:
            self.wait_for_power_up()

    def ensure_off(self):
        self.client.disable_laser()
        self.laser_state.on = LASER_OFF

    # -------------------------------------------------------------------------
    # Power
    # -------------------------------------------------------------------------
    def set_power(self, power_dbm: float):
        """
        Turn laser off, set power, turn on again, wait for stable power if needed.
        """
        self.ensure_off()
        self.client.set_power_dbm(power_dbm)
        self.laser_state.power = power_dbm
        self.ensure_on()

    def get_power(self) -> float:
        """
        Returns the actual measured power from the device.
        """
        p_str = self.client.read_power_dbm()
        p_val = safe_float_regex(p_str)
        return p_val if p_val is not None else float('nan')

    def wait_for_power_up(self, tolerance_ratio=0.004, consecutive=32, timeout=30):
        """
        Checks laser power stability with safeguards for edge cases and timeouts.
        """
        if self.laser_state.power is None:
            return

        req = self.laser_state.power
        count = 0
        start_time = time.time()
        max_retries = 100
        retries = 0

        while (time.time() - start_time) < timeout and retries < max_retries:
            p_str = self.client.read_power_dbm()
            p_val = safe_float_regex(p_str)

            if p_val is None:
                retries += 1
                time.sleep(0.05)
                continue

            # Calculate error (absolute for near-zero targets)
            err = abs(p_val - req) / abs(req)
            tolerance = tolerance_ratio

            # Update progress
            print(
                f"Laser power: {p_val:.3f} dBm \t Error: {err:5.3f} \t Count: {count}",
                end="\r",
                flush=True,
            )

            if err < tolerance:
                count += 1
                if count >= consecutive:
                    print(f"\nLaser power stable at {p_val:.3f} dBm")
                    return
            else:
                count = 0

            time.sleep(0.05)

        raise TimeoutError("Laser power stabilization failed")


    # def wait_for_power_up(self, tolerance_ratio=0.004, consecutive=32):
    #     """
    #     Simple loop checking measured power vs. self.laser_state.power 
    #     for 'consecutive' times within 'tolerance_ratio' (0.3% by default).
    #     """
    #     if self.laser_state.power is None:
    #         return
    #     req = self.laser_state.power
    #     count = 0
    #     while True:
    #         p_str = self.client.read_power_dbm()
    #         p_val = safe_float_regex(p_str)
    #         if p_val is not None:
    #             err = abs(p_val - req) / max(abs(req), 1e-9)
    #             print(f"\rLaser power: {p_val:.3f} dBm \t Error: {err:5.3f} \t Count: {count}", end="")
    #             if err < tolerance_ratio:
    #                 count += 1
    #                 print(f"\n count {count}")
    #                 if count >= consecutive:
    #                     print(f"\nLaser power stable at {p_val:.3f} dBm")
    #                     break
    #             else:
    #                 count = 0
    #         # wait(0.05)

    # -------------------------------------------------------------------------
    # Frequency
    # -------------------------------------------------------------------------
    def set_wavelength(self, wavelength_nm: float):
        """
        Turn laser off, set freq, turn on again.
        """
        freq_ghz = wavl_to_freq(wavelength_nm)
        freq_thz = freq_ghz / 1000.0
        self.ensure_off()
        self.client.set_frequency_tera_hz(freq_thz)
        self.laser_state.wavelength = wavelength_nm
        self.ensure_on()

    def set_frequency_ghz(self, freq_ghz: float):
        """
        Utility method: sets frequency in GHz. The server expects THz.
        """
        self.set_wavelength(freq_to_wavl(freq_ghz))

    def get_frequency_ghz(self) -> float:
        """
        Return frequency in GHz by reading THz from server and multiplying by 1000.
        """
        freq_str = self.client.get_frequency_tera_hz()
        freq_thz = safe_float_regex(freq_str) or 193.0
        return freq_thz * 1000

    # -------------------------------------------------------------------------
    # Clean Sweep (no triggers)
    # -------------------------------------------------------------------------
    def set_clean_sweep(self, range_ghz: float, rate_ghz_s: float):
        """
        Configure range & rate. Doesn't enable yet.
        For extended range >150GHz, load calibrations if needed.
        """
        if range_ghz > 150:
            # Example calibration approach if needed
            self.client.load_clean_sweep_calibration(1234)

        self.client.set_clean_sweep_range(range_ghz)
        self.client.set_clean_sweep_rate(rate_ghz_s)

    def start_clean_sweep(self):
        """
        Just enable the sweep. Laser will ramp continuously until stopped.
        """
        self.client.enable_clean_sweep()

    def stop_clean_sweep(self):
        """
        Disables the sweep.
        """
        self.client.disable_clean_sweep()

    def read_sweep_offset(self) -> float:
        """
        Returns the current offset in GHz from -range/2 to +range/2.
        """
        off_str = self.client.read_clean_sweep_offset()
        off_val = safe_float_regex(off_str)
        return off_val if off_val is not None else float('nan')

    # -------------------------------------------------------------------------
    # Clean Jump
    # -------------------------------------------------------------------------
    def clean_jump(self, target_freq_thz: float, target_temp_c: float):
        """
        1) set frequency registers
        2) set temperature
        3) enable jump
        """
        self.client.set_clean_jump_frequency(target_freq_thz)
        self.client.set_clean_jump_temperature(target_temp_c)
        self.client.enable_clean_jump()

    def read_clean_jump_offset(self) -> float:
        """
        offset (MHz) = (raw - 10000).
        """
        val_str = self.client.read_clean_jump_offset()
        return safe_float_regex(val_str) or float('nan')

    # -------------------------------------------------------------------------
    # Close / Cleanup
    # -------------------------------------------------------------------------
    def close_port(self):
        self.client.port_close()
