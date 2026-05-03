import math
from typing import List, Dict
from .client_layer import TaskVehicle
from .edge_layer import UAV


class CommunicationModel:
    G0_DB = -60
    G0_LINEAR = 10 ** (G0_DB / 10)
    BANDWIDTH_HZ = 1 * 1e6
    POWER_UAV_WATT = 2.0
    NOISE_POWER_DBM = -120
    NOISE_POWER_WATT = 10 ** ((NOISE_POWER_DBM - 30) / 10)
    LOS_PROB_A = 10
    LOS_PROB_B = 0.6
    BETA_0 = 3e-7
    ETA_NLOS = 0.2
    POWER_VEHICLE_WATT = 1.0

    @staticmethod
    def calculate_distance(node1, node2):
        if node1.x is None or node2.x is None:
            return float('inf')
        dx = node1.x - node2.x
        dy = node1.y - node2.y
        dz = node1.z - node2.z
        return math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

    @classmethod
    def get_link_status_leader_to_member(cls, leader_uav, member_uav):
        distance = cls.calculate_distance(leader_uav, member_uav)
        eff_distance = max(distance, 1.0)
        gain_linear = cls.G0_LINEAR * (eff_distance ** -2)
        gain_db = 10 * math.log10(gain_linear)
        snr = (gain_linear * cls.POWER_UAV_WATT) / cls.NOISE_POWER_WATT
        transmission_rate_bps = cls.BANDWIDTH_HZ * math.log2(1 + snr)
        return {
            "distance": distance,
            "gain_linear": gain_linear,
            "gain_db": gain_db,
            "rate_mbps": transmission_rate_bps / 1e6,
        }

    @classmethod
    def get_link_status_leader_to_base_station(cls, leader_uav, base_station):
        distance = cls.calculate_distance(leader_uav, base_station)
        eff_distance = max(distance, 1.0)
        gain_linear = cls.G0_LINEAR * (eff_distance ** -2)
        gain_db = 10 * math.log10(gain_linear)
        snr = (gain_linear * cls.POWER_UAV_WATT) / cls.NOISE_POWER_WATT
        transmission_rate_bps = cls.BANDWIDTH_HZ * math.log2(1 + snr)
        return {
            "distance": distance,
            "gain_linear": gain_linear,
            "gain_db": gain_db,
            "rate_mbps": transmission_rate_bps / 1e6,
        }

    @classmethod
    def get_los_probability_vehicle_to_leader(cls, uav, vehicle):
        dx = uav.x - vehicle.x
        dy = uav.y - vehicle.y
        d_horz = math.sqrt(dx ** 2 + dy ** 2)
        d_horz = max(d_horz, 1.0)
        height_diff = uav.z - vehicle.z
        theta_rad = math.atan2(height_diff, d_horz)
        theta_deg = math.degrees(theta_rad)
        exponent = -cls.LOS_PROB_B * (theta_deg - cls.LOS_PROB_A)
        p_los = 1 / (1 + cls.LOS_PROB_A * math.exp(exponent))
        return p_los, theta_deg

    @classmethod
    def get_link_status_vehicle_to_leader(cls, vehicle, leader_uav):
        p_los, theta_deg = cls.get_los_probability_vehicle_to_leader(leader_uav, vehicle)
        p_nlos = 1 - p_los
        distance = cls.calculate_distance(vehicle, leader_uav)
        eff_distance = max(distance, 1.0)
        path_loss_component = cls.BETA_0 * (eff_distance ** -2)
        los_nlos_component = p_los + cls.ETA_NLOS * p_nlos
        gain_linear = path_loss_component * los_nlos_component
        gain_db = 10 * math.log10(gain_linear)
        snr = (gain_linear * cls.POWER_VEHICLE_WATT) / cls.NOISE_POWER_WATT
        transmission_rate_bps = cls.BANDWIDTH_HZ * math.log2(1 + snr)
        return {
            "distance": distance,
            "elevation_angle": theta_deg,
            "p_los": p_los,
            "p_nlos": p_nlos,
            "gain_db": gain_db,
            "rate_mbps": transmission_rate_bps / 1e6,
        }

    @staticmethod
    def calculate_transmission_delay(data_size_bits, rate_bps):
        if rate_bps is None or rate_bps <= 0:
            return float('inf')
        return data_size_bits / rate_bps

    @classmethod
    def calculate_vehicle_to_leader_delays(
            cls,
            vehicles: List[TaskVehicle],
            leader_uav: UAV,
            offload_ratios: Dict[int, float]
    ) -> (float, Dict[int, float], float):
        individual_delays = {}
        total_offloaded_data_mb = 0
        max_delay = 0.0

        for vehicle in vehicles:
            beta_k = offload_ratios.get(vehicle.vehicle_id, 0.0)

            vehicle_total_task_mb = vehicle.get_total_task_size_in_slot()
            vehicle_offloaded_mb = vehicle_total_task_mb * beta_k
            total_offloaded_data_mb += vehicle_offloaded_mb

            vehicle_offloaded_bits = vehicle_offloaded_mb * 8 * 1024 * 1024

            link_status = cls.get_link_status_vehicle_to_leader(vehicle, leader_uav)
            rate_bps = link_status['rate_mbps'] * 1_000_000

            delay = cls.calculate_transmission_delay(
                data_size_bits=vehicle_offloaded_bits,
                rate_bps=rate_bps
            )
            individual_delays[vehicle.vehicle_id] = delay

        if individual_delays:
            max_delay = max(individual_delays.values())

        return total_offloaded_data_mb, individual_delays, max_delay

    @classmethod
    def calculate_leader_to_member_delays(
            cls,
            leader_uav: UAV,
            member_uavs: List[UAV],
            total_offloaded_data_mb: float,
            total_member_capability_ghz: float
    ) -> (Dict[int, Dict[str, float]], float):
        delays_and_allocations = {}
        max_delay = 0.0

        if total_member_capability_ghz <= 0:
            return {}, 0.0

        for member in member_uavs:
            capability_proportion = member.computing_capability_ghz / total_member_capability_ghz
            allocated_data_mb = total_offloaded_data_mb * capability_proportion
            allocated_data_bits = allocated_data_mb * 8 * 1024 * 1024

            link_status = cls.get_link_status_leader_to_member(leader_uav, member)
            rate_bps = link_status['rate_mbps'] * 1_000_000

            transmission_delay = cls.calculate_transmission_delay(
                data_size_bits=allocated_data_bits,
                rate_bps=rate_bps
            )

            delays_and_allocations[member.uav_id] = {
                "allocated_mb": allocated_data_mb,
                "delay_sec": transmission_delay
            }

            if transmission_delay > max_delay:
                max_delay = transmission_delay

        return delays_and_allocations, max_delay
